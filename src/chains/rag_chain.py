from __future__ import annotations

from typing import List, Any, Tuple

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda, RunnableParallel
from langchain_openai import ChatOpenAI
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain_core.documents import Document
import openai


from src.config import settings
from src.vectorstore.loader import get_vectorstore
from src.query.rewriter import rewrite_query
from src.rerank.cross_encoder import CrossEncoderReranker

Document = Any


def _format_docs(docs: List[Document]) -> str:
    """
    Join the retrieved documents' content into a single string.
    """
    return "\n\n".join(doc.page_content for doc in docs)


# Load vector store and create retriever
_vectorstore = get_vectorstore()

def _build_bm25_from_chroma(vs) -> BM25Retriever | None:
    try:
        coll = vs._collection  # chromadb collection
        data = coll.get(include=["documents", "metadatas"])  # type: ignore[attr-defined]
        docs = []
        for txt, meta in zip(data.get("documents", []) or [], data.get("metadatas", []) or []):
            if not txt:
                continue
            docs.append(Document(page_content=txt, metadata=meta or {}))
        if not docs:
            return None
        return BM25Retriever.from_documents(docs)
    except Exception:
        return None

# Vector retriever (MMR)
vector_retriever = _vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 8, "fetch_k": 24})

# BM25 retriever built from stored documents (if available)
bm25_retriever = _build_bm25_from_chroma(_vectorstore)

# Hybrid retriever (if BM25 is available), else fallback to vector-only
retriever = (
    EnsembleRetriever(retrievers=[bm25_retriever, vector_retriever], weights=[0.5, 0.5])
    if bm25_retriever is not None
    else vector_retriever
)

# Optional cross-encoder reranker (local small model or cached)
_reranker = CrossEncoderReranker.from_env()

# Define prompt for RAG
prompt = ChatPromptTemplate.from_template(
    """
You are a helpful assistant that answers questions ONLY using the context provided.
If the answer is not in the context, say that you don't know and suggest rephrasing the question.

Context:
{context}

Question:
{question}

Answer in a clear and concise way.
"""
)

# LLM
llm = ChatOpenAI(
    model=settings.model_name,
    api_key=settings.openai_api_key,
    temperature=0,
)


def _highlight_preview(preview: str, query: str, max_len: int = 200) -> str:
    text = (preview or "")[: max_len]
    q = (query or "").lower()
    # Tokenize on words and pick unique keywords of length >= 3
    import re as _re
    toks = [t for t in _re.findall(r"[a-zA-Z0-9]+", q) if len(t) >= 3]
    toks = list(dict.fromkeys(toks))
    if not toks:
        return text
    # Replace occurrences with <mark>
    def repl(m):
        return f"<mark>{m.group(0)}</mark>"
    for t in toks:
        try:
            text = _re.sub(rf"(?i)\b{_re.escape(t)}\b", repl, text)
        except Exception:
            continue
    return text


def _extract_citations(docs: List[Document], query: str = "") -> list[dict]:
    cites = []
    seen = set()
    for d in docs:
        src = d.metadata.get("source") or d.metadata.get("url") or d.metadata.get("source_type") or "unknown"
        url = d.metadata.get("url") or (src if isinstance(src, str) and src.startswith("http") else None)
        key = src
        if key in seen:
            continue
        seen.add(key)
        cites.append({
            "source": src,
            "url": url,
            "preview": (d.page_content or "")[:200],
            "preview_html": _highlight_preview(d.page_content or "", query),
            "rerank_score": d.metadata.get("rerank_score"),
        })
    return cites

def _confidence_from_scores(scores: List[float]) -> float:
    if not scores:
        return 0.0
    # Normalize via sigmoid then average top-3
    import math
    sig = [1.0 / (1.0 + math.exp(-float(s))) for s in scores]
    sig.sort(reverse=True)
    k = min(3, len(sig))
    return float(sum(sig[:k]) / k)


# Build a chain that returns both the answer and citations with query rewriting and reranking
_pre = RunnableLambda(lambda q: (lambda rq: {"question": q, "query": rq[0], "query_expansion": rq[1]})(rewrite_query(q)))

def _retrieve(x: dict) -> List[Document]:
    return retriever.get_relevant_documents(x["query"])  # type: ignore[attr-defined]

def _maybe_rerank(x: dict) -> Tuple[List[Document], List[float]]:
    docs = x["docs"]
    if _reranker is None:
        return docs, []
    reranked, scores = _reranker.rerank(x["query"], docs, top_k=8)
    return reranked, scores

_docs = _pre | RunnableLambda(lambda x: {**x, "docs": _retrieve(x)})
_docs = _docs | RunnableLambda(lambda x: (lambda rr: {**x, "docs": rr[0], "scores": rr[1]})(_maybe_rerank(x)))

_answer = (
    _docs
    | RunnableLambda(lambda x: {"context": _format_docs(x["docs"]), "question": x["question"]})
    | prompt
    | llm
)

_sources = _docs | RunnableLambda(lambda x: _extract_citations(x["docs"], x["question"]))

_confidence = _docs | RunnableLambda(lambda x: _confidence_from_scores(x.get("scores") or []))

rag_chain = RunnableParallel(
    answer=_answer,
    sources=_sources,
    rewritten_query=_docs | RunnableLambda(lambda x: x.get("query")),
    confidence=_confidence,
)
