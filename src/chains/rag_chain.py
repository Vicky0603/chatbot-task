from __future__ import annotations

from typing import List, Any, Tuple

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda, RunnableParallel
from langchain_openai import ChatOpenAI
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
import openai


from src.config import settings
from src.vectorstore.loader import get_vectorstore
from src.query.rewriter import rewrite_query
from src.rerank.cross_encoder import CrossEncoderReranker, CohereReranker
from src.query.classifier import classify, retrieval_params
from src.chains.verify import groundedness_score

Document = Any


def _format_docs(docs: List[Document]) -> str:
    """
    Join the retrieved documents' content into a single string.
    """
    return "\n\n".join(doc.page_content for doc in docs)


_vectorstore = None
_bm25_retriever = None

def _ensure_vectorstore():
    global _vectorstore, _bm25_retriever
    if _vectorstore is None:
        _vectorstore = get_vectorstore()
    if _bm25_retriever is None:
        _bm25_retriever = _build_bm25_from_chroma(_vectorstore)

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

# Vector retriever: created per-call to support dynamic k


# We'll do hybrid retrieval manually to allow dynamic k/weights per query
def _hybrid_retrieve(query: str, k_vec: int, k_bm25: int, weights: tuple[float, float]) -> List[Document]:
    _ensure_vectorstore()
    # Create a temporary retriever with dynamic k for vectors
    try:
        vec_r = _vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": int(k_vec), "fetch_k": max(24, int(k_vec) * 3)})
        vec_docs = vec_r.invoke(query)  # type: ignore[assignment]
    except Exception:
        vec_docs = []
    if _bm25_retriever is not None:
        bm_docs = _bm25_retriever.get_relevant_documents(query)[:k_bm25]
    else:
        bm_docs = []
    # Simple weighted interleave: score=rank-based
    scored = []
    for i, d in enumerate(vec_docs):
        scored.append((d, (k_vec - i) / max(k_vec, 1) * weights[1]))
    for i, d in enumerate(bm_docs):
        scored.append((d, (k_bm25 - i) / max(k_bm25, 1) * weights[0]))
    # Deduplicate by source+chunk_id
    seen = set()
    merged = []
    for d, s in sorted(scored, key=lambda x: x[1], reverse=True):
        key = (d.metadata.get("chunk_id") or d.metadata.get("source") or id(d))
        if key in seen:
            continue
        seen.add(key)
        merged.append(d)
        if len(merged) >= max(k_vec, k_bm25, 8):
            break
    return merged

# Optional cross-encoder reranker (local small model or cached)
_reranker: Any = None
_provider = (settings.rerank_provider or "none").lower()
if _provider == "cohere" and settings.cohere_api_key:
    _reranker = CohereReranker.from_env()
elif _provider == "local":
    _reranker = CrossEncoderReranker.from_env()
else:
    _reranker = None

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
llm_fallback = None
if settings.fallback_model_name and settings.fallback_model_name != settings.model_name:
    llm_fallback = ChatOpenAI(model=settings.fallback_model_name, api_key=settings.openai_api_key, temperature=0)


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
            "fragment": d.metadata.get("fragment"),
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


_CACHE: dict = {}


def _cache_key(query: str, docs: List[Document]) -> str:
    ids = []
    for d in docs:
        cid = d.metadata.get("chunk_id") or (
            (d.metadata.get("source") or "unknown") + ":" + (d.page_content[:50] if d.page_content else "")
        )
        ids.append(str(cid))
    return query + "::" + "|".join(ids[:8])


# Build a chain that returns both the answer and citations with query rewriting and reranking
_pre = RunnableLambda(lambda q: (lambda rq: {"question": q, "query": rq[0], "query_expansion": rq[1]})(rewrite_query(q)))

def _retrieve(x: dict) -> List[Document]:
    qc = classify(x["question"])
    params = retrieval_params(qc)
    return _hybrid_retrieve(x["query"], params["k_vec"], params["k_bm25"], params["weights"])  # type: ignore[index]

def _maybe_rerank(x: dict) -> Tuple[List[Document], List[float]]:
    docs = x["docs"]
    if _reranker is None:
        return docs, []
    reranked, scores = _reranker.rerank(x["query"], docs, top_k=8)
    return reranked, scores

_docs = _pre | RunnableLambda(lambda x: {**x, "docs": _retrieve(x)})
_docs = _docs | RunnableLambda(lambda x: (lambda rr: {**x, "docs": rr[0], "scores": rr[1]})(_maybe_rerank(x)))

def _choose_llm(scores: List[float]):
    conf = _confidence_from_scores(scores)
    if conf < settings.confidence_threshold and llm_fallback is not None:
        return llm_fallback
    return llm

_answer = (
    _docs
    | RunnableLambda(lambda x: {**x, "cache_key": _cache_key(x["query"], x["docs"])})
    | RunnableLambda(lambda x: (x if _CACHE.get(x["cache_key"]) is None else {**x, "cached_answer": _CACHE[x["cache_key"]]}))
    | RunnableLambda(lambda x: (x if "cached_answer" in x else {**x, "context": _format_docs(x["docs"]), "question_text": x["question"], "_llm": _choose_llm(x.get("scores") or [])}))
    | RunnableLambda(lambda x: (x if "cached_answer" in x else {**x, "answer_obj": (prompt | x["_llm"]).invoke({"context": x["context"], "question": x["question_text"]})}))
    | RunnableLambda(lambda x: (x if "cached_answer" in x else {**x, "answer": getattr(x["answer_obj"], "content", str(x["answer_obj"]))}))
    | RunnableLambda(lambda x: (x if "cached_answer" in x else (_CACHE.setdefault(x["cache_key"], x["answer"]) or x)))
    | RunnableLambda(lambda x: (x["cached_answer"] if "cached_answer" in x else x["answer"]))
)

_sources = _docs | RunnableLambda(lambda x: _extract_citations(x["docs"], x["question"]))

_confidence = _docs | RunnableLambda(lambda x: _confidence_from_scores(x.get("scores") or []))


def _apply_hallucination_guard(payload: dict) -> dict:
    conf = float(payload.get("confidence") or 0.0)
    num_src = len(payload.get("sources") or [])
    if num_src < settings.min_sources_required or conf < settings.confidence_threshold:
        return {
            **payload,
            "answer": "I don't have enough grounded context to answer confidently. Could you clarify or provide more details?",
        }
    return payload

_base = RunnableParallel(
    answer=_answer,
    sources=_sources,
    rewritten_query=_docs | RunnableLambda(lambda x: x.get("query")),
    confidence=_confidence,
    grounded=_docs | RunnableLambda(lambda x: groundedness_score("", [d.page_content for d in x["docs"]])),
)

rag_chain = _base | RunnableLambda(_apply_hallucination_guard)
