from __future__ import annotations

from typing import List, Any

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda, RunnableParallel
from langchain_openai import ChatOpenAI
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain_core.documents import Document
import openai


from src.config import settings
from src.vectorstore.loader import get_vectorstore

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


def _extract_citations(docs: List[Document]) -> list[dict]:
    cites = []
    seen = set()
    for d in docs:
        src = d.metadata.get("source") or d.metadata.get("url") or d.metadata.get("source_type") or "unknown"
        key = src
        if key in seen:
            continue
        seen.add(key)
        cites.append({
            "source": src,
            "preview": (d.page_content or "")[:200]
        })
    return cites

# Build a chain that returns both the answer and citations
_docs_and_q = {"docs": retriever, "question": RunnablePassthrough()}

_answer = (
    _docs_and_q
    | RunnableLambda(lambda x: {"context": _format_docs(x["docs"]), "question": x["question"]})
    | prompt
    | llm
)

_sources = _docs_and_q | RunnableLambda(lambda x: _extract_citations(x["docs"]))

rag_chain = RunnableParallel(answer=_answer, sources=_sources)
