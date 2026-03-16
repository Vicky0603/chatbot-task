from __future__ import annotations

from pathlib import Path
import time

from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.ingestion.load_promtior_site import get_promtior_documents
from src.config import settings
import hashlib
from typing import Dict, List, Tuple


def get_project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def get_vectorstore_dir() -> Path:
    return get_project_root() / "data" / "vectorstore"


def _chunk_id(source: str, chunk_text: str, idx: int) -> str:
    h = hashlib.sha1(chunk_text.encode("utf-8", errors="ignore")).hexdigest()[:16]
    return f"{source}#chunk{idx}:{h}"


def build_vector_store() -> None:
    """
    - Load Promtior documents (web pages and optional PDF)
    - Chunk them
    - Create and persist a Chroma vector store
    """
    print("Loading Promtior documents...")
    docs, pdf_loaded = get_promtior_documents()
    print(f"Total original documents: {len(docs)}")
    print(f"PDF loaded: {pdf_loaded}")

    if not docs:
        print("No documents to process.")
        return

    # Chunking
    try:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=200,
            length_function=len,
        )
    except Exception as e:
        print("Error initializing text splitter:", e)
        raise

    split_docs = text_splitter.split_documents(docs)
    print(f"Total chunks after split: {len(split_docs)}")

    # OpenAI embeddings required (no sentence-transformers fallback)
    if not settings.openai_api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. This script requires OpenAI embeddings. "
            "Set OPENAI_API_KEY or modify the loader to use local embeddings."
        )

    try:
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small", api_key=settings.openai_api_key)
        print("Using OpenAIEmbeddings.")
    except Exception as e:
        print("OpenAIEmbeddings failed to initialize:", e)
        raise RuntimeError("Failed to initialize OpenAIEmbeddings. Check OPENAI_API_KEY and connectivity.") from e

    vectorstore_dir = get_vectorstore_dir()
    vectorstore_dir.mkdir(parents=True, exist_ok=True)
    print(f"Building and persisting vector store at: {vectorstore_dir}")

    # Initialize Chroma and add in batches
    vectorstore = Chroma(persist_directory=str(vectorstore_dir), embedding_function=embeddings)
    collection = vectorstore._collection  # type: ignore[attr-defined]

    # Precompute deterministic IDs and group by source
    chunk_texts: List[str] = []
    chunk_metas: List[dict] = []
    chunk_ids: List[str] = []
    by_source: Dict[str, List[str]] = {}
    for i, d in enumerate(split_docs):
        meta = dict(getattr(d, "metadata", {}) or {})
        src = meta.get("source") or meta.get("url") or meta.get("source_type") or "unknown"
        src = str(src)
        cid = _chunk_id(src, d.page_content or "", i)
        meta["source"] = src
        meta["chunk_index"] = i
        meta["chunk_id"] = cid
        chunk_texts.append(d.page_content)
        chunk_metas.append(meta)
        chunk_ids.append(cid)
        by_source.setdefault(src, []).append(cid)

    # Determine which IDs already exist to skip
    existing_ids: List[str] = []
    try:
        got = collection.get(ids=chunk_ids, include=["metadatas"])  # type: ignore[attr-defined]
        existing_ids = list(got.get("ids", []) or [])
    except Exception:
        existing_ids = []

    to_add_texts: List[str] = []
    to_add_metas: List[dict] = []
    to_add_ids: List[str] = []
    for t, m, i in zip(chunk_texts, chunk_metas, chunk_ids):
        if i in existing_ids:
            continue
        to_add_texts.append(t)
        to_add_metas.append(m)
        to_add_ids.append(i)

    batch_size = 8
    total = len(to_add_texts)
    added = 0
    for start in range(0, total, batch_size):
        texts = to_add_texts[start : start + batch_size]
        metadatas = to_add_metas[start : start + batch_size]
        ids = to_add_ids[start : start + batch_size]

        retries = 0
        while True:
            try:
                vectorstore.add_texts(texts=texts, metadatas=metadatas, ids=ids)
                added += len(texts)
                print(f"Batch added: {start} - {start + len(texts)} (total added: {added}/{total})")
                break
            except Exception as e:
                msg = str(e).lower()
                if "insufficient_quota" in msg or "quota" in msg or "429" in msg or "rate limit" in msg:
                    raise RuntimeError(
                        "OpenAI quota/rate-limit error while adding embeddings. "
                        "Check OPENAI_API_KEY, billing and quota limits."
                    ) from e

                retries += 1
                if retries > 5:
                    print("Max retries reached while adding batch:", e)
                    raise
                sleep = 2 ** retries
                print(f"Error adding batch, retrying in {sleep}s... (attempt {retries}) - error: {e}")
                time.sleep(sleep)

    # Cleanup: remove stale chunks per source
    removed_total = 0
    try:
        for src, new_ids in by_source.items():
            try:
                existing = collection.get(where={"source": src}, include=["metadatas"])  # type: ignore[attr-defined]
                old_ids = set(existing.get("ids", []) or [])
            except Exception:
                old_ids = set()
            stale = list(old_ids.difference(new_ids))
            if stale:
                try:
                    collection.delete(ids=stale)  # type: ignore[attr-defined]
                    removed_total += len(stale)
                except Exception:
                    pass
        if removed_total:
            print(f"Removed stale chunks: {removed_total}")
    except Exception as e:
        print("Cleanup step failed:", e)

    # Persistir
    try:
        vectorstore.persist()
        print("Vector store persisted successfully.")
        print(f"Chunks added: {added}")
    except Exception as e:
        print("Error persisting vector store:", e)
        raise

    print("Vector store created and persisted successfully.")


if __name__ == "__main__":
    build_vector_store()
