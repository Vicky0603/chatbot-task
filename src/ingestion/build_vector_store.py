from __future__ import annotations

from pathlib import Path
import time

from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.ingestion.load_promtior_site import get_promtior_documents
from src.config import settings


def get_project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def get_vectorstore_dir() -> Path:
    return get_project_root() / "data" / "vectorstore"


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
    batch_size = 8
    total = len(split_docs)
    added = 0
    for start in range(0, total, batch_size):
        batch = split_docs[start : start + batch_size]
        texts = [d.page_content for d in batch]
        metadatas = [getattr(d, "metadata", {}) for d in batch]

        retries = 0
        while True:
            try:
                vectorstore.add_texts(texts=texts, metadatas=metadatas)
                added += len(texts)
                print(f"Batch added: {start} - {start + len(texts)} (total added: {added}/{total})")
                break
            except Exception as e:
                msg = str(e).lower()
                # Si es un error de cuota/rate-limit -> fallar con mensaje instructivo
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
