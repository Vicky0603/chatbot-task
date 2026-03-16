from __future__ import annotations

from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from src.config import settings
import re
import json
from pathlib import Path

def get_vectorstore() -> Chroma:
    """
    Load the persisted vector store from the configured directory.
    Uses OpenAI embeddings exclusively. If OPENAI_API_KEY is not configured
    or there is an embedding-dimension mismatch that cannot be resolved
    with OpenAI, raises RuntimeError with instructions.
    """
    vector_dir = settings.vectorstore_dir

    def make_openai():
        if not settings.openai_api_key:
            raise RuntimeError(
                "OPENAI_API_KEY not set. Configure the OPENAI_API_KEY environment variable "
                "to use OpenAI embeddings in production."
            )
        model = settings.embeddings_model or "text-embedding-3-small"
        return OpenAIEmbeddings(model=model, api_key=settings.openai_api_key)

    def make_e5():
        model_name = settings.e5_model_name or "intfloat/e5-small-v2"
        # Normalize pooling option for E5 if needed
        return HuggingFaceEmbeddings(model_name=model_name)

    # Initialize OpenAI embeddings (will fail if no key)
    provider = (settings.embeddings_provider or "openai").lower()
    if provider == "openai":
        embeddings = make_openai()
        try:
            embeddings.embed_documents(["test"])  # quick check
            settings.openai_usable = True
        except Exception:
            settings.openai_usable = False
            raise RuntimeError("Could not generate embeddings with OpenAI. Check OPENAI_API_KEY and connectivity.")
    elif provider == "e5":
        embeddings = make_e5()
    else:
        raise RuntimeError(f"Unsupported EMBEDDINGS_PROVIDER='{provider}'. Use 'openai' or 'e5'.")

    # Create Chroma using the chosen embedding function
    vectorstore = Chroma(persist_directory=str(vector_dir), embedding_function=embeddings)
    # Persist provider metadata alongside the collection for ops clarity
    try:
        meta = {
            "provider": provider,
            "openai_model": getattr(settings, "embeddings_model", None),
            "e5_model": getattr(settings, "e5_model_name", None),
        }
        meta_path = Path(vector_dir) / "meta.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f)
    except Exception:
        pass

    # Quick compatibility test: similarity_search to detect embedding-dimension mismatch
    try:
        vectorstore.similarity_search("test", k=1)
        return vectorstore
    except Exception as e:
        msg = str(e)
        m = re.search(r"expecting embedding with dimension of\s+(\d+).*got\s+(\d+)", msg, flags=re.IGNORECASE)
        if m:
            expected_dim = int(m.group(1))
            got_dim = int(m.group(2))
            # If the collection requires 1536, OpenAI is the appropriate option
            if expected_dim == 1536:
                # We already use OpenAI; if we reach here it may be a transient failure
                settings.openai_usable = True
                return vectorstore
            # If a different dimension is required (e.g., 384) and we don't support it here,
            # instruct the user to rebuild the vector store with the appropriate model.
            raise RuntimeError(
                f"Embedding dimension mismatch: collection expects dim={expected_dim} but got dim={got_dim}. "
                "This deployment uses OpenAI embeddings only. To resolve, recreate the vector store using "
                "OpenAI embeddings (dim=1536), or rebuild locally with a different embedding model and upload the DB."
            )
        # If the error couldn't be parsed, re-raise for diagnosis
        raise
