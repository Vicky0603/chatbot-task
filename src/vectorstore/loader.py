from __future__ import annotations

from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from src.config import settings
import re

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
        return OpenAIEmbeddings(model="text-embedding-3-small", api_key=settings.openai_api_key)

    # Initialize OpenAI embeddings (will fail if no key)
    try:
        embeddings = make_openai()
        # quick connectivity/functionality check
        try:
            embeddings.embed_documents(["test"])
            settings.openai_usable = True
        except Exception:
            settings.openai_usable = False
            raise RuntimeError("Could not generate embeddings with OpenAI. Check OPENAI_API_KEY and connectivity.")
    except Exception as e:
        # Propagar error con mensaje claro
        raise

    # Create Chroma using the chosen embedding function
    vectorstore = Chroma(persist_directory=str(vector_dir), embedding_function=embeddings)

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
