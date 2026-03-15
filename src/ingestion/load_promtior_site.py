from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Any, Tuple

from langchain_community.document_loaders import WebBaseLoader, PyPDFLoader, TextLoader

Document = Any


# Promtior URLs to index
PROMTIOR_URLS = [
    "https://www.promtior.ai/",
    "https://www.promtior.ai/use-cases",
    "https://www.promtior.ai/service",
    "https://www.promtior.ai/contacto",
    "https://www.promtior.ai/blog",
]


def get_project_root() -> Path:
    """
    Return the project root assuming this file lives at src/ingestion/...
    """
    return Path(__file__).resolve().parents[2]


def get_raw_data_dir() -> Path:
    return get_project_root() / "data" / "raw"


def load_promtior_web_pages(urls: Optional[List[str]] = None) -> List[Document]:
    """
    Load Promtior web pages content using WebBaseLoader.
    """
    if urls is None:
        urls = PROMTIOR_URLS

    loader = WebBaseLoader(urls)
    docs = loader.load()
    return docs


def load_promtior_presentation() -> List[Document]:
    """
    Load Promtior presentation PDF (if any).
    Look for the expected filename, and if not found, fall back to any .pdf in data/raw.
    """
    raw_dir = get_raw_data_dir()
    expected_name = "AI Engineer-Tecnical-Test.pdf"
    pdf_path = raw_dir / expected_name

    if not raw_dir.exists():
        return []

    # Si existe el archivo con el nombre exacto, usarlo
    if pdf_path.exists():
        loader = PyPDFLoader(str(pdf_path))
        return loader.load()

    # Fallback: buscar cualquier PDF en data/raw
    pdf_files = list(raw_dir.glob("*.pdf"))
    if pdf_files:
        found = pdf_files[0]
        print(f"[debug] PDF found by fallback: {found}")
        loader = PyPDFLoader(str(found))
        return loader.load()

    return []


def load_local_text_notes() -> List[Document]:
    """
    Load any .md/.txt notes from data/raw to supplement the corpus with curated facts.
    Useful for ensuring key facts (e.g., founding date) are available.
    """
    raw_dir = get_raw_data_dir()
    docs: List[Document] = []
    if not raw_dir.exists():
        return docs
    for pattern in ("*.md", "*.txt"):
        for path in raw_dir.glob(pattern):
            try:
                docs.extend(TextLoader(str(path), encoding="utf-8").load())
            except Exception:
                # skip unreadable files
                continue
    return docs


def get_promtior_documents(
    extra_urls: Optional[List[str]] = None,
    include_presentation: bool = True,
) -> Tuple[List[Document], bool]:
    """
    Main function used by the rest of the code:
    - Load Promtior web pages
    - Optionally load the presentation PDF
    - Return a tuple: (documents, pdf_loaded_flag)
    """
    urls = PROMTIOR_URLS.copy()
    if extra_urls:
        urls.extend(extra_urls)

    web_docs = load_promtior_web_pages(urls)
    pdf_docs: List[Document] = []
    note_docs: List[Document] = []

    if include_presentation:
        pdf_docs = load_promtior_presentation()
        note_docs = load_local_text_notes()

    all_docs = web_docs + pdf_docs + note_docs

    # Normalize metadata if desired
    for d in all_docs:
        d.metadata.setdefault("source_type", "web_or_pdf")

    pdf_loaded = bool(pdf_docs)
    return all_docs, pdf_loaded


if __name__ == "__main__":
    docs, pdf_loaded = get_promtior_documents()
    print(f"Documents loaded: {len(docs)}")
    print(f"PDF loaded: {pdf_loaded}")
    if docs:
        print("Content example:\n")
        print(docs[0].page_content[:1000])
