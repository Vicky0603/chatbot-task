from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Any, Tuple
import yaml

from langchain_community.document_loaders import WebBaseLoader, PyPDFLoader, TextLoader
import requests
from bs4 import BeautifulSoup
from hashlib import sha1
from src.config import settings
from src.ingestion.html_chunker import fetch_html, extract_sections

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
    # Extend with registry URLs if present
    urls = list(dict.fromkeys(urls + load_sources_registry()))

    loader = WebBaseLoader(urls)
    docs = loader.load()
    # Add DOM hash to metadata for freshness tracking
    for d in docs:
        u = d.metadata.get("source") or d.metadata.get("url")
        if not u:
            continue
        html, dom_hash = fetch_html(u)
        if dom_hash:
            d.metadata["dom_hash"] = dom_hash
        d.metadata.setdefault("title", d.metadata.get("title") or "")
    return docs


def discover_urls_from_sitemaps(base_urls: List[str]) -> List[str]:
    if not settings.enable_sitemap_discovery:
        return []
    found: List[str] = []
    sess = requests.Session()
    for base in base_urls:
        try:
            robots = base.rstrip("/") + "/robots.txt"
            r = sess.get(robots, timeout=10)
            if r.status_code != 200:
                continue
            for line in r.text.splitlines():
                if line.lower().startswith("sitemap:"):
                    sm_url = line.split(":", 1)[1].strip()
                    try:
                        sm = sess.get(sm_url, timeout=15)
                        if sm.status_code == 200:
                            soup = BeautifulSoup(sm.text, "xml")
                            for loc in soup.find_all("loc"):
                                url = (loc.text or "").strip()
                                if url:
                                    found.append(url)
                    except Exception:
                        continue
        except Exception:
            continue
    # dedupe
    return list(dict.fromkeys(found))


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

    # augment urls with sitemaps if enabled
    urls = list(dict.fromkeys(urls + discover_urls_from_sitemaps(["https://www.promtior.ai/"])) )
    web_docs = load_promtior_web_pages(urls)
    pdf_docs: List[Document] = []
    note_docs: List[Document] = []

    if include_presentation:
        pdf_docs = load_promtior_presentation()
        note_docs = load_local_text_notes()

    # Optional HTML-aware chunking
    if settings.enable_html_chunking:
        section_docs: List[Document] = []
        for d in web_docs:
            u = d.metadata.get("source") or d.metadata.get("url")
            if not u:
                continue
            html, _ = fetch_html(u)
            if not html:
                continue
            section_docs.extend(extract_sections(u, html))
        if section_docs:
            web_docs = section_docs

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
def load_sources_registry() -> List[str]:
    """Load optional YAML source registry at data/sources.yaml."""
    try:
        path = get_project_root() / "data" / "sources.yaml"
        if not path.exists():
            return []
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        urls = []
        for item in data.get("urls", []):
            if isinstance(item, str):
                urls.append(item)
            elif isinstance(item, dict) and item.get("url"):
                urls.append(item["url"])
        return urls
    except Exception:
        return []
