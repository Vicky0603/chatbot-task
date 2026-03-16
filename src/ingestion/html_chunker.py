from __future__ import annotations

import hashlib
from typing import List, Any

import requests
from bs4 import BeautifulSoup
from langchain_core.documents import Document


def fetch_html(url: str) -> tuple[str, str | None]:
    try:
        r = requests.get(url, timeout=15)
        r.raise_for_status()
        html = r.text
        dom_hash = hashlib.sha1(html.encode("utf-8", errors="ignore")).hexdigest()
        return html, dom_hash
    except Exception:
        return "", None


def extract_sections(url: str, html: str) -> List[Document]:
    docs: List[Document] = []
    try:
        soup = BeautifulSoup(html, "lxml")
        title = (soup.title.string if soup.title else "").strip() if soup else ""
        # heuristic main content
        main = soup.find("main") or soup.find("article") or soup.body
        if not main:
            return []
        # Collect sections by h1/h2
        current_title = title
        buffer = []
        anchors = []
        for el in main.descendants:
            if el.name in ("h1", "h2"):
                # flush previous
                if buffer:
                    text = "\n".join(buffer).strip()
                    if text:
                        frag = anchors[-1] if anchors else None
                        meta = {"source": url, "url": url, "title": title, "h1": current_title, "fragment": frag}
                        docs.append(Document(page_content=text, metadata=meta))
                    buffer = []
                current_title = el.get_text(" ", strip=True)
                # create a fragment id
                frag = (el.get("id") or current_title.lower().replace(" ", "-"))
                anchors.append(f"#{frag}")
            elif hasattr(el, "get_text") and el.name in ("p", "li"):
                txt = el.get_text(" ", strip=True)
                if txt:
                    buffer.append(txt)
        # flush tail
        if buffer:
            text = "\n".join(buffer).strip()
            if text:
                frag = anchors[-1] if anchors else None
                meta = {"source": url, "url": url, "title": title, "h1": current_title, "fragment": frag}
                docs.append(Document(page_content=text, metadata=meta))
    except Exception:
        return []
    return docs

