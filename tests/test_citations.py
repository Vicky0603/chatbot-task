from langchain_core.documents import Document
from src.chains.rag_chain import _extract_citations


def test_extract_citations_dedupes_and_formats():
    docs = [
        Document(page_content="A", metadata={"source": "s1"}),
        Document(page_content="B", metadata={"source": "s1"}),
        Document(page_content="C", metadata={"url": "https://example.com"}),
    ]
    cites = _extract_citations(docs)
    sources = [c["source"] for c in cites]
    assert "s1" in sources
    assert "https://example.com" in sources
    # ensure no duplicates for same source
    assert sources.count("s1") == 1

