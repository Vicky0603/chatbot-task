from langchain_core.documents import Document
from src.rerank.cross_encoder import CrossEncoderReranker


def test_cross_encoder_fallback_returns_docs_when_unavailable():
    r = CrossEncoderReranker(model_name="nonexistent-model-hopefully")
    docs = [Document(page_content="A"), Document(page_content="B")]
    out, scores = r.rerank("q", docs, top_k=2)
    assert len(out) == 2
    assert len(scores) == 2

