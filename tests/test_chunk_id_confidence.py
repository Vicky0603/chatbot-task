from src.ingestion.build_vector_store import _chunk_id
from src.chains.rag_chain import _confidence_from_scores


def test_chunk_id_stability():
    a = _chunk_id("src", "hello world", 1)
    b = _chunk_id("src", "hello world", 1)
    assert a == b


def test_confidence_mapping():
    conf = _confidence_from_scores([0.1, 1.0, 3.0])
    assert 0.0 <= conf <= 1.0
    assert conf > 0.0

