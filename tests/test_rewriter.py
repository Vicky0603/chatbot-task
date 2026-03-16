from src.query.rewriter import rewrite_query


def test_rewrite_query_adds_synonyms():
    q = "When was the company founded?"
    rewritten, syns = rewrite_query(q)
    assert rewritten.lower().startswith(q.lower())
    assert any(s in rewritten.lower() for s in ["promtior", "promtior.ai"]) or syns

