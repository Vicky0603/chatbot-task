from __future__ import annotations

import re
from typing import Literal

QueryClass = Literal["navigational", "faq", "comparison", "other"]


def classify(query: str) -> QueryClass:
    q = (query or "").lower().strip()
    if not q:
        return "other"
    if re.search(r"\b(contact|address|phone|email|site|website|homepage)\b", q):
        return "navigational"
    if re.search(r"\b(how|what|when|where|why|who)\b", q):
        return "faq"
    if re.search(r"\b(vs\.|versus|compare|difference between)\b", q):
        return "comparison"
    return "other"


def retrieval_params(qc: QueryClass) -> dict:
    if qc == "navigational":
        return {"k_vec": 6, "k_bm25": 10, "weights": (0.4, 0.6)}
    if qc == "faq":
        return {"k_vec": 8, "k_bm25": 8, "weights": (0.5, 0.5)}
    if qc == "comparison":
        return {"k_vec": 10, "k_bm25": 10, "weights": (0.6, 0.4)}
    return {"k_vec": 8, "k_bm25": 8, "weights": (0.5, 0.5)}

