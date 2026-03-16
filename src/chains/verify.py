from __future__ import annotations

from typing import List, Dict
import re


def sentences(text: str) -> List[str]:
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p for p in parts if p]


def groundedness_score(answer: str, docs: List[str]) -> tuple[float, List[Dict]]:
    if not answer or not docs:
        return 0.0, []
    ans_sents = sentences(answer)
    supports = []
    hit = 0
    for s in ans_sents:
        toks = set(t.lower() for t in re.findall(r"[a-zA-Z0-9]{3,}", s))
        if not toks:
            continue
        found_doc = None
        for i, d in enumerate(docs):
            dtoks = set(t.lower() for t in re.findall(r"[a-zA-Z0-9]{3,}", d))
            inter = toks & dtoks
            if len(inter) >= max(2, int(0.2 * len(toks))):
                found_doc = i
                break
        if found_doc is not None:
            hit += 1
            supports.append({"sentence": s, "doc_index": found_doc})
    score = hit / max(1, len(ans_sents))
    return float(score), supports

