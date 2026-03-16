from __future__ import annotations

import os
import json
import re
from typing import Dict, List, Tuple


def _default_synonyms() -> Dict[str, List[str]]:
    # Simple, conservative synonyms tailored to Promtior domain.
    return {
        "company": ["promtior", "promtior.ai"],
        "services": ["offerings", "capabilities"],
        "use case": ["use-case", "use cases", "examples"],
        "found": ["founded", "foundation", "established"],
        "contact": ["reach", "email", "phone"],
    }


def _load_synonyms_from_env() -> Dict[str, List[str]]:
    raw = os.getenv("QUERY_SYNONYMS_JSON")
    if not raw:
        return {}
    try:
        data = json.loads(raw)
        if isinstance(data, dict):
            # normalize to list[str]
            out: Dict[str, List[str]] = {}
            for k, v in data.items():
                if isinstance(v, list):
                    out[str(k).lower()] = [str(x).lower() for x in v]
                else:
                    out[str(k).lower()] = [str(v).lower()]
            return out
    except Exception:
        pass
    return {}


_SYNONYMS = _default_synonyms()
_SYNONYMS.update(_load_synonyms_from_env())


def normalize(text: str) -> str:
    t = text.strip()
    t = re.sub(r"\s+", " ", t)
    return t


def expand_with_synonyms(query: str) -> List[str]:
    q = query.lower()
    exp: List[str] = []
    for key, syns in _SYNONYMS.items():
        if key in q:
            exp.extend(syns)
    # Deduplicate while keeping order
    seen = set()
    out = []
    for x in exp:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def rewrite_query(query: str) -> Tuple[str, List[str]]:
    """
    Normalize and expand vague terms with domain synonyms.
    Returns (rewritten_query, added_terms)
    """
    base = normalize(query)
    syns = expand_with_synonyms(base)
    if syns:
        extra = " ".join(f"{s}" for s in syns)
        rewritten = f"{base} {extra}"
    else:
        rewritten = base
    return rewritten, syns

