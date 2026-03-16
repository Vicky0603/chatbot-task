from __future__ import annotations

import os
import logging
from typing import List, Any, Optional, Tuple

logger = logging.getLogger("app.rerank")


class CrossEncoderReranker:
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2") -> None:
        self.model_name = model_name
        self._model = None

    def _lazy_load(self) -> bool:
        if self._model is not None:
            return True
        try:
            from sentence_transformers import CrossEncoder  # type: ignore
        except Exception as e:
            logger.info("CrossEncoder not available: %s", e)
            return False
        try:
            # Let sentence-transformers handle caching; offline will work if cache is present
            self._model = CrossEncoder(self.model_name)
            return True
        except Exception as e:
            logger.warning("Failed to load CrossEncoder '%s': %s", self.model_name, e)
            return False

    @classmethod
    def from_env(cls) -> Optional["CrossEncoderReranker"]:
        enabled = str(os.getenv("RERANKING_ENABLED", "true")).lower() == "true"
        if not enabled:
            return None
        name = os.getenv("RERANKER_MODEL_NAME", "cross-encoder/ms-marco-MiniLM-L-6-v2")
        inst = cls(model_name=name)
        if inst._lazy_load():
            logger.info("Cross-encoder reranking enabled: %s", name)
            return inst
        return None

    def rerank(self, query: str, docs: List[Any], top_k: int = 8) -> Tuple[List[Any], List[float]]:
        if not self._lazy_load():
            return docs, [0.0 for _ in docs]
        pairs = [(query, d.page_content) for d in docs]
        scores = self._model.predict(pairs)  # type: ignore[attr-defined]
        # Build list with scores attached
        indexed = list(zip(docs, scores))
        indexed.sort(key=lambda x: float(x[1]), reverse=True)
        reranked_docs = [d for d, _ in indexed[:top_k]]
        reranked_scores = [float(s) for _, s in indexed[:top_k]]
        # Write scores into metadata for downstream use
        for d, s in zip(reranked_docs, reranked_scores):
            try:
                d.metadata["rerank_score"] = float(s)
            except Exception:
                pass
        return reranked_docs, reranked_scores


class CohereReranker:
    def __init__(self, api_key: str, model: str = "rerank-english-v3.0") -> None:
        self.api_key = api_key
        self.model = model

    @classmethod
    def from_env(cls) -> Optional["CohereReranker"]:
        key = os.getenv("COHERE_API_KEY")
        if not key:
            return None
        return cls(api_key=key)

    def rerank(self, query: str, docs: List[Any], top_k: int = 8) -> Tuple[List[Any], List[float]]:
        import requests
        endpoint = "https://api.cohere.com/v1/rerank"
        contents = [d.page_content or "" for d in docs]
        payload = {
            "model": self.model,
            "query": query,
            "documents": contents,
            "top_n": min(top_k, len(contents)),
        }
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        try:
            resp = requests.post(endpoint, json=payload, headers=headers, timeout=15)
            resp.raise_for_status()
            data = resp.json()
            results = data.get("results", [])
            results.sort(key=lambda r: float(r.get("relevance_score", 0.0)), reverse=True)
            idxs = [int(r.get("index", 0)) for r in results]
            scores = [float(r.get("relevance_score", 0.0)) for r in results]
            reranked_docs = [docs[i] for i in idxs]
            for d, s in zip(reranked_docs, scores):
                try:
                    d.metadata["rerank_score"] = float(s)
                except Exception:
                    pass
            return reranked_docs, scores
        except Exception:
            return docs, [0.0 for _ in docs]
