from __future__ import annotations

import logging
from typing import Any

from src.config import settings

logger = logging.getLogger("app.metrics")

try:
    from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
except Exception:  # pragma: no cover
    Counter = None  # type: ignore
    Histogram = None  # type: ignore
    generate_latest = None  # type: ignore
    CONTENT_TYPE_LATEST = "text/plain"  # type: ignore


REQUEST_COUNT = None
REQUEST_LATENCY = None
RETRIEVAL_DOCS = None


def setup_metrics() -> None:
    global REQUEST_COUNT, REQUEST_LATENCY, RETRIEVAL_DOCS
    if not settings.metrics_enabled or Counter is None:
        return
    REQUEST_COUNT = Counter("app_requests_total", "Total requests", ["path", "method", "status"])
    REQUEST_LATENCY = Histogram("app_request_latency_seconds", "Request latency", ["path", "method"])
    RETRIEVAL_DOCS = Histogram("app_retrieval_docs", "Retrieved doc count", buckets=(0, 2, 4, 8, 12, 16))


def record_request(path: str, method: str, status: int, latency: float) -> None:
    if not settings.metrics_enabled or Counter is None:
        return
    try:
        REQUEST_COUNT.labels(path=path, method=method, status=str(status)).inc()
        REQUEST_LATENCY.labels(path=path, method=method).observe(latency)
    except Exception:
        pass


def record_retrieval_count(n: int) -> None:
    if not settings.metrics_enabled or Histogram is None:
        return
    try:
        RETRIEVAL_DOCS.observe(n)
    except Exception:
        pass


def render_metrics() -> tuple[bytes, str]:
    if not settings.metrics_enabled or generate_latest is None:
        return (b"", CONTENT_TYPE_LATEST)
    data = generate_latest()  # type: ignore
    return data, CONTENT_TYPE_LATEST  # type: ignore

