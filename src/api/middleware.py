from __future__ import annotations

import base64
import time
import uuid
from collections import deque, defaultdict
from typing import Callable, Deque, Dict, Optional

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

from .errors import make_json_error


class RequestIDMiddleware(BaseHTTPMiddleware):
    """Attach a correlation ID to every request/response via X-Request-ID."""

    def __init__(self, app: ASGIApp, header_name: str = "X-Request-ID") -> None:
        super().__init__(app)
        self.header_name = header_name

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        req_id = request.headers.get(self.header_name) or str(uuid.uuid4())
        request.state.request_id = req_id
        response = await call_next(request)
        response.headers[self.header_name] = req_id
        return response


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Simple in-memory rate limiter per client IP.

    Sliding window: 60 seconds. Limit is configurable via constructor.
    Excludes health and static paths.
    """

    def __init__(self, app: ASGIApp, limit_per_minute: int = 60) -> None:
        super().__init__(app)
        self.limit = max(1, int(limit_per_minute))
        self.hits: Dict[str, Deque[float]] = defaultdict(deque)

    def _excluded(self, path: str) -> bool:
        return path.startswith("/healthz") or path.startswith("/static")

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        if self._excluded(request.url.path):
            return await call_next(request)

        now = time.time()
        ip = (request.client.host if request.client else "unknown")
        q = self.hits[ip]
        # drop entries older than 60s
        cutoff = now - 60
        while q and q[0] < cutoff:
            q.popleft()

        if len(q) >= self.limit:
            req_id = getattr(request.state, "request_id", None)
            return make_json_error(429, "rate_limit", "Too many requests", request_id=req_id)

        q.append(now)
        return await call_next(request)


class BasicAuthMiddleware(BaseHTTPMiddleware):
    """Optional Basic Auth protection for all routes except healthz/static."""

    def __init__(self, app: ASGIApp, username: str, password: str) -> None:
        super().__init__(app)
        self.username = username
        self.password = password

    def _excluded(self, path: str) -> bool:
        return path.startswith("/healthz") or path.startswith("/static")

    @staticmethod
    def _parse_basic_auth(header: str) -> Optional[tuple[str, str]]:
        try:
            scheme, b64 = header.split(" ", 1)
            if scheme.lower() != "basic":
                return None
            raw = base64.b64decode(b64).decode("utf-8")
            if ":" not in raw:
                return None
            user, pwd = raw.split(":", 1)
            return user, pwd
        except Exception:
            return None

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        if self._excluded(request.url.path):
            return await call_next(request)

        auth = request.headers.get("Authorization")
        creds = self._parse_basic_auth(auth) if auth else None
        if not creds or creds[0] != self.username or creds[1] != self.password:
            req_id = getattr(request.state, "request_id", None)
            return make_json_error(
                401,
                "unauthorized",
                "Authentication required",
                request_id=req_id,
                headers={"WWW-Authenticate": "Basic realm=\"Restricted\""},
            )
        return await call_next(request)

