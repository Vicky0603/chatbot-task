from __future__ import annotations

import logging
from typing import Any, Dict

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException


logger = logging.getLogger("app.errors")


def _make_error_payload(status: int, message: str, type_: str, detail: Any | None = None, request_id: str | None = None) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "error": {
            "status": status,
            "type": type_,
            "message": message,
        }
    }
    if detail is not None:
        payload["error"]["detail"] = detail
    if request_id is not None:
        payload["error"]["request_id"] = request_id
    return payload


def register_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(StarletteHTTPException)
    async def http_exception_handler(request: Request, exc: StarletteHTTPException):
        logger.warning("HTTPException: %s %s -> %s", request.method, request.url.path, exc.detail)
        return JSONResponse(
            status_code=exc.status_code,
            content=_make_error_payload(exc.status_code, str(exc.detail), "http_error", request_id=getattr(request.state, "request_id", None)),
        )

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError):
        logger.warning("ValidationError: %s %s -> %s", request.method, request.url.path, exc.errors())
        return JSONResponse(
            status_code=422,
            content=_make_error_payload(422, "Invalid request payload", "validation_error", detail=exc.errors(), request_id=getattr(request.state, "request_id", None)),
        )

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(request: Request, exc: Exception):
        logger.exception("Unhandled error on %s %s", request.method, request.url.path)
        return JSONResponse(
            status_code=500,
            content=_make_error_payload(500, "Internal server error", "internal_error", request_id=getattr(request.state, "request_id", None)),
        )


def make_json_error(status: int, type_: str, message: str, request_id: str | None = None, headers: Dict[str, str] | None = None) -> JSONResponse:
    return JSONResponse(status_code=status, content=_make_error_payload(status, message, type_, request_id=request_id), headers=headers or {})
