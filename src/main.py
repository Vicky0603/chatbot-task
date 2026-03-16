import logging
import os
from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from langserve import add_routes

from src.chains.rag_chain import rag_chain
from src.config import settings
from src.api.errors import register_error_handlers
from src.api.middleware import RequestIDMiddleware, RateLimitMiddleware, BasicAuthMiddleware
from src.logging_config import setup_logging
from src.telemetry.tracing import setup_tracing_from_env

setup_logging(settings.log_level)
setup_tracing_from_env()
logger = logging.getLogger("app")

app = FastAPI(title="Promtior RAG Chatbot")

# CORS: in production, restrict allow_origins to specific domains
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Correlation IDs
app.add_middleware(RequestIDMiddleware)

# Optional protection for production
if settings.basic_auth_enabled and settings.basic_auth_user and settings.basic_auth_pass:
    app.add_middleware(BasicAuthMiddleware, username=settings.basic_auth_user, password=settings.basic_auth_pass)

if settings.rate_limit_enabled:
    app.add_middleware(RateLimitMiddleware, limit_per_minute=settings.rate_limit_per_minute)

# simple request logging
@app.middleware("http")
async def log_requests(request, call_next):
    logger.info("%s %s", request.method, request.url.path)
    resp = await call_next(request)
    logger.info("%s %s -> %s", request.method, request.url.path, resp.status_code)
    return resp

# LangServe
add_routes(app, rag_chain, path="/promtior-rag")

# Centralized error handlers
register_error_handlers(app)

# Frontend
if os.path.isdir(settings.frontend_dir):
    app.mount("/static", StaticFiles(directory=settings.frontend_dir), name="static")

@app.get("/")
def index():
    idx = os.path.join(settings.frontend_dir, "index.html")
    # asegurar content-type correcto
    return FileResponse(idx, media_type="text/html")


@app.get("/healthz")
def healthz():
    vector_dir = settings.vectorstore_dir
    vs_exists = os.path.isdir(vector_dir) and any(True for _ in os.scandir(vector_dir))
    status = {
        "status": "ok",
        "openai": {
            "configured": bool(settings.openai_api_key),
            "model": settings.model_name,
        },
        "vectorstore": {
            "path": vector_dir,
            "exists": vs_exists,
        },
        "playground": "/promtior-rag/playground/",
    }
    return JSONResponse(status_code=200, content=status)
