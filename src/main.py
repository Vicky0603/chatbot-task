import logging
import os
from fastapi import FastAPI, Request, Header
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from langserve import add_routes
from starlette.responses import StreamingResponse

from src.chains.rag_chain import rag_chain
from src.config import settings
from src.api.errors import register_error_handlers
from src.api.middleware import RequestIDMiddleware, RateLimitMiddleware, BasicAuthMiddleware
from src.logging_config import setup_logging
from src.telemetry.tracing import setup_tracing_from_env
from src.telemetry.otel import setup_otel_from_env
from src.metrics import setup_metrics, record_request, render_metrics
from time import perf_counter
import json
from typing import List, Dict
from src.query.rewriter import rewrite_query
from src.query.classifier import classify, retrieval_params
from src.chains import rag_chain as rc
import asyncio
import os
from src.ingestion.build_vector_store import build_vector_store
from src.chains.verify import groundedness_score

setup_logging(settings.log_level)
setup_tracing_from_env()
setup_metrics()
setup_otel_from_env()
logger = logging.getLogger("app")

app = FastAPI(title="Promtior RAG Chatbot")
try:
    from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor  # type: ignore
    if os.getenv("OTEL_ENABLED", "false").lower() == "true":
        FastAPIInstrumentor.instrument_app(app)
except Exception:
    pass

# CORS: in production, restrict allow_origins to specific domains
origins = [o.strip() for o in (settings.allowed_origins or "*").split(",")]
app.add_middleware(CORSMiddleware, allow_origins=origins, allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

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
    start = perf_counter()
    logger.info("%s %s", request.method, request.url.path)
    resp = await call_next(request)
    dur = perf_counter() - start
    logger.info("%s %s -> %s (%.3fs)", request.method, request.url.path, resp.status_code, dur)
    try:
        record_request(str(request.url.path), request.method, resp.status_code, dur)
    except Exception:
        pass
    return resp


@app.middleware("http")
async def security_headers(request: Request, call_next):
    resp = await call_next(request)
    resp.headers["Content-Security-Policy"] = settings.content_security_policy
    return resp


@app.middleware("http")
async def api_key_guard(request: Request, call_next):
    if settings.api_key and str(request.url.path).startswith("/promtior-rag"):
        key = request.headers.get("x-api-key") or request.headers.get("X-API-Key")
        if key != settings.api_key:
            return JSONResponse(status_code=401, content={"error": {"type": "unauthorized", "status": 401, "message": "Invalid API key"}})
    # Optional per-route scoped keys
    if settings.route_api_keys_json:
        try:
            mapping = json.loads(settings.route_api_keys_json)
            for prefix, k in mapping.items():
                if str(request.url.path).startswith(prefix):
                    key = request.headers.get("x-api-key") or request.headers.get("X-API-Key")
                    if key != k:
                        return JSONResponse(status_code=401, content={"error": {"type": "unauthorized", "status": 401, "message": "Invalid scope key"}})
        except Exception:
            pass
    return await call_next(request)

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
        "indexing": bool(getattr(app.state, "indexing", False)),
        "playground": "/promtior-rag/playground/",
    }
    return JSONResponse(status_code=200, content=status)


@app.post("/feedback")
async def feedback(payload: dict, request: Request, x_request_id: str | None = Header(default=None)):
    # Minimal sink: log feedback with request_id and rating/comment
    rid = x_request_id or getattr(request.state, "request_id", None)
    rating = payload.get("rating")  # up|down|neutral
    comment = payload.get("comment")
    logger.info("feedback rid=%s rating=%s comment=%s", rid, rating, (comment or "").replace("\n", " ")[:200])
    return {"ok": True}


@app.get("/metrics")
def metrics():
    data, ct = render_metrics()
    return Response(content=data, media_type=ct)


def build_context(question: str) -> Dict:
    rq, _ = rewrite_query(question)
    qc = classify(question)
    params = retrieval_params(qc)
    docs = rc._hybrid_retrieve(rq, params["k_vec"], params["k_bm25"], params["weights"])  # type: ignore
    # optional rerank
    scores: List[float] = []
    if rc._reranker is not None:
        docs, scores = rc._reranker.rerank(rq, docs, top_k=max(params["k_vec"], params["k_bm25"]))  # type: ignore
    ctx = rc._format_docs(docs)
    sources = rc._extract_citations(docs, question)
    conf = rc._confidence_from_scores(scores)
    return {"query": rq, "qc": qc, "docs": docs, "context": ctx, "sources": sources, "confidence": conf, "scores": scores}


def render_messages(history: List[Dict], question: str, context: str) -> List:
    # Build messages with a short history (last 4 turns)
    hist = history[-4:]
    hist_text = "\n".join(f"{h.get('role','user').title()}: {h.get('content','')}" for h in hist)
    q = question
    if hist_text:
        q = f"Conversation so far:\n{hist_text}\n\nCurrent question:\n{question}"
    sys = rc.prompt.format_prompt(context=context, question=q).to_messages()
    # sys already includes a System+Human; prepend/merge minimal history textually
    msgs = []
    for m in sys:
        msgs.append(m)
    return msgs


@app.on_event("startup")
async def maybe_build_index_on_start():
    try:
        if not settings.auto_index_on_start:
            return
        vector_dir = settings.vectorstore_dir
        needs_build = (not os.path.isdir(vector_dir)) or (not any(True for _ in os.scandir(vector_dir)))
        if not needs_build:
            return
        # Avoid duplicate runs
        setattr(app.state, "indexing", True)
        loop = asyncio.get_running_loop()
        # Run in a thread to avoid blocking the event loop
        await loop.run_in_executor(None, build_vector_store)
    except Exception as e:
        logger.warning("Auto index build failed: %s", e)
    finally:
        setattr(app.state, "indexing", False)


@app.post("/chat/invoke")
async def chat_invoke(payload: dict):
    text = str(payload.get("input", ""))
    history = payload.get("history") or []
    if not text.strip():
        return JSONResponse(status_code=400, content={"error": {"type": "bad_request", "message": "Missing input"}})
    info = build_context(text)
    llm = rc._choose_llm(info.get("scores") or [])
    messages = render_messages(history, text, info["context"])
    out = llm.invoke(messages)
    answer = getattr(out, "content", str(out))
    # Apply guard using raw sources/conf, then decide what to display
    raw_sources = info["sources"]
    conf = float(info.get("confidence") or 0.0)
    guard_triggered = (len(raw_sources) < settings.min_sources_required) or (
        (conf > 0.0) and (conf < settings.confidence_threshold)
    )
    if guard_triggered:
        return {
            "answer": "I don't have enough grounded context to answer confidently. Could you clarify or provide more details?",
            "sources": [],
            "rewritten_query": info["query"],
            "confidence": info["confidence"],
        }
    # Suppress sources in the display unless conditions are met
    srcs = raw_sources
    show_sources = (
        (info.get("qc") != "other")
        and (len(raw_sources) >= settings.min_sources_required)
        and (not answer.strip().lower().startswith("i don't know"))
        and (conf >= settings.show_sources_min_conf)
    )
    if not show_sources:
        srcs = []
    # Display confidence: use rerank-based if available; otherwise fallback to groundedness heuristic
    if conf <= 0.0:
        g_score, _ = groundedness_score(answer, [d.page_content for d in info.get("docs", [])])
        disp_conf = float(g_score)
    else:
        disp_conf = conf
    payload = {"answer": answer, "sources": srcs, "rewritten_query": info["query"], "confidence": disp_conf}
    return payload


@app.post("/chat/stream")
async def chat_stream(payload: dict):
    text = str(payload.get("input", ""))
    history = payload.get("history") or []
    if not text.strip():
        return JSONResponse(status_code=400, content={"error": {"type": "bad_request", "message": "Missing input"}})
    info = build_context(text)
    llm = rc._choose_llm(info.get("scores") or [])
    messages = render_messages(history, text, info["context"])

    def gen():
        try:
            for chunk in llm.stream(messages):
                token = getattr(chunk, "content", "")
                if token:
                    yield f"data: {json.dumps({'type':'token','data': token})}\n\n"
        except Exception:
            pass
        finally:
            # Decide whether to include sources in the final event
            # Display confidence: prefer rerank-based; fallback to groundedness
            conf = float(info.get("confidence") or 0.0)
            if conf <= 0.0:
                try:
                    g_score, _ = groundedness_score("", [d.page_content for d in info.get("docs", [])])
                    conf = float(g_score)
                except Exception:
                    conf = 0.0
            include_sources = (
                (info.get("qc") != "other")
                and (len(info.get("sources") or []) >= settings.min_sources_required)
                and (conf >= settings.show_sources_min_conf)
            )
            final_payload = {
                "type": "final",
                "data": {
                    "sources": (info["sources"] if include_sources else []),
                    "rewritten_query": info["query"],
                    "confidence": conf,
                },
            }
            yield f"data: {json.dumps(final_payload)}\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream")
