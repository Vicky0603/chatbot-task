from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from langserve import add_routes
import os

from src.chains.rag_chain import rag_chain
from src.config import settings

app = FastAPI(title="Promtior RAG Chatbot")

# CORS: in production, restrict allow_origins to specific domains
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# simple request logging
@app.middleware("http")
async def log_requests(request, call_next):
    print(f"Incoming request: {request.method} {request.url}")
    resp = await call_next(request)
    print(f"Response status: {resp.status_code} for {request.method} {request.url}")
    return resp

# LangServe
add_routes(app, rag_chain, path="/promtior-rag")

# Frontend
if os.path.isdir(settings.frontend_dir):
    app.mount("/static", StaticFiles(directory=settings.frontend_dir), name="static")

@app.get("/")
def index():
    idx = os.path.join(settings.frontend_dir, "index.html")
    # asegurar content-type correcto
    return FileResponse(idx, media_type="text/html")
