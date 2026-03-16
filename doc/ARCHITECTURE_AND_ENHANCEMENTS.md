
# Promtior RAG Chatbot — Architecture & Enhancements

## Overview
This document details the architecture, rationale, and implemented enhancements that improve retrieval quality, UX, and operations. It complements the quickstart in `README.md` and the high‑level description in `doc/project_overview.md`.

## Architecture Summary

- Data ingestion: website pages and optional PDFs + curated notes (`data/raw/*.md|*.txt`).
- Indexing: chunking with `RecursiveCharacterTextSplitter`, embeddings with OpenAI, persisted in Chroma.
- Retrieval: hybrid BM25 + vector (MMR), with optional cross-encoder reranking.
- Generation: OpenAI chat model with a grounded prompt that only uses retrieved context.
- Serving: FastAPI + LangServe exposes the chain at `/promtior-rag`.
- Frontend: static HTML/CSS/JS with streaming, citations, and confidence display.

Key files
- Server: `src/main.py` (app wiring, routes, middleware) and `src/config.py` (env, defaults)
- Chain: `src/chains/rag_chain.py` (query rewriting, hybrid retrieval, optional reranking, citations, prompt + LLM)
- Ingestion + index: `src/ingestion/load_promtior_site.py`, `src/ingestion/build_vector_store.py` (incremental indexing)
- Vector store: `src/vectorstore/loader.py`
- Frontend: `frontend/index.html`, `frontend/app.js`, `frontend/styles.css`
- Ops: `src/api/errors.py` (centralized errors), `src/api/middleware.py` (request IDs, rate limit, basic auth), `src/logging_config.py`
- Telemetry: `src/telemetry/tracing.py`

## Implemented Enhancements
1) Query rewriting
   - Normalizes and expands vague queries with domain synonyms, improving recall.
   - Code: `src/query/rewriter.py`, integrated in `src/chains/rag_chain.py`.
   - Config: override synonyms via `QUERY_SYNONYMS_JSON`.

2) Cross-encoder reranking (optional)
   - Reranks retrieved documents using a lightweight `sentence-transformers` cross-encoder.
   - Graceful fallback if model isn’t available or offline.
   - Code: `src/rerank/cross_encoder.py`, wired in `src/chains/rag_chain.py`.
   - Config: `RERANKING_ENABLED=true` (default true), `RERANKER_MODEL_NAME`.

3) Citations UX
   - Backend emits clickable `url` when present, plus `preview_html` with query-term highlights and `rerank_score`.
   - Frontend renders links, highlights, and confidence badge.
   - Code: `src/chains/rag_chain.py`, `frontend/app.js`.

4) Incremental indexing
   - Deterministic chunk IDs; only embeds new/changed chunks and removes stale chunks per source.
   - Faster rebuilds; safer updates when source content changes.
   - Code: `src/ingestion/build_vector_store.py`.

5) Telemetry (opt‑in)
   - LangSmith/LC tracing via env toggle; no-op if not configured.
   - Code: `src/telemetry/tracing.py`, initialized in `src/main.py`.
   - Config: `ENABLE_TELEMETRY=true`, `LANGCHAIN_API_KEY`, optional `LANGCHAIN_PROJECT`.

6) Structured outputs
   - Chain returns `{ answer, sources, rewritten_query, confidence }`.
   - Confidence is derived from reranker scores (sigmoid-averaged), if reranking is active.
   - Code: `src/chains/rag_chain.py`.

## Data and Indexing Details
- Sources
  - Web pages via `WebBaseLoader` using Promtior URLs.
  - PDFs from `data/raw/` (filename autodetected; fallback to first `.pdf`).
  - Local notes (`data/raw/*.md|*.txt`) for curated facts and overrides.
- Chunking
  - `RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=200)` in `build_vector_store`.
- Embeddings
  - OpenAI `text-embedding-3-small` (1536-dim). Clear error messages for configuration issues.
- Persistence
  - Chroma persisted under `./data/vectorstore`. Mount as a volume in production.
- Incremental mode
  - Deterministic `chunk_id` per source+content; skip existing IDs, delete stale ones per source.

## Retrieval & Generation
- Vector retriever
  - Chroma retriever with MMR: `k=8`, `fetch_k=24`.
- BM25 retriever
  - Constructed from Chroma documents for lexical matches.
- Ensemble
  - Weighted fusion (0.5 BM25, 0.5 vector), fallback to vector-only if BM25 build fails.
- Query rewriting
  - Retrieval uses the rewritten query; LLM prompt still uses the user’s original wording for the question text.
- Reranking
  - If enabled, cross-encoder reranks top candidates and writes scores to metadata.
- Prompt
  - Guardrails: only answer from context, otherwise suggest rephrasing.
- Output
  - `RunnableParallel` returns `{ answer, sources, rewritten_query, confidence }`.

## Frontend Behavior
- Streams tokens via `/promtior-rag/stream`, then fetches the final structured payload via `/promtior-rag/invoke` to render citations.
- Displays clickable source links when available, highlighted previews, and a confidence indicator.
- Error handling parses standardized JSON error payloads including `request_id`.

## Error Handling & Observability
- Centralized handlers return a consistent error schema and include `request_id`.
- Request logging: method, path, status.
- `/healthz` returns OpenAI config status, vector store presence, and a link to the LangServe Playground.
- Optional telemetry wiring for deeper traces (opt‑in via env).

## Testing & CI
- Unit tests (no network):
  - `tests/test_bm25.py`: BM25 retrieves a founding-date fact.
  - `tests/test_citations.py`: Citation extraction dedupes and formats correctly.
- CI: `.github/workflows/ci.yml` runs tests and builds the Docker image.

## Deployment Notes
- Dockerfile builds a slim Python image and runs `uvicorn src.main:app`.
- For production hardening:
  - `RATE_LIMIT_ENABLED=true`, `RATE_LIMIT_PER_MINUTE=60`
  - `PROTECT_WITH_BASIC_AUTH=true`, `PROTECT_USERNAME=admin`, `PROTECT_PASSWORD=secret`
  - Mount `./data/vectorstore` to a persistent volume.

---
This document records the architecture and the enhancement rationale to guide maintenance and future development.
7) Pluggable embeddings
   - Supports OpenAI and E5 via `EMBEDDINGS_PROVIDER` with per-provider model configuration.
   - Writes `meta.json` next to the vector store with provider info.

8) Query classification
   - Heuristic classification influences retrieval params (`k`, weights) per query type.
   - Code: `src/query/classifier.py`.

9) Diversified rerankers
   - Cohere API reranker available via `RERANK_PROVIDER=cohere` (requires `COHERE_API_KEY`), otherwise local CrossEncoder.

10) Hallucination guard
   - Enforces `MIN_SOURCES_REQUIRED` and `CONFIDENCE_THRESHOLD` before returning an answer, asks for clarification otherwise.

11) Response cache
   - In-memory cache of final answers keyed by rewritten query and top document IDs.

12) Security & UX
   - CSP headers; configurable CORS `ALLOWED_ORIGINS`; optional API key `API_KEY` for programmatic access.
   - UI renders markdown safely (basic support), adds copy button, clickable sources with highlights.

13) Data ops
   - Optional `data/sources.yaml` registry augments the static URL list from code.

14) HTML-aware chunking and anchors
   - Optional (`ENABLE_HTML_CHUNKING=true`) parses pages into section-level chunks with `metadata.fragment` for footnotes and deep links.

15) Sitemap/robots discovery
   - Optional (`ENABLE_SITEMAP_DISCOVERY=true`) fetches robots.txt and sitemaps to auto-augment URLs.

16) Observability
   - Prometheus metrics exposed at `/metrics` (`METRICS_ENABLED=true`), tracking requests and retrieval counts.
   - OpenTelemetry instrumentation (`OTEL_ENABLED=true`), export via `OTEL_EXPORTER_OTLP_ENDPOINT`.

17) Structured streaming
   - New endpoints `/chat/stream` (SSE JSON events) and `/chat/invoke` for multi-turn interactions with client-side history.

18) Verification and fallbacks
   - Simple groundedness score computation with sentence support matches.
   - Fallback model selection if predicted confidence is low (`FALLBACK_MODEL_NAME`).

19) Access control
   - Optional per-route API keys via `ROUTE_API_KEYS_JSON` mapping path prefixes to required keys.
- Updated component diagram: see `doc/Component-Diagram.png` (source: `doc/component-diagram.mmd`).
