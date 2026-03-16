# Promtior RAG Chatbot – Project Overview

## 1) Introduction
This project implements a Retrieval-Augmented Generation (RAG) assistant for questions about Promtior. It ingests the public website, optional PDFs/notes, builds a vector index, and serves a chat UI and API. The stack uses FastAPI, LangChain, Chroma, and OpenAI (with an option to use E5 embeddings and Cohere reranking).

See also: `doc/ARCHITECTURE_AND_ENHANCEMENTS.md` for deeper details and `doc/Component-Diagram.png` for the system diagram (source in `doc/component-diagram.mmd`).

---

## 2) End‑to‑End Flow (Happy Path)
1. User submits a question from the browser or via the API.
2. Server normalizes and expands the query (domain synonyms) and classifies it (navigational/FAQ/comparison).
3. Retrieval runs as a hybrid of vector MMR and BM25; dynamic `k` and weights depend on the classification.
4. An optional reranker (Cohere API or local cross-encoder) reorders top candidates; scores derive a confidence estimate.
5. Context is assembled; a short chat history window is folded into the prompt.
6. The LLM (or a fallback model if confidence is low) generates a markdown answer.
7. A simple groundedness check maps answer sentences to supporting chunks; a hallucination guard may ask for clarification if evidence is weak.
8. The server returns a structured payload: answer, sources (with deep links and highlighted previews), rewritten query, and confidence.
9. The UI renders streaming tokens, then footnoted sources with a preview toggle and feedback buttons.

---

## 2.1) Sequence Diagram
- See `doc/sequence-diagram.mmd` (Mermaid) for the detailed request/response flow.
- Export a PNG (optional) with Mermaid CLI:
  - `npx @mermaid-js/mermaid-cli -i doc/sequence-diagram.mmd -o doc/Sequence-Diagram.png`
  - Or use a Mermaid extension in your editor to preview/export.

---

## 3) Data Ingestion & Freshness
- Sources
  - Static URLs plus optional YAML registry at `data/sources.yaml`.
  - Optional robots/sitemap discovery (`ENABLE_SITEMAP_DISCOVERY=true`) augments URLs automatically.
- Loaders
  - Website via `WebBaseLoader` (adds `metadata.dom_hash` for change detection).
  - PDF via `PyPDFLoader` and curated notes via `TextLoader`.
- HTML-aware chunking
  - Optional (`ENABLE_HTML_CHUNKING=true`): parse HTML into section-level chunks, store `metadata.fragment` anchors and headings (`metadata.h1`).
- Chunking
  - Default token-aware splitting; fallback to character splitter.
  - Deterministic `chunk_id` and per-source cleanup for incremental indexing.

---

## 4) Embeddings & Vector Store
- Vector store: Chroma persisted under `./data/vectorstore`.
- Embeddings providers
  - OpenAI (`EMBEDDINGS_PROVIDER=openai`, `EMBEDDINGS_MODEL=text-embedding-3-small`).
  - E5 (`EMBEDDINGS_PROVIDER=e5`, `E5_MODEL_NAME=intfloat/e5-small-v2`).
- A `meta.json` is written next to the vector DB with provider metadata for ops clarity.

---

## 5) Retrieval, Reranking, and Confidence
- Query rewriting expands domain terms (e.g., “company” → “Promtior/Promtior.ai”).
- Heuristic query classifier tunes `k` and BM25/vector weights per query type.
- Hybrid retrieval merges BM25 and vector hits; rerankers reorder candidates:
  - Cohere API (`RERANK_PROVIDER=cohere`, requires `COHERE_API_KEY`).
  - Local CrossEncoder (sentence-transformers), with safe fallback if the model is unavailable.
- Confidence estimate derives from reranker scores (sigmoid + average of top‑k).

---

## 6) Generation & Grounding
- Prompt instructs to answer only from provided context; suggests rephrasing when unknown.
- Short multi‑turn history is inlined above the current question to retain context within limits.
- Hallucination guard enforces a minimum number of sources and a confidence threshold.
- A groundedness check maps answer sentences to likely supporting chunks; results surface in telemetry and can be extended.
- Fallback model selection is used if predicted confidence is low (`FALLBACK_MODEL_NAME`).

---

## 7) Serving & Endpoints
- UI: static files served from `/`.
- LangServe: `/promtior-rag/*` including the Playground.
- Structured chat API:
  - `POST /chat/stream` (SSE JSON events with `{type:'token'|'final'}`)
  - `POST /chat/invoke` (returns structured payload; supports `history`)
- Health and Observability:
  - `GET /healthz` (basic health info)
  - `GET /metrics` (Prometheus exposition; enable with `METRICS_ENABLED=true`)
- Feedback sink: `POST /feedback` to log rating/comments with a `request_id`.

---

## 8) Security & Access Control
- CORS restricted via `ALLOWED_ORIGINS`; CSP headers set via `CONTENT_SECURITY_POLICY`.
- Optional Basic Auth, rate limiting, and per‑route API keys (`ROUTE_API_KEYS_JSON`).
- If `API_KEY` is set, `/promtior-rag/*` and `/chat/*` require `x-api-key`.

---

## 9) Frontend Behavior
- Streams tokens from `/chat/stream` and then fetches the final structured payload.
- Renders markdown safely, shows footnoted sources with deep links (when `fragment` present) and a preview toggle.
- Includes Copy and simple thumbs up/down feedback.

---

## 10) Running Locally (Quick Start)
1. Install: `pip install -r requirements.txt`
2. Set env: create `environments/local.env` with `OPENAI_API_KEY` (and optional toggles).
3. Index: `python -m src.ingestion.build_vector_store`
4. Run: `uvicorn src.main:app --reload --port 8000`
5. Test:
   - UI: http://localhost:8000/
   - Stream: `curl -N -H 'Content-Type: application/json' -d '{"input":"hi","history":[]}' http://localhost:8000/chat/stream`
   - Invoke: `curl -H 'Content-Type: application/json' -d '{"input":"hi","history":[]}' http://localhost:8000/chat/invoke`
   - Metrics: `curl http://localhost:8000/metrics`

---

## 11) Scope & Extensibility
This overview describes the operational flow and key extension points (embeddings, rerankers, verification, metrics, and security). The implementation favors safe defaults with feature flags so you can grow capabilities without breaking the baseline.
