# Promtior RAG Chatbot

Small RAG project exposing a LangServe endpoint and a static frontend to chat with indexed Promtior information.

## Key Contents
- FastAPI + LangServe server: `src/main.py`
- RAG chain: `src/chains/rag_chain.py`
- Web/PDF ingestion: `src/ingestion/load_promtior_site.py`
- Vector store builder: `src/ingestion/build_vector_store.py`
- Vector store / embeddings: `src/vectorstore/loader.py`
- Static frontend: `frontend/index.html`, `frontend/app.js`, `frontend/styles.css`
- Architecture & Enhancements: `doc/ARCHITECTURE_AND_ENHANCEMENTS.md`
- Component diagram (Mermaid source): `doc/component-diagram.mmd` (export to `doc/Component-Diagram.png`)
- Sequence diagram (Mermaid source): `doc/sequence-diagram.mmd` (export to `doc/Sequence-Diagram.png`)

## Requirements
- Python 3.10+
- `OPENAI_API_KEY` to use OpenAI (embeddings and LLM)
- Install dependencies: `pip install -r requirements.txt`

## Quick Setup (local)

Follow these steps to run locally (PowerShell examples for Windows included):

1. Create and activate a virtual environment (from the project root):

  - Windows (PowerShell):
    ```powershell
    python -m venv venv
    .\venv\Scripts\Activate.ps1
    ```

  - Unix / macOS:
    ```bash
    python -m venv .venv
    source .venv/bin/activate
    ```

2. Install dependencies:

  ```bash
  pip install --upgrade pip
  pip install -r requirements.txt
  ```

3. Configure environment variables (required file):

  The app reads `environments/local.env` or alternatively `environments/.env`. Create one of them with at least:

  ```text
  # environments/local.env (or environments/.env)
  OPENAI_API_KEY=sk-...
  MODEL_NAME=gpt-4o-mini
  VECTORSTORE_DIR=./data/vectorstore
  FRONTEND_DIR=./frontend
  ```

  - This project uses OpenAI embeddings. Make sure to set `OPENAI_API_KEY`.

4. (Optional) Rebuild the vector store:

  If `data/vectorstore` doesn’t exist, or you want to re-index documents/PDFs:

  ```powershell
  Remove-Item .\data\vectorstore -Recurse -Force
  python -m src.ingestion.build_vector_store
  ```

  It will read the website (predefined URLs) and, if a PDF exists in `data/raw` (e.g., `AI Engineer-Tecnical-Test.pdf`), it will include it and persist a Chroma vector store.

5. Run the app (development):

  ```powershell
  uvicorn src.main:app --reload --port 8000
  ```

  - Frontend: http://127.0.0.1:8000/
  - LangServe Playground: http://127.0.0.1:8000/promtior-rag/playground/
  - Programmatic endpoint: POST `http://127.0.0.1:8000/promtior-rag/invoke` (JSON `{ "input": "your question" }`).
  - Structured chat: POST `/chat/stream` (SSE JSON `{type:'token'|'final'}`), POST `/chat/invoke` (supports `history`).
  - Health check: GET `http://127.0.0.1:8000/healthz`
  - Metrics: GET `/metrics` (Prometheus; `METRICS_ENABLED=true`)

## Production hardening (optional)
- Enable rate limiting:
  - Set `RATE_LIMIT_ENABLED=true` and `RATE_LIMIT_PER_MINUTE=60` (or as needed).
- Protect with Basic Auth:
  - Set `PROTECT_WITH_BASIC_AUTH=true`, `PROTECT_USERNAME=admin`, `PROTECT_PASSWORD=secret`.
  - Applies to all routes except `/healthz` and static files.
- Correlation IDs:
  - Every request/response includes `X-Request-ID`. Errors include `request_id` in the JSON payload.

## Test the API from PowerShell

Quick example using `Invoke-RestMethod`:

```powershell
Invoke-RestMethod -Method Post -Uri "http://localhost:8000/promtior-rag/invoke" -ContentType "application/json" -Body '{"input":"What does Promtior do?"}'
```

Additional local checks (after `uvicorn src.main:app --reload --port 8000`):

- Chat (structured):
  ```powershell
  Invoke-RestMethod -Method Post -Uri "http://localhost:8000/chat/invoke" -ContentType "application/json" -Body '{"input":"Give me a short overview of Promtior","history": []}'
  ```

- Chat (streaming SSE, JSON events) with curl:
  ```bash
  curl -N -H "Content-Type: application/json" -d '{"input":"List a couple of services","history": []}' http://localhost:8000/chat/stream
  ```

- Metrics (Prometheus exposition):
  ```bash
  curl http://localhost:8000/metrics
  ```

- Health check:
  ```bash
  curl http://localhost:8000/healthz
  ```

If you set `API_KEY`, include it in requests to protected routes (e.g., `/promtior-rag/*`, `/chat/*`):
```bash
curl -H "x-api-key: $API_KEY" -H "Content-Type: application/json" -d '{"input":"..."}' http://localhost:8000/chat/invoke
```

To try optional features locally:
- Enable HTML-aware chunking and sitemap discovery before rebuilding the index:
  ```bash
  export ENABLE_HTML_CHUNKING=true
  export ENABLE_SITEMAP_DISCOVERY=true
  python -m src.ingestion.build_vector_store
  ```
- Switch to E5 embeddings (then rebuild the vector store):
  ```bash
  export EMBEDDINGS_PROVIDER=e5
  export E5_MODEL_NAME=intfloat/e5-small-v2
  python -m src.ingestion.build_vector_store
  ```
- Use Cohere reranker:
  ```bash
  export RERANK_PROVIDER=cohere
  export COHERE_API_KEY=your_key
  ```



## Docker (optional)

Recommended `Dockerfile` is included. Common steps:

1. Build:

```bash
docker build -t promtior-rag:latest -f Dockerfile .
```

2. Run (mount `data` and pass env):

PowerShell:
```powershell
docker run --rm -p 8000:8000 --env-file .\environments\local.env -v "${PWD}\data:/app/data" promtior-rag:latest
```

Bash/WSL:
```bash
docker run --rm -p 8000:8000 --env-file ./environments/local.env -v "$(pwd)/data:/app/data" promtior-rag:latest
```



## Enhancements Included

- Query rewriting: normalizes queries and expands domain terms (e.g., "company" → "Promtior").
  - Config: override synonyms via `QUERY_SYNONYMS_JSON` (JSON dict).
- Cross-encoder reranking: optional local reranker to improve retrieval ordering.
  - Enable: `RERANKING_ENABLED=true` (default true)
  - Model: `RERANKER_MODEL_NAME=cross-encoder/ms-marco-MiniLM-L-6-v2`
- Citations UX: frontend shows clickable links (when available) and highlights matched spans.
- Incremental indexing: ingestion computes deterministic chunk IDs and only embeds new/changed chunks; removes stale ones.
- Telemetry: optional LangSmith/LC tracing.
  - Enable: `ENABLE_TELEMETRY=true`, provide `LANGCHAIN_API_KEY` and optional `LANGCHAIN_PROJECT`.
- Structured outputs: chain returns `answer`, `sources`, `rewritten_query`, and `confidence` (0–1).
 - Pluggable embeddings: choose OpenAI or E5.
   - Config: `EMBEDDINGS_PROVIDER=openai|e5`, `EMBEDDINGS_MODEL`, `E5_MODEL_NAME`.
 - Query classification: adjusts retrieval params per query type.
   - Code: `src/query/classifier.py`.
 - Reranking providers: `RERANK_PROVIDER=local|cohere` (falls back to local CrossEncoder).
- Markdown answers: basic markdown rendering in UI with simple sanitization.
 - Structured chat endpoints: `/chat/stream`, `/chat/invoke` with history.
 - Response cache: in-memory cache keyed by rewritten query + top doc IDs.
 - Hallucination guard: `MIN_SOURCES_REQUIRED` and `CONFIDENCE_THRESHOLD` control answering.
 - Security: configurable CORS via `ALLOWED_ORIGINS`, CSP headers, optional API key via `API_KEY`.
 - Feedback endpoint: `POST /feedback` with `{rating, comment}`; logs with request_id.
 - Source registry: optional `data/sources.yaml` listing extra URLs.
 - HTML-aware chunking: `ENABLE_HTML_CHUNKING=true` adds `metadata.fragment` and section anchors.
 - Sitemap discovery: `ENABLE_SITEMAP_DISCOVERY=true` augments URLs via robots/sitemaps.
- OpenTelemetry: `OTEL_ENABLED=true` with `OTEL_EXPORTER_OTLP_ENDPOINT` instruments FastAPI.

## Updating the Component Diagram
- The diagram source is in `doc/component-diagram.mmd` (Mermaid). To regenerate the PNG (`doc/Component-Diagram.png`):
  - Install Mermaid CLI and export:
    - `npx @mermaid-js/mermaid-cli -i doc/component-diagram.mmd -o doc/Component-Diagram.png`
  - Or use VS Code Mermaid extension to preview/export.

## Updating the Sequence Diagram
- The sequence diagram source is in `doc/sequence-diagram.mmd`.
- Regenerate the PNG with Mermaid CLI:
  - `npx @mermaid-js/mermaid-cli -i doc/sequence-diagram.mmd -o doc/Sequence-Diagram.png`

CI automation
- A GitHub Action (`.github/workflows/diagrams.yml`) auto-generates `doc/Component-Diagram.png` and `doc/Sequence-Diagram.png` on pushes to `main`/`master` when `.mmd` files change.

## Deploy to Railway

Follow these steps to get a production URL in minutes.

1) Connect the repo
- In Railway, create a new project → Deploy from GitHub → pick this repo.
- Railway detects the Dockerfile and builds the service.

2) Configure environment variables (Service → Variables)
- Required:
  - `OPENAI_API_KEY=sk-...`
  - `MODEL_NAME=gpt-4o-mini`
  - `VECTORSTORE_DIR=/app/data/vectorstore`
  - `FRONTEND_DIR=/app/frontend`
- Recommended (security/UX):
  - `ALLOWED_ORIGINS=https://your-domain.com`
  - `API_KEY=some-strong-secret` (requires `x-api-key` header on `/chat/*` and `/promtior-rag/*`)
  - `RATE_LIMIT_ENABLED=true`, `RATE_LIMIT_PER_MINUTE=60`
  - `PROTECT_WITH_BASIC_AUTH=true`, `PROTECT_USERNAME=admin`, `PROTECT_PASSWORD=secret`

3) Add a persistent volume (Service → Settings → Volumes)
- Add a volume named `data` and mount it at `/app/data`.
- This ensures the Chroma vectorstore persists across deploys and restarts.

4) Build the index (one-time job)
- Option A (temporary start command):
  - Change the service Start Command to: `python -m src.ingestion.build_vector_store`
  - Deploy once. When it finishes (vectorstore created under `/app/data/vectorstore`), switch Start Command back to:
    - `uvicorn src.main:app --host 0.0.0.0 --port 8000`
  - Redeploy.
- Option B (second service as a job):
  - Add a second service from the same repo, set Start Command to `python -m src.ingestion.build_vector_store` and attach the same `data` volume + env vars. Run it once and then disable/delete it.

5) Verify
- Open your Railway public URL.
- Check:
  - `/` (UI)
  - `/healthz` (should show vectorstore exists)
  - `/promtior-rag/playground/` (LangServe UI)
  - `/chat/invoke` (POST JSON `{"input":"What does Promtior do?","history":[]}`)
- If `API_KEY` is set, include header `x-api-key: <your key>` in requests.

Notes
- To update content later, rerun the index job (Step 4) to refresh the vectorstore while keeping the same volume.
- Optional features can be toggled via env (e.g., `ENABLE_HTML_CHUNKING=true`, `ENABLE_SITEMAP_DISCOVERY=true`, `RERANK_PROVIDER=cohere`).
