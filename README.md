# Promtior RAG Chatbot

Small RAG project exposing a LangServe endpoint and a static frontend to chat with indexed Promtior information.

## Key Contents
- FastAPI + LangServe server: `src/main.py`
- RAG chain: `src/chains/rag_chain.py`
- Web/PDF ingestion: `src/ingestion/load_promtior_site.py`
- Vector store builder: `src/ingestion/build_vector_store.py`
- Vector store / embeddings: `src/vectorstore/loader.py`
- Static frontend: `frontend/index.html`, `frontend/app.js`, `frontend/styles.css`

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

## Test the API from PowerShell

Quick example using `Invoke-RestMethod`:

```powershell
Invoke-RestMethod -Method Post -Uri "http://localhost:8000/promtior-rag/invoke" -ContentType "application/json" -Body '{"input":"What does Promtior do?"}'
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



