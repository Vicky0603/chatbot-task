from pydantic import BaseModel
from dotenv import load_dotenv
import os
from pathlib import Path
import logging

# Resolve project root robustly
PROJECT_ROOT = Path(__file__).resolve().parents[1]  # src/ -> project root
ENV_PATH = PROJECT_ROOT / "environments" / "local.env"
ALT_ENV_PATH = PROJECT_ROOT / "environments" / ".env"

# Load .env only if present, or use ENV_FILE override.
# Do not raise if missing: on Railway, env vars come from the host.
env_file_override = os.getenv("ENV_FILE")
if env_file_override:
    env_path = Path(env_file_override)
    if env_path.exists():
        load_dotenv(dotenv_path=str(env_path))
elif ENV_PATH.exists():
    load_dotenv(dotenv_path=str(ENV_PATH))
elif ALT_ENV_PATH.exists():
    load_dotenv(dotenv_path=str(ALT_ENV_PATH))
else:
    # No local env found — OK for hosted envs (Railway). Do not raise.
    logging.getLogger(__name__).info("No local env file found; using environment variables from host.")

# Ensure USER_AGENT is set to silence some client warnings (can be overridden via env)
os.environ.setdefault("USER_AGENT", "promtior-rag/1.0")

# Sanitize OPENAI_API_KEY to remove trailing newlines/quotes/spaces that break HTTP headers
_raw_openai_key = os.getenv("OPENAI_API_KEY")
if _raw_openai_key:
    _clean_key = _raw_openai_key.strip().strip('"').strip("'").replace("\n", "").replace("\r", "")
    # Put back into env so other libraries see the cleaned value
    os.environ["OPENAI_API_KEY"] = _clean_key
else:
    _clean_key = None

# Normalize MODEL_NAME; treat missing or blank as default
_raw_model = os.getenv("MODEL_NAME")
if _raw_model is None or not _raw_model.strip().strip('"').strip("'"):
    _clean_model = "gpt-4o-mini"
else:
    _clean_model = _raw_model.strip().strip('"').strip("'")

class Settings(BaseModel):
    openai_api_key: str | None = _clean_key
    model_name: str = _clean_model
    vectorstore_dir: str = os.getenv("VECTORSTORE_DIR", "./data/vectorstore")
    frontend_dir: str = os.getenv("FRONTEND_DIR", "./frontend")
    log_level: str = os.getenv("APP_LOG_LEVEL", "info")
    # CORS / Security
    allowed_origins: str = os.getenv("ALLOWED_ORIGINS", "*")
    content_security_policy: str = os.getenv(
        "CONTENT_SECURITY_POLICY",
        "default-src 'self'; style-src 'self' 'unsafe-inline'; script-src 'self'; img-src 'self' data:; connect-src 'self'"
    )
    api_key: str | None = os.getenv("API_KEY")
    # Runtime flag: indicates whether OpenAI is usable (true by default)
    openai_usable: bool = True
    # Rate limit & auth
    rate_limit_enabled: bool = os.getenv("RATE_LIMIT_ENABLED", "false").lower() == "true"
    rate_limit_per_minute: int = int(os.getenv("RATE_LIMIT_PER_MINUTE", "60"))
    basic_auth_enabled: bool = os.getenv("PROTECT_WITH_BASIC_AUTH", "false").lower() == "true"
    basic_auth_user: str | None = os.getenv("PROTECT_USERNAME")
    basic_auth_pass: str | None = os.getenv("PROTECT_PASSWORD")
    # Embeddings provider
    embeddings_provider: str = os.getenv("EMBEDDINGS_PROVIDER", "openai")  # openai|e5
    embeddings_model: str = os.getenv("EMBEDDINGS_MODEL", "text-embedding-3-small")
    e5_model_name: str = os.getenv("E5_MODEL_NAME", "intfloat/e5-small-v2")
    # Reranking
    rerank_provider: str = os.getenv("RERANK_PROVIDER", "local")  # local|cohere
    cohere_api_key: str | None = os.getenv("COHERE_API_KEY")
    # Hallucination guard
    min_sources_required: int = int(os.getenv("MIN_SOURCES_REQUIRED", "1"))
    confidence_threshold: float = float(os.getenv("CONFIDENCE_THRESHOLD", "0.35"))
    # Fallback model
    fallback_model_name: str | None = os.getenv("FALLBACK_MODEL_NAME")
    # Discovery & chunking
    enable_sitemap_discovery: bool = os.getenv("ENABLE_SITEMAP_DISCOVERY", "false").lower() == "true"
    enable_html_chunking: bool = os.getenv("ENABLE_HTML_CHUNKING", "false").lower() == "true"
    # Metrics / tracing
    metrics_enabled: bool = os.getenv("METRICS_ENABLED", "true").lower() == "true"
    # Scoped API keys for specific route prefixes (JSON: {"/metrics":"key1","/chat":"key2"})
    route_api_keys_json: str | None = os.getenv("ROUTE_API_KEYS_JSON")

settings = Settings()
