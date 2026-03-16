from __future__ import annotations

import os
import logging

logger = logging.getLogger("app.telemetry")


def setup_tracing_from_env() -> None:
    """
    Enable LangSmith/LC tracing if env vars indicate to do so.
    - ENABLE_TELEMETRY=true to opt-in
    - Supports LANGCHAIN_TRACING_V2, LANGCHAIN_ENDPOINT, LANGCHAIN_API_KEY, LANGCHAIN_PROJECT
    Does nothing if not enabled.
    """
    enabled = str(os.getenv("ENABLE_TELEMETRY", "false")).lower() == "true"
    if not enabled:
        return
    # Respect existing env if provided
    os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
    os.environ.setdefault("LANGCHAIN_ENDPOINT", os.getenv("LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com"))
    if os.getenv("LANGCHAIN_API_KEY"):
        logger.info("LangSmith tracing enabled (project=%s)", os.getenv("LANGCHAIN_PROJECT", "default"))
    else:
        logger.warning("ENABLE_TELEMETRY true but LANGCHAIN_API_KEY not set; tracing will be inactive.")

