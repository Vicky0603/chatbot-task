from __future__ import annotations

import os
import logging

logger = logging.getLogger("app.otel")


def setup_otel_from_env() -> None:
    if os.getenv("OTEL_ENABLED", "false").lower() != "true":
        return
    try:
        from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
        import opentelemetry.trace as trace
    except Exception as e:  # pragma: no cover
        logger.warning("OpenTelemetry not available: %s", e)
        return
    # Configure tracer provider
    service_name = os.getenv("OTEL_SERVICE_NAME", "promtior-rag")
    resource = Resource.create({"service.name": service_name})
    provider = TracerProvider(resource=resource)
    endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4318")
    processor = BatchSpanProcessor(OTLPSpanExporter(endpoint=f"{endpoint}/v1/traces"))
    provider.add_span_processor(processor)
    trace.set_tracer_provider(provider)

    # The FastAPIInstrumentor will be applied in main after app creation
    logger.info("OpenTelemetry tracing initialized for %s", service_name)

