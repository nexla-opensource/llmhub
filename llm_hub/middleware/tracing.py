from __future__ import annotations
from contextlib import asynccontextmanager, contextmanager
from typing import AsyncIterator, Optional, Dict, Any
from opentelemetry import trace

@asynccontextmanager
async def traced_span(enabled: bool, name: str, attributes: Optional[Dict[str, Any]] = None) -> AsyncIterator[None]:
    if not enabled:
        yield
        return
    tracer = trace.get_tracer("llm_hub")
    with tracer.start_as_current_span(name) as span:
        if attributes:
            for k, v in attributes.items():
                try:
                    span.set_attribute(k, v)
                except Exception:
                    pass
        yield

@contextmanager
def traced_span_sync(enabled: bool, name: str, attributes: Optional[Dict[str, Any]] = None):
    if not enabled:
        yield
        return
    tracer = trace.get_tracer("llm_hub")
    with tracer.start_as_current_span(name) as span:
        if attributes:
            for k, v in attributes.items():
                try:
                    span.set_attribute(k, v)
                except Exception:
                    pass
        yield
