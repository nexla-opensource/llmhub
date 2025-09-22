from __future__ import annotations
from typing import Dict, Any, Iterable, Callable
from .types import ToolSpec, ToolResult

def schema_from_pydantic(model_cls) -> Dict[str, Any]:
    return model_cls.model_json_schema()
