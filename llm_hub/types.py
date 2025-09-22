from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Iterable, List, Optional, Union, Callable
from pydantic import BaseModel

class Role(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    TOOL = "tool"

@dataclass
class Message:
    role: Role
    content: Union[str, List[Dict[str, Any]]]  # text or provider-style parts for multimodal

@dataclass
class InputMedia:
    # For images/PDFs: specify one (uri, path, or bytes)
    mime_type: str
    uri: Optional[str] = None
    path: Optional[str] = None
    data: Optional[bytes] = None

@dataclass
class ToolSpec:
    name: str
    description: Optional[str] = None
    parameters_json_schema: Dict[str, Any] = field(default_factory=dict)

@dataclass
class StructuredSchema:
    # Accept either a Pydantic model class or a raw JSON schema
    pydantic_model: Optional[type[BaseModel]] = None
    json_schema: Optional[Dict[str, Any]] = None
    strict: bool = True
    name: str = "StructuredOutput"

@dataclass
class StreamEvent:
    provider: str
    type: str  # "text.delta", "tool.call.delta", "reasoning.delta", "message.completed", "error", "usage"
    data: Any

@dataclass
class ToolCall:
    name: str
    arguments: Dict[str, Any]
    call_id: Optional[str] = None

@dataclass
class ToolResult:
    call_id: Optional[str]
    content: Any
