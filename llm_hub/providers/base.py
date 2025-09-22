from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, AsyncIterator, Dict, List, Optional
from ..types import Message, InputMedia, ToolSpec, StructuredSchema, StreamEvent, ToolResult
from ..usage import Usage

class LLMProvider(ABC):
    name: str

    @abstractmethod
    async def generate(
        self,
        *,
        model: str,
        messages: List[Message],
        system: Optional[str],
        tools: Optional[List[ToolSpec]],
        tool_results: Optional[List[ToolResult]],
        structured: Optional[StructuredSchema],
        temperature: Optional[float],
        reasoning_effort: Optional[str],
        stream: bool,
    ) -> Any:
        """Return a provider-native response object (or an async iterator if stream=True)."""

    @abstractmethod
    async def stream(
        self,
        *,
        model: str,
        messages: List[Message],
        system: Optional[str],
        tools: Optional[List[ToolSpec]],
        tool_results: Optional[List[ToolResult]],
        structured: Optional[StructuredSchema],
        temperature: Optional[float],
        reasoning_effort: Optional[str],
    ) -> AsyncIterator[StreamEvent]:
        ...

    @abstractmethod
    async def upload(self, *, media: InputMedia, purpose: Optional[str] = None) -> Dict[str, Any]:
        """Upload a file/image/doc and return provider-native file handle/metadata."""

    @abstractmethod
    async def batch(self, *, model: str, requests_file_id: str, endpoint: str = "/v1/responses") -> Dict[str, Any]:
        """Create a provider-native batch job where supported (OpenAI, Anthropic)."""

    @abstractmethod
    def parse_usage(self, response: Any) -> Usage:
        """Extract token usage into a unified Usage object."""
