from __future__ import annotations
from typing import Any, Iterator, Dict, List, Optional
from google import genai
from google.genai import types as ggtypes
from ..types import Message, InputMedia, ToolSpec, StructuredSchema, StreamEvent, ToolResult
from ..usage import Usage

class GeminiSyncProvider:
    name = "gemini"

    def __init__(self, *, api_key: Optional[str], max_retries: Optional[int], timeout: Optional[float], vertexai: bool, project: Optional[str], location: Optional[str], api_version: Optional[str]):
        http_options = ggtypes.HttpOptions(api_version=api_version) if api_version else None
        self.client = genai.Client(
            api_key=api_key,
            vertexai=vertexai,
            project=project,
            location=location,
            http_options=http_options,
        )

    def _to_contents(self, messages: List[Message]) -> List[Any]:
        return [m.content for m in messages]

    def _to_tools(self, tools: Optional[List[ToolSpec]]) -> Optional[List[Any]]:
        if not tools:
            return None
        out = []
        for t in tools:
            out.append(ggtypes.Tool(function_declarations=[
                ggtypes.FunctionDeclaration(
                    name=t.name,
                    description=t.description or "",
                    parameters=t.parameters_json_schema or {"type": "object", "properties": {}},
                )
            ]))
        return out

    def _structured_cfg(self, structured: Optional[StructuredSchema]) -> Dict[str, Any]:
        cfg: Dict[str, Any] = {}
        if structured:
            schema = structured.pydantic_model.model_json_schema() if structured.pydantic_model else (structured.json_schema or {"type":"object"})
            cfg["response_mime_type"] = "application/json"
            cfg["response_schema"] = schema
        return cfg

    def generate_sync(
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
        if stream:
            return self.stream_sync(
                model=model, messages=messages, system=system, tools=tools, tool_results=tool_results,
                structured=structured, temperature=temperature, reasoning_effort=reasoning_effort
            )
        cfg = ggtypes.GenerateContentConfig(
            temperature=temperature,
            tools=self._to_tools(tools),
            system_instruction=system,
            **self._structured_cfg(structured),
        )
        return self.client.models.generate_content(model=model, contents=self._to_contents(messages), config=cfg)

    def stream_sync(
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
    ) -> Iterator[StreamEvent]:
        cfg = ggtypes.GenerateContentConfig(
            temperature=temperature,
            tools=self._to_tools(tools),
            system_instruction=system,
            **self._structured_cfg(structured),
        )
        stream = self.client.models.generate_content_stream(model=model, contents=self._to_contents(messages), config=cfg)
        for chunk in stream:
            text = getattr(chunk, "text", None)
            if text:
                yield StreamEvent(provider=self.name, type="text.delta", data=text)

    def upload_sync(self, *, media: InputMedia, purpose: Optional[str] = None) -> Dict[str, Any]:
        if media.path:
            uploaded = self.client.files.upload(file=media.path)
            return uploaded.to_dict() if hasattr(uploaded, "to_dict") else dict(uploaded)
        if media.data:
            return {"inline": True, "mime_type": media.mime_type, "size": len(media.data)}
        if media.uri:
            return {"type": "url", "url": media.uri}
        raise ValueError("Gemini upload expects media.path or media.data or media.uri")

    def batch_sync(self, *, model: str, requests_file_id: str, endpoint: str = "") -> Dict[str, Any]:
        return {"supported": False, "message": "Batch processing not provided by Gemini Developer API"}

    def parse_usage(self, response: Any) -> Usage:
        usage_meta = getattr(response, "usage_metadata", None) or getattr(response, "usageMetadata", None)
        model = getattr(response, "model", "gemini")
        if not usage_meta:
            return Usage(provider=self.name, model=model)
        prompt = int(getattr(usage_meta, "prompt_token_count", 0) or getattr(usage_meta, "promptTokenCount", 0) or 0)
        completion = int(getattr(usage_meta, "candidates_token_count", 0) or getattr(usage_meta, "candidatesTokenCount", 0) or 0)
        total = int(getattr(usage_meta, "total_token_count", 0) or getattr(usage_meta, "totalTokenCount", 0) or (prompt + completion))
        return Usage(provider=self.name, model=model, prompt_tokens=prompt, completion_tokens=completion, total_tokens=total)
