from __future__ import annotations
from typing import Any, AsyncIterator, Dict, List, Optional
from google import genai
from google.genai import types as ggtypes
from ..types import Message, InputMedia, ToolSpec, StructuredSchema, StreamEvent, ToolResult, Role
from ..usage import Usage

class GeminiProvider:
    name = "gemini"

    def __init__(self, *, api_key: Optional[str], max_retries: Optional[int], timeout: Optional[float], vertexai: bool, project: Optional[str], location: Optional[str], api_version: Optional[str]):
        http_options = None
        if api_version:
            http_options = ggtypes.HttpOptions(api_version=api_version)
        # Create sync client then use .aio for async
        self._client = genai.Client(
            api_key=api_key,
            vertexai=vertexai,
            project=project,
            location=location,
            http_options=http_options,
        )
        self.client = self._client.aio  # Async client

    def _to_contents(self, messages: List[Message]) -> List[Any]:
        # Gemini expects a list of Content items; we pass strings or parts as-is.
        contents: List[Any] = []
        for m in messages:
            # The SDK will coerce strings; for multimodal parts you can pass ggtypes.Part instances
            if isinstance(m.content, str):
                # Role-aware content is supported, but for basic generation a flat list is OK
                contents.append(m.content)
            else:
                contents.append(m.content)
        return contents

    def _to_tools(self, tools: Optional[List[ToolSpec]]) -> Optional[List[Any]]:
        if not tools:
            return None
        # Gemini function calling uses Tool(FunctionDeclaration)
        tool_list = []
        for t in tools:
            tool_list.append(
                ggtypes.Tool(
                    function_declarations=[
                        ggtypes.FunctionDeclaration(
                            name=t.name,
                            description=t.description or "",
                            parameters=t.parameters_json_schema or {"type": "object", "properties": {}},
                        )
                    ]
                )
            )
        return tool_list

    def _to_structured_config(self, structured: Optional[StructuredSchema]) -> Dict[str, Any]:
        cfg: Dict[str, Any] = {}
        if structured:
            if structured.pydantic_model:
                schema = structured.pydantic_model.model_json_schema()
            else:
                schema = structured.json_schema or {"type": "object"}
            cfg["response_mime_type"] = "application/json"
            cfg["response_schema"] = schema
        return cfg

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
        contents = self._to_contents(messages)
        tool_defs = self._to_tools(tools)
        gen_cfg = ggtypes.GenerateContentConfig(
            temperature=temperature,
            tools=tool_defs,
            system_instruction=system,
            **self._to_structured_config(structured),
        )

        if stream:
            return self.stream(
                model=model, messages=messages, system=system, tools=tools, tool_results=tool_results,
                structured=structured, temperature=temperature, reasoning_effort=reasoning_effort
            )

        resp = await self.client.models.generate_content(
            model=model,
            contents=contents,
            config=gen_cfg,
        )
        return resp

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
        contents = self._to_contents(messages)
        tool_defs = self._to_tools(tools)
        gen_cfg = ggtypes.GenerateContentConfig(
            temperature=temperature,
            tools=tool_defs,
            system_instruction=system,
            **self._to_structured_config(structured),
        )
        # Streaming: generate_content_stream
        stream = await self.client.models.generate_content_stream(
            model=model,
            contents=contents,
            config=gen_cfg,
        )
        async for chunk in stream:
            text = getattr(chunk, "text", None)
            if text:
                yield StreamEvent(provider=self.name, type="text.delta", data=text)

        # Gemini usage metadata on final responses; here streaming returns chunks; caller can
        # also do non-stream to collect usage.
        # No separate usage event emitted here due to API shape; call parse_usage on final response in non-stream.

    async def upload(self, *, media: InputMedia, purpose: Optional[str] = None) -> Dict[str, Any]:
        # Gemini File API (Developer API) supports client.files.upload(file=...) or inline bytes parts.
        # For Vertex AI, note: file uploads may not be supported; pass inline bytes instead or disable vertexai (docs/forums).
        if media.path:
            uploaded = await self.client.files.upload(file=media.path)  # accepts path string
            return uploaded.to_dict() if hasattr(uploaded, "to_dict") else dict(uploaded)
        if media.data:
            # Inline: use Part.from_bytes directly in contents (no upload)
            return {"inline": True, "mime_type": media.mime_type, "size": len(media.data)}
        if media.uri:
            # You can also pass ggtypes.Part.from_uri in contents; this is just metadata
            return {"type": "url", "url": media.uri}
        raise ValueError("Gemini upload expects media.path or media.data or media.uri")

    async def batch(self, *, model: str, requests_file_id: str, endpoint: str = "") -> Dict[str, Any]:
        # Gemini developer API does not expose a global batch endpoint analogous to OpenAI/Anthropic Message Batches.
        # Return a standardized not-supported response.
        return {"supported": False, "message": "Batch processing not provided by Gemini Developer API"}

    def parse_usage(self, response: Any) -> Usage:
        # Gemini returns usageMetadata: promptTokenCount, candidatesTokenCount, totalTokenCount
        usage_meta = getattr(response, "usage_metadata", None) or getattr(response, "usageMetadata", None)
        model = getattr(response, "model", "gemini")
        if not usage_meta:
            return Usage(provider=self.name, model=model)
        prompt = int(getattr(usage_meta, "prompt_token_count", 0) or getattr(usage_meta, "promptTokenCount", 0) or 0)
        completion = int(getattr(usage_meta, "candidates_token_count", 0) or getattr(usage_meta, "candidatesTokenCount", 0) or 0)
        total = int(getattr(usage_meta, "total_token_count", 0) or getattr(usage_meta, "totalTokenCount", 0) or (prompt + completion))
        return Usage(provider=self.name, model=model, prompt_tokens=prompt, completion_tokens=completion, total_tokens=total)
