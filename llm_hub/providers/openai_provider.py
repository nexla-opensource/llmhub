from __future__ import annotations
from typing import Any, AsyncIterator, Dict, List, Optional
from openai import AsyncOpenAI
from ..types import Message, InputMedia, ToolSpec, StructuredSchema, StreamEvent, ToolResult, Role
from ..usage import Usage
from ..exceptions import FeatureNotSupported
import json

class OpenAIProvider:
    name = "openai"

    def __init__(self, *, api_key: Optional[str], max_retries: Optional[int], timeout: Optional[float], organization: Optional[str], base_url: Optional[str]):
        # Native retries via SDK; you can also use with_options later per call.
        self.client = AsyncOpenAI(
            api_key=api_key,
            organization=organization,
            base_url=base_url,
            max_retries=max_retries,
            timeout=timeout,
        )

    # ---- helpers -------------------------------------------------------------

    def _to_responses_input(self, messages: List[Message], system: Optional[str]) -> Any:
        """
        Build the Responses API 'input' structure. We'll lean on simple text messages by default.
        For multimodal, Message.content can contain pre-built parts as dicts.
        """
        parts = []
        if system:
            parts.append({"role": "system", "content": system})
        for m in messages:
            if isinstance(m.content, str):
                parts.append({"role": m.role.value, "content": m.content})
            else:
                parts.append({"role": m.role.value, "content": m.content})
        return parts

    def _to_tools(self, tools: Optional[List[ToolSpec]]) -> Optional[List[Dict[str, Any]]]:
        if not tools:
            return None
        out = []
        for t in tools:
            out.append({
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description or "",
                    "parameters": t.parameters_json_schema or {"type": "object", "properties": {}}
                }
            })
        return out

    def _to_response_format(self, structured: Optional[StructuredSchema]) -> Optional[Dict[str, Any]]:
        if not structured:
            return None
        if structured.pydantic_model:
            schema = structured.pydantic_model.model_json_schema()
        else:
            schema = structured.json_schema or {"type": "object"}
        return {"type": "json_schema", "json_schema": {"name": structured.name, "schema": schema, "strict": structured.strict}}

    # ---- public API ----------------------------------------------------------

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
        inputs = self._to_responses_input(messages, system)
        tool_defs = self._to_tools(tools)
        resp_format = self._to_response_format(structured)

        # Reasoning effort for o-series (low|medium|high...); use Responses API parameter
        reasoning = {"effort": reasoning_effort} if reasoning_effort else None

        if stream:
            return self.stream(
                model=model, messages=messages, system=system, tools=tools,
                tool_results=tool_results, structured=structured,
                temperature=temperature, reasoning_effort=reasoning_effort
            )

        response = await self.client.responses.create(
            model=model,
            input=inputs,
            temperature=temperature,
            tools=tool_defs,
            response_format=resp_format,
            reasoning=reasoning,
        )
        return response

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
        inputs = self._to_responses_input(messages, system)
        tool_defs = self._to_tools(tools)
        resp_format = self._to_response_format(structured)
        reasoning = {"effort": reasoning_effort} if reasoning_effort else None

        # Streaming via Responses API context manager (async)
        async with self.client.responses.stream(
            model=model,
            input=inputs,
            temperature=temperature,
            tools=tool_defs,
            response_format=resp_format,
            reasoning=reasoning,
        ) as stream:
            async for event in stream:
                et = getattr(event, "type", "")
                # Common event types: response.output_text.delta, response.function_call.*, response.completed
                if et.endswith(".output_text.delta"):
                    yield StreamEvent(provider=self.name, type="text.delta", data=event.delta)
                elif ".reasoning." in et or et.endswith(".reasoning.delta"):
                    yield StreamEvent(provider=self.name, type="reasoning.delta", data=getattr(event, "delta", None))
                elif ".function_call." in et or ".tool_call." in et:
                    # You can inspect event.item/id/function arguments; we bubble up raw for app routing.
                    yield StreamEvent(provider=self.name, type="tool.call.delta", data=event.__dict__)
                elif et == "response.completed":
                    # Provide final usage
                    final = event.response
                    yield StreamEvent(provider=self.name, type="usage", data=self.parse_usage(final))
                    yield StreamEvent(provider=self.name, type="message.completed", data=final)
                elif et == "response.error":
                    yield StreamEvent(provider=self.name, type="error", data=event.error)

    async def upload(self, *, media: InputMedia, purpose: Optional[str] = None) -> Dict[str, Any]:
        # OpenAI Files API supports uploads for vision inputs and batches, etc.
        # For images: you can also pass image_url in content, but we support file storage too.
        if media.path:
            with open(media.path, "rb") as f:
                created = await self.client.files.create(file=f, purpose=purpose or "assistants")
                return created.model_dump()
        if media.data:
            # OpenAI SDK expects a file-like; here we advise users to prefer path for simplicity.
            raise FeatureNotSupported("OpenAI upload via raw bytes not implemented; use path/URL.")
        if media.uri:
            # No server-side fetch here; pass URLs directly in prompt parts instead.
            return {"type": "url", "url": media.uri, "mime_type": media.mime_type}
        raise ValueError("No media.path, media.uri, or media.data provided")

    async def batch(self, *, model: str, requests_file_id: str, endpoint: str = "/v1/responses") -> Dict[str, Any]:
        # OpenAI Batch API (24h window). Input must be uploaded as a JSONL file to Files API first.
        created = await self.client.batches.create(
            input_file_id=requests_file_id,
            endpoint=endpoint,
            completion_window="24h",
        )
        return created.model_dump()

    # ---- usage ---------------------------------------------------------------

    def parse_usage(self, response: Any) -> Usage:
        # OpenAI Responses API returns usage tokens incl. reasoning for o-series (and details by tokens)
        u = getattr(response, "usage", None)
        model = getattr(response, "model", "unknown")
        if not u:
            return Usage(provider=self.name, model=model)
        prompt = int(getattr(u, "prompt_tokens", 0))
        completion = int(getattr(u, "completion_tokens", 0))
        total = int(getattr(u, "total_tokens", prompt + completion))
        # Reasoning tokens: under completion_tokens_details.reasoning_tokens (varies by model)
        reasoning_tokens = 0
        details = {}
        try:
            ctd = u.completion_tokens_details or {}
            reasoning_tokens = int(getattr(ctd, "reasoning_tokens", 0))
            details = {
                "prompt_tokens": prompt,
                "completion_tokens": completion,
                "reasoning_tokens": reasoning_tokens,
                "cached_tokens": int(getattr(u.prompt_tokens_details, "cached_tokens", 0)) if getattr(u, "prompt_tokens_details", None) else 0,
            }
        except Exception:
            pass
        return Usage(provider=self.name, model=model, prompt_tokens=prompt, completion_tokens=completion, reasoning_tokens=reasoning_tokens, total_tokens=total, details=details)
