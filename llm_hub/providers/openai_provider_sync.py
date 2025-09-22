from __future__ import annotations
from typing import Any, Iterator, Dict, List, Optional
from openai import OpenAI
from ..types import Message, InputMedia, ToolSpec, StructuredSchema, StreamEvent, ToolResult, Role
from ..usage import Usage
from ..exceptions import FeatureNotSupported

class OpenAISyncProvider:
    name = "openai"

    def __init__(self, *, api_key: Optional[str], max_retries: Optional[int], timeout: Optional[float], organization: Optional[str], base_url: Optional[str]):
        self.client = OpenAI(
            api_key=api_key,
            organization=organization,
            base_url=base_url,
            max_retries=max_retries,
            timeout=timeout,
        )

    def _to_responses_input(self, messages: List[Message], system: Optional[str]) -> Any:
        parts = []
        if system:
            parts.append({"role": "system", "content": system})
        for m in messages:
            parts.append({"role": m.role.value, "content": m.content})
        return parts

    def _to_tools(self, tools: Optional[List[ToolSpec]]) -> Optional[List[Dict[str, Any]]]:
        if not tools:
            return None
        return [{"type": "function", "function": {
            "name": t.name,
            "description": t.description or "",
            "parameters": t.parameters_json_schema or {"type": "object", "properties": {}}
        }} for t in tools]

    def _to_response_format(self, structured: Optional[StructuredSchema]) -> Optional[Dict[str, Any]]:
        if not structured:
            return None
        schema = structured.pydantic_model.model_json_schema() if structured.pydantic_model else (structured.json_schema or {"type": "object"})
        return {"type": "json_schema", "json_schema": {"name": structured.name, "schema": schema, "strict": structured.strict}}

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
        inputs = self._to_responses_input(messages, system)
        reasoning = {"effort": reasoning_effort} if reasoning_effort else None
        return self.client.responses.create(
            model=model,
            input=inputs,
            temperature=temperature,
            tools=self._to_tools(tools),
            response_format=self._to_response_format(structured),
            reasoning=reasoning,
        )

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
        inputs = self._to_responses_input(messages, system)
        reasoning = {"effort": reasoning_effort} if reasoning_effort else None
        with self.client.responses.stream(
            model=model,
            input=inputs,
            temperature=temperature,
            tools=self._to_tools(tools),
            response_format=self._to_response_format(structured),
            reasoning=reasoning,
        ) as stream:
            for event in stream:
                et = getattr(event, "type", "")
                if et.endswith(".output_text.delta"):
                    yield StreamEvent(provider=self.name, type="text.delta", data=event.delta)
                elif ".reasoning." in et or et.endswith(".reasoning.delta"):
                    yield StreamEvent(provider=self.name, type="reasoning.delta", data=getattr(event, "delta", None))
                elif ".function_call." in et or ".tool_call." in et:
                    yield StreamEvent(provider=self.name, type="tool.call.delta", data=event.__dict__)
                elif et == "response.completed":
                    final = event.response
                    yield StreamEvent(provider=self.name, type="usage", data=self.parse_usage(final))
                    yield StreamEvent(provider=self.name, type="message.completed", data=final)
                elif et == "response.error":
                    yield StreamEvent(provider=self.name, type="error", data=event.error)

    def upload_sync(self, *, media: InputMedia, purpose: Optional[str] = None) -> Dict[str, Any]:
        if media.path:
            with open(media.path, "rb") as f:
                created = self.client.files.create(file=f, purpose=purpose or "assistants")
                return created.model_dump()
        if media.uri:
            return {"type": "url", "url": media.uri, "mime_type": media.mime_type}
        raise FeatureNotSupported("OpenAI upload via raw bytes not implemented; use path/URL.")

    def batch_sync(self, *, model: str, requests_file_id: str, endpoint: str = "/v1/responses") -> Dict[str, Any]:
        created = self.client.batches.create(
            input_file_id=requests_file_id,
            endpoint=endpoint,
            completion_window="24h",
        )
        return created.model_dump()

    def parse_usage(self, response: Any) -> Usage:
        u = getattr(response, "usage", None)
        model = getattr(response, "model", "unknown")
        if not u:
            return Usage(provider=self.name, model=model)
        prompt = int(getattr(u, "prompt_tokens", 0))
        completion = int(getattr(u, "completion_tokens", 0))
        total = int(getattr(u, "total_tokens", prompt + completion))
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
