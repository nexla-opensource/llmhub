from __future__ import annotations
from typing import Any, Iterator, Dict, List, Optional
import anthropic
from anthropic import Anthropic
from ..types import Message, InputMedia, ToolSpec, StructuredSchema, StreamEvent, ToolResult
from ..usage import Usage

class AnthropicSyncProvider:
    name = "anthropic"

    def __init__(self, *, api_key: Optional[str], max_retries: Optional[int], timeout: Optional[float], base_url: Optional[str], anthropic_version: str, enable_files_beta: bool):
        headers = {"anthropic-version": anthropic_version}
        if enable_files_beta:
            headers["anthropic-beta"] = "files-api-2025-04-14"
        self.client = Anthropic(
            api_key=api_key,
            base_url=base_url,
            max_retries=max_retries,
            timeout=timeout,
            default_headers=headers,
        )

    def _to_messages(self, messages: List[Message]) -> List[Dict[str, Any]]:
        out = []
        for m in messages:
            out.append({"role": m.role.value, "content": m.content})
        return out

    def _to_tools(self, tools: Optional[List[ToolSpec]]) -> Optional[List[Dict[str, Any]]]:
        if not tools:
            return None
        return [{"name": t.name, "description": t.description or "", "input_schema": t.parameters_json_schema or {"type":"object","properties":{}}} for t in tools]

    def _to_output_schema(self, structured: Optional[StructuredSchema]) -> Optional[Dict[str, Any]]:
        if not structured:
            return None
        schema = structured.pydantic_model.model_json_schema() if structured.pydantic_model else (structured.json_schema or {"type":"object"})
        return {"name": structured.name, "schema": schema, "strict": structured.strict}

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
        return self.client.messages.create(
            model=model,
            messages=self._to_messages(messages),
            system=system,
            temperature=temperature,
            tools=self._to_tools(tools),
            tool_choice="auto" if tools else None,
            output_json_schema=self._to_output_schema(structured),
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
        with self.client.messages.stream(
            model=model,
            messages=self._to_messages(messages),
            system=system,
            temperature=temperature,
            tools=self._to_tools(tools),
            tool_choice="auto" if tools else None,
            output_json_schema=self._to_output_schema(structured),
        ) as stream:
            for event in stream:
                et = getattr(event, "type", "")
                if et == "content_block_delta" and getattr(event.delta, "type", "") == "text_delta":
                    yield StreamEvent(provider=self.name, type="text.delta", data=event.delta.text)
                elif et.endswith("input_json_delta") or et.endswith("tool_use_delta"):
                    yield StreamEvent(provider=self.name, type="tool.call.delta", data=event.__dict__)
                elif et in ("message_delta", "message_stop"):
                    if hasattr(event, "usage") and event.usage:
                        usage = Usage(
                            provider=self.name,
                            model=model,
                            prompt_tokens=getattr(event.usage, "input_tokens", 0) or 0,
                            completion_tokens=getattr(event.usage, "output_tokens", 0) or 0,
                            reasoning_tokens=getattr(event.usage, "thinking_tokens", 0) or 0,
                            total_tokens=(getattr(event.usage, "input_tokens", 0) or 0)
                                         + (getattr(event.usage, "output_tokens", 0) or 0)
                                         + (getattr(event.usage, "thinking_tokens", 0) or 0),
                        )
                        yield StreamEvent(provider=self.name, type="usage", data=usage)
                elif et == "error":
                    yield StreamEvent(provider=self.name, type="error", data=event)

    def upload_sync(self, *, media: InputMedia, purpose: Optional[str] = None) -> Dict[str, Any]:
        if media.path:
            with open(media.path, "rb") as f:
                created = self.client.files.create(file=f)
                return created.model_dump()
        raise ValueError("Anthropic upload expects media.path")

    def batch_sync(self, *, model: str, requests_file_id: str, endpoint: str = "/v1/messages") -> Dict[str, Any]:
        # Minimal pass-through; tailor to your batch flow (file-backed request sets recommended).
        created = self.client.messages.batches.create(requests=[{"custom_id": "batch", "params": {"model": model}}])
        return created.model_dump()

    def parse_usage(self, response: Any) -> Usage:
        u = getattr(response, "usage", None)
        model = getattr(response, "model", "unknown")
        if not u:
            return Usage(provider=self.name, model=model)
        prompt = int(getattr(u, "input_tokens", 0))
        completion = int(getattr(u, "output_tokens", 0))
        thinking = int(getattr(u, "thinking_tokens", 0)) if hasattr(u, "thinking_tokens") else 0
        total = prompt + completion + thinking
        return Usage(provider=self.name, model=model, prompt_tokens=prompt, completion_tokens=completion, reasoning_tokens=thinking, total_tokens=total)
