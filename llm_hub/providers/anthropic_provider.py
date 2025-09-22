from __future__ import annotations
from typing import Any, AsyncIterator, Dict, List, Optional
import anthropic
from anthropic import AsyncAnthropic
from ..types import Message, InputMedia, ToolSpec, StructuredSchema, StreamEvent, ToolResult, Role
from ..usage import Usage

class AnthropicProvider:
    name = "anthropic"

    def __init__(self, *, api_key: Optional[str], max_retries: Optional[int], timeout: Optional[float], base_url: Optional[str], anthropic_version: str, enable_files_beta: bool):
        headers = {"anthropic-version": anthropic_version}
        if enable_files_beta:
            headers["anthropic-beta"] = "files-api-2025-04-14"  # official Files API beta header
        self.client = AsyncAnthropic(
            api_key=api_key,
            base_url=base_url,
            max_retries=max_retries,
            timeout=timeout,
            default_headers=headers,
        )

    def _to_messages(self, messages: List[Message]) -> List[Dict[str, Any]]:
        # Anthropic uses messages=[{role:"user/assistant", content:[{type:"text", text:"..."}|{type:"image", source:{...}}]}]
        out: List[Dict[str, Any]] = []
        for m in messages:
            if isinstance(m.content, str):
                out.append({"role": m.role.value, "content": m.content})
            else:
                out.append({"role": m.role.value, "content": m.content})
        return out

    def _to_tools(self, tools: Optional[List[ToolSpec]]) -> Optional[List[Dict[str, Any]]]:
        if not tools:
            return None
        return [{
            "name": t.name,
            "description": t.description or "",
            "input_schema": t.parameters_json_schema or {"type": "object", "properties": {}},
        } for t in tools]

    def _to_output_schema(self, structured: Optional[StructuredSchema]) -> Optional[Dict[str, Any]]:
        if not structured:
            return None
        schema = structured.pydantic_model.model_json_schema() if structured.pydantic_model else (structured.json_schema or {"type": "object"})
        return {"name": structured.name, "schema": schema, "strict": structured.strict}

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
        msg = self._to_messages(messages)
        tool_defs = self._to_tools(tools)
        output_schema = self._to_output_schema(structured)

        if stream:
            return self.stream(
                model=model, messages=messages, system=system, tools=tools, tool_results=tool_results,
                structured=structured, temperature=temperature, reasoning_effort=reasoning_effort
            )

        resp = await self.client.messages.create(
            model=model,
            messages=msg,
            system=system,
            temperature=temperature,
            tools=tool_defs,
            tool_choice="auto" if tool_defs else None,
            output_json_schema=output_schema,  # official structured output parameter
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
        msg = self._to_messages(messages)
        tool_defs = self._to_tools(tools)
        output_schema = self._to_output_schema(structured)

        async with self.client.messages.stream(
            model=model,
            messages=msg,
            system=system,
            temperature=temperature,
            tools=tool_defs,
            tool_choice="auto" if tool_defs else None,
            output_json_schema=output_schema,
        ) as stream:
            async for event in stream:
                et = getattr(event, "type", "")
                if et == "content_block_delta" and getattr(event.delta, "type", "") == "text_delta":
                    yield StreamEvent(provider=self.name, type="text.delta", data=event.delta.text)
                elif et.endswith("input_json_delta") or et.endswith("tool_use_delta"):
                    yield StreamEvent(provider=self.name, type="tool.call.delta", data=event.__dict__)
                elif et in ("message_delta", "message_stop"):
                    # message completion and usage come across delta/stop; usage also appears via message_delta.usage
                    if hasattr(event, "usage") and event.usage:
                        # event.usage = MessageDeltaUsage(input_tokens=..., output_tokens=..., thinking_tokens?)
                        usage = Usage(
                            provider=self.name,
                            model=model,
                            prompt_tokens=getattr(event.usage, "input_tokens", 0) or 0,
                            completion_tokens=getattr(event.usage, "output_tokens", 0) or 0,
                            reasoning_tokens=getattr(event.usage, "thinking_tokens", 0) or 0,
                            total_tokens=(
                                (getattr(event.usage, "input_tokens", 0) or 0)
                                + (getattr(event.usage, "output_tokens", 0) or 0)
                                + (getattr(event.usage, "thinking_tokens", 0) or 0)
                            ),
                        )
                        yield StreamEvent(provider=self.name, type="usage", data=usage)
                elif et == "message_start":
                    # Anthropic streaming also supports thinking/extended thinking events (thinking_delta)
                    pass
                elif et == "error":
                    yield StreamEvent(provider=self.name, type="error", data=event)

    async def upload(self, *, media: InputMedia, purpose: Optional[str] = None) -> Dict[str, Any]:
        # Anthropic Files API (beta header) supports upload and then referencing via file_id in message content.
        if media.path:
            with open(media.path, "rb") as f:
                created = await self.client.files.create(file=f)  # beta header included in default_headers
                return created.model_dump()
        raise ValueError("Anthropic upload currently expects media.path")

    async def batch(self, *, model: str, requests_file_id: str, endpoint: str = "/v1/messages") -> Dict[str, Any]:
        # Anthropic Message Batches API
        created = await self.client.messages.batches.create(requests=[{"custom_id": "batch", "params": {"model": model}}],  # usually you stream requests via file/results API; kept minimal here
        )
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
