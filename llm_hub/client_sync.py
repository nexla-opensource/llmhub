from __future__ import annotations
from typing import Any, Dict, List, Optional, Iterator
from pydantic import BaseModel
from .config import HubConfig
from .providers.openai_provider_sync import OpenAISyncProvider
from .providers.anthropic_provider_sync import AnthropicSyncProvider
from .providers.gemini_provider_sync import GeminiSyncProvider
from .middleware.tracing import traced_span_sync
from .types import Message, Role, ToolSpec, StructuredSchema, InputMedia, StreamEvent, ToolResult
from .usage import UsageTotals, Usage
from .exceptions import ProviderNotAvailable, StructuredOutputValidationError

class LLMHubSync:
    def __init__(self, cfg: HubConfig):
        self.cfg = cfg
        self.providers: Dict[str, Any] = {}
        self.usage = UsageTotals()
        self._init_providers()

    def _init_providers(self) -> None:
        if self.cfg.openai:
            c = self.cfg.openai
            self.providers["openai"] = OpenAISyncProvider(
                api_key=c.api_key, max_retries=c.max_retries, timeout=c.timeout,
                organization=c.organization, base_url=c.base_url
            )
        if self.cfg.anthropic:
            c = self.cfg.anthropic
            self.providers["anthropic"] = AnthropicSyncProvider(
                api_key=c.api_key, max_retries=c.max_retries, timeout=c.timeout,
                base_url=c.base_url, anthropic_version=c.anthropic_version, enable_files_beta=c.enable_files_beta
            )
        if self.cfg.gemini:
            c = self.cfg.gemini
            self.providers["gemini"] = GeminiSyncProvider(
                api_key=c.api_key, max_retries=c.max_retries, timeout=c.timeout,
                vertexai=c.vertexai, project=c.project, location=c.location, api_version=c.api_version
            )

    def _get(self, provider: str):
        if provider not in self.providers:
            raise ProviderNotAvailable(f"Provider '{provider}' not configured")
        return self.providers[provider]

    def _normalize_messages(self, messages: List[Message] | str) -> List[Message]:
        if isinstance(messages, str):
            return [Message(role=Role.USER, content=messages)]
        return messages

    # ---- sync core -----------------------------------------------------------

    def generate(
        self,
        *,
        provider: str,
        model: str,
        messages: List[Message] | str,
        system: Optional[str] = None,
        tools: Optional[List[ToolSpec]] = None,
        tool_results: Optional[List[ToolResult]] = None,
        structured: Optional[StructuredSchema] = None,
        temperature: Optional[float] = None,
        reasoning_effort: Optional[str] = None,
        stream: bool = False,
    ) -> Any:
        prov = self._get(provider)
        msgs = self._normalize_messages(messages)
        with traced_span_sync(self.cfg.enable_tracing, f"{provider}.generate.sync", {"model": model}):
            resp_or_iter = prov.generate_sync(
                model=model, messages=msgs, system=system, tools=tools, tool_results=tool_results,
                structured=structured, temperature=temperature, reasoning_effort=reasoning_effort, stream=stream
            )
        if stream:
            return resp_or_iter
        try:
            u = prov.parse_usage(resp_or_iter)
            self.usage.add(u)
        except Exception:
            pass
        return resp_or_iter

    def stream(
        self,
        *,
        provider: str,
        model: str,
        messages: List[Message] | str,
        system: Optional[str] = None,
        tools: Optional[List[ToolSpec]] = None,
        tool_results: Optional[List[ToolResult]] = None,
        structured: Optional[StructuredSchema] = None,
        temperature: Optional[float] = None,
        reasoning_effort: Optional[str] = None,
    ) -> Iterator[StreamEvent]:
        prov = self._get(provider)
        msgs = self._normalize_messages(messages)
        with traced_span_sync(self.cfg.enable_tracing, f"{provider}.stream.sync", {"model": model}):
            for ev in prov.stream_sync(
                model=model, messages=msgs, system=system, tools=tools, tool_results=tool_results,
                structured=structured, temperature=temperature, reasoning_effort=reasoning_effort
            ):
                if ev.type == "usage" and isinstance(ev.data, Usage):
                    self.usage.add(ev.data)
                yield ev

    def structured(
        self,
        *,
        provider: str,
        model: str,
        messages: List[Message] | str,
        schema: StructuredSchema,
        system: Optional[str] = None,
        temperature: Optional[float] = None,
        reasoning_effort: Optional[str] = None,
    ):
        resp = self.generate(
            provider=provider, model=model, messages=messages, system=system,
            structured=schema, temperature=temperature, reasoning_effort=reasoning_effort, stream=False
        )
        model_cls = schema.pydantic_model
        if model_cls:
            try:
                if provider == "openai":
                    text = getattr(resp, "output_text", None) or (resp.output[0].content[0].text if getattr(resp, "output", None) else None)
                elif provider == "anthropic":
                    text = "".join([blk.text for blk in resp.content if getattr(blk, "type", "") == "text"])
                elif provider == "gemini":
                    text = getattr(resp, "text", None)
                payload = text or "{}"
                return model_cls.model_validate_json(payload)
            except Exception as e:
                raise StructuredOutputValidationError(str(e)) from e
        return resp

    def upload(self, *, provider: str, media: InputMedia, purpose: Optional[str] = None) -> Dict[str, Any]:
        return self._get(provider).upload_sync(media=media, purpose=purpose)

    def batch(self, *, provider: str, model: str, requests_file_id: str, endpoint: Optional[str] = None) -> Dict[str, Any]:
        return self._get(provider).batch_sync(model=model, requests_file_id=requests_file_id, endpoint=endpoint or "")
