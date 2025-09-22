from __future__ import annotations
from typing import Any, AsyncIterator, Dict, List, Optional, Type
from pydantic import BaseModel
from .config import HubConfig, OpenAIConfig, AnthropicConfig, GeminiConfig
from .providers.base import LLMProvider
from .providers.openai_provider import OpenAIProvider
from .providers.anthropic_provider import AnthropicProvider
from .providers.gemini_provider import GeminiProvider
from .middleware.tracing import traced_span
from .types import Message, Role, ToolSpec, StructuredSchema, InputMedia, StreamEvent, ToolResult
from .usage import Usage, UsageTotals
from .exceptions import ProviderNotAvailable, FeatureNotSupported, StructuredOutputValidationError

class LLMHub:
    """
    Unified async interface:
      - generate(...)
      - stream(...)
      - structured(..., schema=StructuredSchema(pydantic_model=YourModel))
      - upload(...)
      - batch(...)
    """
    def __init__(self, cfg: HubConfig):
        self.cfg = cfg
        self.providers: Dict[str, LLMProvider] = {}
        self.usage = UsageTotals()
        self._init_providers()

    def _init_providers(self) -> None:
        if self.cfg.openai:
            c: OpenAIConfig = self.cfg.openai
            self.providers["openai"] = OpenAIProvider(
                api_key=c.api_key, max_retries=c.max_retries, timeout=c.timeout,
                organization=c.organization, base_url=c.base_url
            )
        if self.cfg.anthropic:
            c: AnthropicConfig = self.cfg.anthropic
            self.providers["anthropic"] = AnthropicProvider(
                api_key=c.api_key, max_retries=c.max_retries, timeout=c.timeout,
                base_url=c.base_url, anthropic_version=c.anthropic_version, enable_files_beta=c.enable_files_beta
            )
        if self.cfg.gemini:
            c: GeminiConfig = self.cfg.gemini
            self.providers["gemini"] = GeminiProvider(
                api_key=c.api_key, max_retries=c.max_retries, timeout=c.timeout,
                vertexai=c.vertexai, project=c.project, location=c.location, api_version=c.api_version
            )

    def _get(self, provider: str) -> LLMProvider:
        if provider not in self.providers:
            raise ProviderNotAvailable(f"Provider '{provider}' not configured")
        return self.providers[provider]

    # ---- core APIs -----------------------------------------------------------

    async def generate(
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
        async with traced_span(self.cfg.enable_tracing, f"{provider}.generate", {"model": model}):
            resp = await prov.generate(
                model=model, messages=msgs, system=system, tools=tools, tool_results=tool_results,
                structured=structured, temperature=temperature, reasoning_effort=reasoning_effort, stream=stream
            )
        if not stream:
            # Track usage for non-stream calls
            try:
                u = prov.parse_usage(resp)
                self.usage.add(u)
            except Exception:
                pass
        return resp

    async def stream(
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
    ) -> AsyncIterator[StreamEvent]:
        prov = self._get(provider)
        msgs = self._normalize_messages(messages)
        async with traced_span(self.cfg.enable_tracing, f"{provider}.stream", {"model": model}):
            async for ev in prov.stream(
                model=model, messages=msgs, system=system, tools=tools, tool_results=tool_results,
                structured=structured, temperature=temperature, reasoning_effort=reasoning_effort
            ):
                # Collect usage event
                if ev.type == "usage" and hasattr(ev, "data") and isinstance(ev.data, Usage):
                    self.usage.add(ev.data)
                yield ev

    async def structured(
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
        resp = await self.generate(
            provider=provider, model=model, messages=messages, system=system,
            structured=schema, temperature=temperature, reasoning_effort=reasoning_effort, stream=False
        )
        # Parse text/JSON from provider response using Pydantic if provided
        model_cls = schema.pydantic_model
        if model_cls:
            # Provider constrained output to JSON via their structured output features.
            # We still validate here to ensure type safety.
            try:
                # OpenAI/Anthropic: extract text
                text = None
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
        # If raw JSON schema only, return raw provider response; callers may parse JSON payload as needed
        return resp

    async def upload(self, *, provider: str, media: InputMedia, purpose: Optional[str] = None) -> Dict[str, Any]:
        prov = self._get(provider)
        return await prov.upload(media=media, purpose=purpose)

    async def batch(self, *, provider: str, model: str, requests_file_id: str, endpoint: Optional[str] = None) -> Dict[str, Any]:
        prov = self._get(provider)
        return await prov.batch(model=model, requests_file_id=requests_file_id, endpoint=endpoint or "")

    # ---- utils ---------------------------------------------------------------

    def _normalize_messages(self, messages: List[Message] | str) -> List[Message]:
        if isinstance(messages, str):
            return [Message(role=Role.USER, content=messages)]
        return messages
