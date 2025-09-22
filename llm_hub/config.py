from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Dict

@dataclass
class Pricing:
    # Optional mapping of token prices (USD) for cost estimates if you want cost, not just usage.
    # If omitted, LLM Hub only records token usage reported by providers.
    # Units: USD per 1K tokens. Provide per-model mapping for accuracy.
    prompt_per_1k: float | None = None
    completion_per_1k: float | None = None
    reasoning_per_1k: float | None = None  # where available (e.g., OpenAI o-series)

@dataclass
class ProviderConfig:
    api_key: Optional[str] = None
    max_retries: Optional[int] = None  # use provider native retry if supported
    timeout: Optional[float] = None    # seconds; forwarded to SDK if supported
    organization: Optional[str] = None # OpenAI org, if applicable
    base_url: Optional[str] = None     # if using Azure/OpenAI compatible endpoints, etc.
    pricing: Dict[str, Pricing] = field(default_factory=dict)  # per-model optional prices

@dataclass
class OpenAIConfig(ProviderConfig):
    pass

@dataclass
class AnthropicConfig(ProviderConfig):
    # Anthropic Files API currently gated via beta header; set to True to enable
    enable_files_beta: bool = True  # uses "anthropic-beta: files-api-2025-04-14"
    anthropic_version: str = "2023-06-01"

@dataclass
class GeminiConfig(ProviderConfig):
    vertexai: bool = False         # use Vertex AI instead of Gemini Developer API
    project: Optional[str] = None  # Vertex only
    location: Optional[str] = None # Vertex only
    api_version: Optional[str] = None  # e.g., "v1" or "v1alpha" via http_options

@dataclass
class HubConfig:
    openai: Optional[OpenAIConfig] = None
    anthropic: Optional[AnthropicConfig] = None
    gemini: Optional[GeminiConfig] = None
    # Tracing
    enable_tracing: bool = False
    tracer_name: str = "llm_hub"
