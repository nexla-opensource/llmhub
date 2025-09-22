from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Dict

@dataclass
class Usage:
    provider: str
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    reasoning_tokens: int = 0  # where available (OpenAI o-series; Anthropic thinking; Gemini thoughts)
    total_tokens: int = 0
    # Optional cost estimates (if Pricing provided)
    estimated_cost_usd: Optional[float] = None
    details: Dict[str, int] = field(default_factory=dict)

@dataclass
class UsageTotals:
    by_provider: Dict[str, Usage] = field(default_factory=dict)

    def add(self, u: Usage) -> None:
        k = f"{u.provider}:{u.model}"
        if k not in self.by_provider:
            self.by_provider[k] = u
            return
        agg = self.by_provider[k]
        agg.prompt_tokens += u.prompt_tokens
        agg.completion_tokens += u.completion_tokens
        agg.reasoning_tokens += u.reasoning_tokens
        agg.total_tokens += u.total_tokens
        # cost sum if available
        if u.estimated_cost_usd:
            agg.estimated_cost_usd = (agg.estimated_cost_usd or 0) + u.estimated_cost_usd
