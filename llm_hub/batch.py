from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

@dataclass
class BatchJob:
    provider: str
    id: str
    status: str
    results_url: Optional[str] = None
