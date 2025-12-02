"""Cost estimation helpers for Semrush API usage."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class CostEstimate:
    action: str
    price_per_line: int
    lines: int

    @property
    def total(self) -> int:
        return self.price_per_line * self.lines


# Derived from api_cost_reference.md. Values are conservative defaults and can be
# updated as Semrush publishes more granular pricing data.
ACTION_COST_MAP: Dict[str, CostEstimate] = {}

_COST_CONFIG = {
    "get_keyword_overview": (10, 1),
    "get_organic_keywords": (10, 10),
    "get_competitors": (10, 10),
    "get_domain_overview": (10, 1),
    "get_backlinks": (10, 10),
}

for action_name, (price, default_lines) in _COST_CONFIG.items():
    ACTION_COST_MAP[action_name] = CostEstimate(
        action=action_name,
        price_per_line=price,
        lines=default_lines,
    )


def estimate_cost(
    action_name: str, *, display_limit: Optional[int] = None
) -> Optional[CostEstimate]:
    baseline = ACTION_COST_MAP.get(action_name)
    if not baseline:
        return None
    lines = display_limit or baseline.lines
    return CostEstimate(
        action=baseline.action,
        price_per_line=baseline.price_per_line,
        lines=lines,
    )

