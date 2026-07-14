from __future__ import annotations

from enum import Enum
from pydantic import BaseModel, Field


class DecisionType(str, Enum):
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"
    IGNORE = "IGNORE"
    ROTATE = "ROTATE"


class CycleType(str, Enum):
    OPEN = "OPEN"
    MID = "MID"
    CLOSE = "CLOSE"


class StructuredEdgeScore(BaseModel):
    """Replaces the arbitrary 0-5 float. Each component is defined and bounded.

    Claude outputs all three. Code computes total. No faking.
    """

    catalyst_strength: float = Field(
        0.0,
        ge=0.0,
        le=2.0,
        description=(
            "Real news catalyst present and specific to this company? "
            "0=no catalyst, 1=minor/unclear, 2=clear strong catalyst"
        ),
    )
    technical_confirmation: float = Field(
        0.0,
        ge=0.0,
        le=2.0,
        description=(
            "Do price/volume metrics support the move? "
            "0=against trend, 1=neutral, 2=strong confirmation"
        ),
    )
    portfolio_fit: float = Field(
        0.0,
        ge=0.0,
        le=1.0,
        description=(
            "Does this trade improve the portfolio? "
            "0=redundant/risky, 0.5=neutral, 1=clearly improves"
        ),
    )
    total: float = 0.0

    def compute_total(self) -> "StructuredEdgeScore":
        """Call after Claude fills the three components. Sets total in code."""
        self.total = round(
            min(self.catalyst_strength, 2.0)
            + min(self.technical_confirmation, 2.0)
            + min(self.portfolio_fit, 1.0),
            4,
        )
        return self


class JournalEntry(BaseModel):
    """One timestamped entry in a ticker's running journal."""

    timestamp: str
    cycle_type: str
    bull_thesis: str = ""
    bear_thesis: str = ""
    decision: str = ""
    decision_rationale: str = ""
    key_metrics: dict = Field(default_factory=dict)
    news_used: list[str] = Field(default_factory=list)
    fill_price: float | None = None
    unrealized_pnl_pct: float | None = None
    earnings_flag: bool = False


class TradeDecision(BaseModel):
    ticker: str = Field(..., min_length=1)
    bull_thesis: str = ""
    bear_thesis: str = ""
    failure_conditions: list[str] = Field(default_factory=list)
    decision: DecisionType
    edge: StructuredEdgeScore = Field(default_factory=StructuredEdgeScore)
    position_size_pct: float = 0.0
    action_type: str = "NONE"
    replaced_ticker: str | None = None
    rotate_from_ticker: str | None = None
    risk_notes: str = ""
    reasoning_summary: str = ""
    timestamp: str = ""
    cycle_type: str = ""

    @property
    def edge_score(self) -> float:
        """Backwards-compatible accessor."""
        return self.edge.total


class ExecutionResult(BaseModel):
    ticker: str = Field(..., min_length=1)
    order_id: str = ""
    status: str = "SIMULATED"
    fill_price: float = 0.0
    quantity: float = 0.0
    timestamp: str = ""
    dry_run: bool = True
    side: str = ""


class ExitDecision(BaseModel):
    ticker: str = Field(..., min_length=1)
    decision: DecisionType
    reasoning_summary: str = ""
    edge_score: float = 0.0


__all__ = [
    "DecisionType",
    "CycleType",
    "StructuredEdgeScore",
    "JournalEntry",
    "TradeDecision",
    "ExecutionResult",
    "ExitDecision",
]
