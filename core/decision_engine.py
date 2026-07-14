from __future__ import annotations

import json
import logging
from datetime import datetime, timezone

from models.decisions import DecisionType, TradeDecision, StructuredEdgeScore, CycleType
from models.market_data import MarketMetrics
from models.portfolio import Portfolio
from utils.anthropic_client import AnthropicClient
from utils.journal_manager import JournalManager
from utils.earnings_calendar import EarningsCalendar
from core.position_manager import ExitSignal


class DecisionEngine:
    """Unified decision engine for exit reviews and new-position recommendations."""

    SYSTEM_PROMPT_TEXT = """\
You are an institutional discretionary portfolio manager responsible for a $10,000 account.
This system runs three cycle types: OPEN, MID, and CLOSE. Behavior must adapt by cycle:

- OPEN: prioritize identifying high-conviction new entries and clear exit reviews. Incorporate journal context and earnings notes.
- MID: intraday reassessment; be conservative opening new positions.
- CLOSE: do NOT open NEW BUY positions. Focus on closing or holding existing positions.

Edge (returned as `edge`):
- catalyst_strength: 0.0-2.0 (company-specific news catalyst strength)
- technical_confirmation: 0.0-2.0 (price/volume confirmation)
- portfolio_fit: 0.0-1.0 (how well this trade improves the portfolio)
  (NOTE: sector concentration should now be checked using the real sector field
   provided in the FUNDAMENTALS block — this field is sourced directly from
   Robinhood and is a real, populated value, not a placeholder.)
   
CRITICAL NUMERICAL DISCIPLINE:
- All percentages and calculations MUST exactly match the numbers in MARKET METRICS.
- Volume deficit % = ((avg_volume_30d - current_volume) / avg_volume_30d) * 100. Show exact math.
- Never say "catastrophic -25.7%" if the actual number is -13%. Be precise.
- drawdown_30d is always positive magnitude. Do not confuse return_30d with drawdown.
- Only reference metrics that actually exist in the JSON.
   
METRIC DEFINITIONS (use these exact definitions, do not infer others):
- drawdown_30d: the peak-to-trough decline within the last 30 trading
  days ONLY, expressed as a positive percentage magnitude (e.g. 15.5
  means a 15.5% decline, never write this as a negative number).
- return_5d / return_30d / return_90d: signed percentage price change
  over that many days. These are RETURNS, not drawdowns — a large
  negative return_90d does NOT mean a "90-day drawdown" exists, since
  no such field is computed. Only cite drawdown_30d when discussing
  drawdown, and only for the 30-day window.
- Do not invent or reference any metric field not explicitly present
  in the MARKET METRICS JSON block provided in the user prompt.

Rules (non-overridable):
- Max position size: 8% of portfolio
- Max open positions: 12
- Only recommend opening a new position when the computed edge.total >= 3.5
- Prefer IGNORE/HOLD. ROTATE only when incoming edge >= 4.0 and clearly superior.

Return only valid JSON with this schema (no markdown or extra text):
{
    "bull_thesis": "string",
    "bear_thesis": "string",
    "failure_conditions": ["string"],
    "decision": "BUY|SELL|HOLD|IGNORE|ROTATE",
    "action_type": "NEW|ADD|REDUCE|CLOSE|NONE",
    "rotate_from_ticker": null,
    "edge": {"catalyst_strength": 0.0, "technical_confirmation": 0.0, "portfolio_fit": 0.0},
    "risk_notes": "string",
    "reasoning_summary": "string"
}"""

    # Cached system prompt block: identical across all calls → cache_control breakpoint
    SYSTEM_PROMPT = [
        {
            "type": "text",
            "text": SYSTEM_PROMPT_TEXT,
            "cache_control": {"type": "ephemeral"},
        }
    ]

    def __init__(self, client: AnthropicClient, journal_manager: JournalManager, earnings_calendar: EarningsCalendar) -> None:
        self.client = client
        self.journal_manager = journal_manager
        self.earnings_calendar = earnings_calendar
        self.logger = logging.getLogger("robinhood-agent.core.decision_engine")

    async def review_exit(
        self, exit_signal: ExitSignal, metrics: MarketMetrics, news: list[str], portfolio: Portfolio, cycle_type: CycleType,
        spy_context: str = "",
    ) -> TradeDecision:
        """Review an exit signal and return a SELL or HOLD recommendation.

        The `cycle_type` is included so the model knows OPEN/MID/CLOSE context.
        """
        if exit_signal.reason == "STOP_LOSS":
            return TradeDecision(
                ticker=exit_signal.ticker,
                bull_thesis="Auto exit due to stop-loss threshold.",
                bear_thesis="Risk budget exhausted.",
                failure_conditions=["Stop-loss breach"],
                decision=DecisionType.SELL,
                position_size_pct=1.0,
                action_type="CLOSE",
                replaced_ticker=None,
                edge=StructuredEdgeScore(catalyst_strength=2.0, technical_confirmation=2.0, portfolio_fit=1.0, total=5.0),
                risk_notes="Forced exit by rule.",
                reasoning_summary=f"Auto-exit: {exit_signal.reason}",
                timestamp=datetime.now(timezone.utc).isoformat(),
            )

        user_prompt = self._build_user_prompt(exit_signal, metrics, news, portfolio, cycle_type, spy_context)
        try:
            response = await self.client.complete(self.SYSTEM_PROMPT, user_prompt)
            payload = json.loads(self._strip_fences(response))

            # Normalize edge output: accept either legacy 'edge_score' or new structured 'edge' dict
            edge_obj = payload.get("edge")
            if not isinstance(edge_obj, dict):
                # fallback to legacy edge_score numeric
                edge_score_val = float(payload.get("edge_score", 0.0) or 0.0)
                edge = StructuredEdgeScore(catalyst_strength=0.0, technical_confirmation=0.0, portfolio_fit=0.0, total=edge_score_val)
            else:
                try:
                    edge = StructuredEdgeScore(
                        catalyst_strength=float(edge_obj.get("catalyst_strength", 0.0)),
                        technical_confirmation=float(edge_obj.get("technical_confirmation", 0.0)),
                        portfolio_fit=float(edge_obj.get("portfolio_fit", 0.0)),
                    )
                    edge.compute_total()
                except Exception:
                    edge = StructuredEdgeScore()

            trade = TradeDecision(
                ticker=exit_signal.ticker,
                bull_thesis=str(payload.get("bull_thesis", "")),
                bear_thesis=str(payload.get("bear_thesis", "")),
                failure_conditions=list(payload.get("failure_conditions", [])) if payload.get("failure_conditions") else [],
                decision=DecisionType(payload.get("decision", "IGNORE")),
                edge=edge,
                action_type=str(payload.get("action_type", "NONE")),
                replaced_ticker=payload.get("replaced_ticker", None),
                rotate_from_ticker=payload.get("rotate_from_ticker", None),
                risk_notes=str(payload.get("risk_notes", "")),
                reasoning_summary=str(payload.get("reasoning_summary", "")),
                timestamp=datetime.now(timezone.utc).isoformat(),
                cycle_type=cycle_type.value if isinstance(cycle_type, CycleType) else str(cycle_type),
            )
            return trade
        except Exception as exc:
            self.logger.warning("exit_review_fallback ticker=%s error=%s", exit_signal.ticker, exc)
            return TradeDecision(
                ticker=exit_signal.ticker,
                bull_thesis="Fallback exit review.",
                bear_thesis="Fallback exit review.",
                failure_conditions=["Model unavailable"],
                decision=DecisionType.HOLD,
                position_size_pct=0.0,
                action_type="NONE",
                replaced_ticker=None,
                edge=StructuredEdgeScore(),
                risk_notes="Fallback decision due to API failure.",
                reasoning_summary="Fallback review due to API failure.",
                timestamp=datetime.now(timezone.utc).isoformat(),
            )

    async def make_decision(
        self,
        ticker: str,
        metrics: MarketMetrics,
        news: list[str],
        portfolio: Portfolio,
        current_price: float,
        effective_buying_power: float,
        existing_positions_count: int,
        cycle_type: CycleType,
        spy_context: str = "",
    ) -> TradeDecision:
        """Create a BUY/HOLD/IGNORE/ROTATE recommendation for the requested ticker.

        `cycle_type` is used to adapt the prompt and to enforce CLOSE-cycle overrides.
        """
        user_prompt = self._build_user_prompt_for_buy(
            ticker, metrics, news, portfolio, current_price, effective_buying_power, existing_positions_count, cycle_type, spy_context
        )
        try:
            response = await self.client.complete(self.SYSTEM_PROMPT, user_prompt)
            payload = json.loads(self._strip_fences(response))

            edge_obj = payload.get("edge")
            if not isinstance(edge_obj, dict):
                edge_score_val = float(payload.get("edge_score", 0.0) or 0.0)
                edge = StructuredEdgeScore(catalyst_strength=0.0, technical_confirmation=0.0, portfolio_fit=0.0, total=edge_score_val)
            else:
                try:
                    edge = StructuredEdgeScore(
                        catalyst_strength=float(edge_obj.get("catalyst_strength", 0.0)),
                        technical_confirmation=float(edge_obj.get("technical_confirmation", 0.0)),
                        portfolio_fit=float(edge_obj.get("portfolio_fit", 0.0)),
                    )
                    edge.compute_total()
                except Exception:
                    edge = StructuredEdgeScore()

            decision = TradeDecision(
                ticker=ticker,
                bull_thesis=str(payload.get("bull_thesis", "")),
                bear_thesis=str(payload.get("bear_thesis", "")),
                failure_conditions=list(payload.get("failure_conditions", [])) if payload.get("failure_conditions") else [],
                decision=DecisionType(payload.get("decision", "IGNORE")),
                edge=edge,
                action_type=str(payload.get("action_type", "NONE")),
                replaced_ticker=payload.get("replaced_ticker", None),
                rotate_from_ticker=payload.get("rotate_from_ticker", None),
                risk_notes=str(payload.get("risk_notes", "")),
                reasoning_summary=str(payload.get("reasoning_summary", "")),
                timestamp=datetime.now(timezone.utc).isoformat(),
                cycle_type=cycle_type.value if isinstance(cycle_type, CycleType) else str(cycle_type),
            )

            # Enforce max open positions (decision-level only)
            if existing_positions_count >= 12 and decision.decision == DecisionType.BUY:
                decision.decision = DecisionType.IGNORE

            # Block buys around earnings within 2 days (convert to IGNORE; size is resolved at execution)
            try:
                upcoming = self.earnings_calendar.get_upcoming(ticker, within_days=2)
                if upcoming and decision.decision == DecisionType.BUY:
                    decision.decision = DecisionType.IGNORE
                    decision.reasoning_summary = (decision.reasoning_summary or "") + " | Blocked: upcoming earnings within 2 days."
            except Exception:
                pass

            # CLOSE-cycle override: do not open new BUYs at close (size resolved during execution)
            if isinstance(cycle_type, CycleType) and cycle_type == CycleType.CLOSE and decision.decision == DecisionType.BUY:
                decision.decision = DecisionType.IGNORE
                decision.reasoning_summary = (decision.reasoning_summary or "") + " | Overridden: CLOSE cycle prevents new buys."

            self.logger.info(
                "DECISION_FINAL ticker=%s final_decision=%s edge_total=%.4f cycle=%s",
                ticker, decision.decision.value, decision.edge.total, cycle_type
            )
            return decision
        except Exception as exc:
            self.logger.warning("decision_fallback ticker=%s error=%s", ticker, exc)
            return TradeDecision(
                ticker=ticker,
                bull_thesis="Fallback decision due to API failure.",
                bear_thesis="Fallback decision due to API failure.",
                failure_conditions=["Model unavailable"],
                decision=DecisionType.IGNORE,
                position_size_pct=0.0,
                action_type="NONE",
                replaced_ticker=None,
                edge=StructuredEdgeScore(),
                risk_notes="Fallback decision due to API failure.",
                reasoning_summary="Fallback due to API failure.",
                timestamp=datetime.now(timezone.utc).isoformat(),
            )

    @staticmethod
    def _strip_fences(text: str) -> str:
        cleaned = text.strip()
        cleaned = cleaned.removeprefix("```")
        cleaned = cleaned.removesuffix("```")
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:].lstrip()
        return cleaned.strip()

    def _build_user_prompt(self, exit_signal: ExitSignal, metrics: MarketMetrics, news: list[str], portfolio: Portfolio, cycle_type: CycleType, spy_context: str = "") -> str:
        journal_block = ""
        try:
            journal_block = self.journal_manager.summarise_for_prompt(exit_signal.ticker)
        except Exception:
            journal_block = ""

        earnings_note = ""
        try:
            earnings_note = self.earnings_calendar.earnings_warning_for_prompt(exit_signal.ticker)
        except Exception:
            earnings_note = ""

        # If multiple reasons are encoded with a '+' delimiter, surface them in human-readable form
        reason_line = exit_signal.reason
        if isinstance(reason_line, str) and "+" in reason_line:
            parts = [p.replace("_", " ") for p in reason_line.split("+")]
            reason_line = f"{exit_signal.reason} ({' and '.join(parts)})"

        return (
            (spy_context + "\n" if spy_context else "")
            + f"CYCLE: {cycle_type.value if isinstance(cycle_type, CycleType) else str(cycle_type)}\n"
            + (earnings_note + "\n" if earnings_note else "")
            + (journal_block + "\n" if journal_block else "")
            + f"EXIT SIGNAL: {reason_line}\n"
            f"CURRENT PRICE: {exit_signal.current_price:.4f}\n"
            f"UNREALIZED P&L: {exit_signal.unrealized_pct * 100:.2f}%\n"
            f"NEWS HEADLINES:\n{json.dumps(news, indent=2)}\n\n"
            f"MARKET METRICS:\n{json.dumps(metrics.model_dump(mode='json'), indent=2)}\n\n"
            f"PORTFOLIO SUMMARY:\n{json.dumps(portfolio.model_dump(mode='json'), indent=2)}\n\n"
            + "Decide whether to HOLD or SELL.\n"
        )

    def _build_user_prompt_for_buy(
        self,
        ticker: str,
        metrics: MarketMetrics,
        news: list[str],
        portfolio: Portfolio,
        current_price: float,
        effective_buying_power: float,
        existing_positions_count: int,
        cycle_type: CycleType,
        spy_context: str = "",
    ) -> str:
        journal_block = ""
        try:
            journal_block = self.journal_manager.summarise_for_prompt(ticker)
        except Exception:
            journal_block = ""

        earnings_note = ""
        try:
            earnings_note = self.earnings_calendar.earnings_warning_for_prompt(ticker)
        except Exception:
            earnings_note = ""

        weakest = min(portfolio.positions, key=lambda item: item.unrealized_pnl_pct, default=None)
        return (
            (spy_context + "\n" if spy_context else "")
            + f"CYCLE: {cycle_type.value if isinstance(cycle_type, CycleType) else str(cycle_type)}\n"
            + (earnings_note + "\n" if earnings_note else "")
            + (journal_block + "\n" if journal_block else "")
            + f"TICKER: {ticker}\n"
            f"CURRENT PRICE: {current_price:.4f}\n"
            f"NEWS HEADLINES:\n{json.dumps(news, indent=2)}\n\n"
            f"MARKET METRICS:\n{json.dumps(metrics.model_dump(mode='json'), indent=2)}\n\n"
            f"FUNDAMENTALS:\n"
            f"  Sector: {metrics.sector or 'unknown'}\n"
            f"  Industry: {metrics.industry or 'unknown'}\n"
            f"  Market Cap: ${metrics.market_cap:,.0f}\n"
            f"  P/E Ratio: {metrics.pe_ratio:.2f}\n"
            f"  Dividend Yield: {metrics.dividend_yield:.2f}%\n\n"
            f"PORTFOLIO SUMMARY:\n"
            f"total_value={portfolio.total_value:.2f}, effective_buying_power={effective_buying_power:.2f}, "
            f"positions_count={existing_positions_count}, max_positions=12\n\n"
            f"EXISTING POSITION: None\n"
            + (f"WEAKEST CURRENT POSITION: {weakest.ticker} unrealized_pnl_pct={weakest.unrealized_pnl_pct:.2f}%\n" if weakest else "WEAKEST CURRENT POSITION: None\n")
            + "Decide whether to BUY, HOLD, IGNORE, or ROTATE."
        )