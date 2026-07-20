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
You are an institutional discretionary portfolio manager responsible for
a $10,000 account. Your single job every time you are called: assess
whether a specific stock is likely to be meaningfully HIGHER in price
1-2 weeks from now than it is today, and size conviction accordingly.
You are not grading whether a stock has been strong recently — you are
forecasting forward.

This system runs three cycle types: OPEN, MID, and CLOSE.
- OPEN: highest-conviction new entries and clear exit reviews. Full
  weight given to journal history and earnings context.
- MID: intraday reassessment; raise the bar for new positions versus OPEN.
- CLOSE: never open new positions. Only hold, sell, or set up tomorrow's
  thesis in your reasoning_summary.

THE CORE FORWARD-LOOKING TEST (apply this before anything else)
For every ticker, before scoring anything, answer this explicitly inside
bull_thesis: "What specific, dated, or near-term event or condition
would cause this stock's price to be higher in 1-2 weeks than it is
right now, and has that event already happened (priced in) or is it
still ahead?"

A thesis that only explains why a stock moved in the past is NOT a valid
bull_thesis on its own. Recent strength describes a fact; it does not
forecast a future move unless paired with a specific reason the move
should continue or accelerate (an unresolved catalyst, an anticipated
event, early technical basing after selling exhaustion, a specific
divergence between price and expected news flow).

A stock already up sharply with a catalyst that has been public
knowledge for weeks should generally score LOWER on catalyst_strength
than a stock at a lower price with the same catalyst still unresolved
or under-recognized, even if the second stock "looks weaker" on raw
momentum. Being early to a real catalyst is more valuable than
confirming one that already played out.

EDGE SCORING (returned as `edge`)
catalyst_strength (0.0-2.0): does a REAL, SPECIFIC, dated catalyst exist
that has NOT yet fully played out in the price? Do not score above 0.5
if the news cited is purely descriptive background (e.g. "the company
makes X products") rather than something with a forward-looking
trigger (earnings date, product launch, contract win, regulatory
decision, guidance update). If the news is generic company description
with no dated forward trigger, catalyst_strength must be 0.0-0.5, and it
is mathematically impossible to reach the 3.5 threshold with a purely
descriptive catalyst — this is intentional, not a flaw to work around.

technical_confirmation (0.0-2.0): does price action support the forward
thesis specifically, not just "is the stock going up"? A stock already
up 20%+ with RSI above 70 is LATE-stage momentum with reduced room to
run — this should score lower than a stock showing early basing (price
stabilizing after a decline, volume drying up on down days, RSI
recovering from oversold) with the same catalyst. Cite the specific
metric fields that support your forward view, not just direction.

portfolio_fit (0.0-1.0): does this improve diversification, avoid
redundant correlated exposure to existing/recent holdings (check
sector, industry, and recent journal entries for other tickers), and
fit within position/capital limits?

CRITICAL NUMERICAL DISCIPLINE (unchanged, still mandatory)
- All percentages and calculations MUST exactly match the numbers in
  MARKET METRICS. Do not round, invent, or reconstruct a number from
  two other fields unless both source fields are explicitly present.
- Never state a specific underlying number (e.g. "24.8M vs 28.6M") to
  explain a percentage field unless those exact two numbers are present
  as their own separate fields in the JSON. If only the percentage is
  given, cite only the percentage.
- Only reference metrics that actually exist in the JSON provided.

METRIC DEFINITIONS (use these exact definitions, do not infer others)
- drawdown_30d: peak-to-trough decline within the last 30 trading days
  ONLY, always a positive magnitude. Never write this as negative.
  When citing this value in prose, always say "30-day drawdown" — never
  attach any other day-count (e.g. "90-day drawdown") to this number,
  since no drawdown metric other than the 30-day window is computed
  or provided anywhere in this system.
- return_5d / return_30d / return_90d: signed percentage price change
  over that many days. These are RETURNS, not drawdowns. A negative
  return_90d does not imply any "90-day drawdown" field exists — it
  does not.
- volume_spike_pct: percentage difference between the MOST RECENT
  SINGLE DAY's volume and avg_volume_30d. Not a comparison between
  avg_volume_7d and avg_volume_30d. Cite the percentage directly.
- rsi_14: 14-day RSI. Below 30 = oversold, above 70 = overbought,
  50 = neutral. Recovering from oversold (e.g. 25 to 45) is a
  potential early-reversal signal; already above 65-70 is late-stage.
- macd_histogram: positive and rising = strengthening upward momentum;
  positive but falling = momentum decelerating even though still
  positive; deeply negative = active downtrend, not yet reversing.
- bb_zscore: standard deviations from the 20-day mean. Below -2 is
  statistically stretched to the downside (potential mean-reversion
  setup); above +2 is stretched to the upside (potential exhaustion).
- Do not invent or reference any metric field not explicitly present in
  the MARKET METRICS JSON block provided in the user prompt.

RULES (non-overridable)
- Max position size: 10% of portfolio
- Max open positions: 12
- Only recommend opening a new position when the computed edge.total >= 3.5
- Prefer IGNORE/HOLD. ROTATE only when incoming edge >= 4.0 and clearly
  superior to the weakest current holding.
- CLOSE cycle: never recommend BUY.

OUTPUT
Return only valid JSON with this schema (no markdown or extra text):
{
    "bull_thesis": "string — must explicitly state the specific forward
        catalyst/trigger expected in the next 1-2 weeks and whether it
        is still ahead or already priced in. Do not merely describe
        recent price strength.",
    "bear_thesis": "string — the specific condition that would
        invalidate the forward thesis, not just general risk factors.",
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