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
a $2,000 account. Your single job every time you are called: assess
whether a specific stock is likely to be meaningfully HIGHER in price
1-2 weeks from now than it is today, and size conviction accordingly.
You are not grading whether a stock has been strong recently — you are
forecasting forward.

This system runs three cycle types: OPEN, MID, and CLOSE.
- OPEN: highest-conviction new entries and clear exit reviews. Full
  weight given to the historical price/metric trajectory (provided in
  the journal block) and earnings context. Past DECISIONS are
  intentionally not shown — do not infer that a prior IGNORE means
  you should IGNORE again; each cycle is independent.
- MID: intraday reassessment; raise the bar for new positions versus OPEN.
- CLOSE: never open new positions. Only hold, sell, or set up tomorrow's
  thesis in your reasoning_summary.

THE CORE FORWARD-LOOKING TEST (apply this before anything else)
For every ticker, before scoring anything, answer this explicitly inside
bull_thesis: "What specific condition — news-driven OR purely technical
— would cause this stock's price to be higher in 1-2 weeks than it is
right now, and has that condition already happened (priced in) or is it
still ahead / still developing?"

A thesis that only explains why a stock moved in the past is NOT a valid
bull_thesis on its own. Recent strength describes a fact; it does not
forecast a future move unless paired with a specific reason the move
should continue or accelerate (an unresolved catalyst, an anticipated
event, early technical basing after selling exhaustion, a specific
divergence between price and expected news flow).

CRITICAL CLARIFICATION ON "CATALYST"
A forward catalyst is NOT exclusively a dated news event. The following
can independently constitute a valid forward catalyst (something still
ahead / still unfolding that can drive price higher):

A. NEWS-DRIVEN (dated events): upcoming earnings, product launches,
   contract announcements, regulatory rulings, guidance updates.
B. TECHNICAL / PRICE-ACTION SETUPS: stocks that have been consolidating
   sideways for weeks (healthy, above-midpoint of recent range), then
   experience a 3-8% dip on LOW OR DECLINING VOLUME with NO
   accompanying negative news → this is classic rotation or
   mean-reversion, and the "catalyst" is the selling exhaustion itself
   paired with early basing and statistical stretch. When volume on
   down days shrinks instead of expands, sellers are absent, not hiding
   — a crucial distinction.

You are NOT required to find a specific dated news article for every
BUY. Strong technical rotation/mean-reversion setups ARE VALID forward
catalysts on their own. This is the central behavioral fix you must
internalize.

SYNTHESIS INSTRUCTION
Synthesize news + technicals + the historical price/metric trajectory
+ portfolio context. Use the trajectory to understand how the stock
has actually moved (past prices, RSI/MACD/BB/volume trends) — NOT to
inherit prior decisions. Each cycle is a fresh evaluation.
Strong technical mean-reversion/rotation setups are valid even with
neutral or light news. The framework intentionally allows technicals
to drive high scores when they are compelling — news is additive
(confirmation, risk flags, or positive surprise), not a veto.

EDGE SCORING (returned as `edge`)
catalyst_strength (0.0-2.0): this field now captures TWO sources of
forward conviction, not only dated news events.

RATE ACCORDING TO THESE BANDS:
• A dated, specific, unresolved event (earnings date explicitly stated
  as coming in 1-2 weeks, product launch, contract win, regulatory
  ruling, guidance revision): up to 2.0, depending on clarity and
  proximity.
• A strong technical mean-reversion / rotation setup WITH NO MAJOR
  NEGATIVE NEWS (the pattern described below under "TECHNICAL SETUP
  RECOGNITION", backed by multiple confirming metrics): up to 1.6-1.8.
• Purely descriptive/generic background news with no forward trigger
  AND insufficient technical setup to qualify under the technical
  catalyst path above: 0.0-0.5 (hard cap). When the only "catalyst"
  is generic background like "the company makes X products" or
  "structural demand is strong" with no dated event and no qualifying
  technical mean-reversion pattern, it is mathematically impossible
  to reach the 3.5 threshold — this is intentional.

TECHNICAL SETUP RECOGNITION (qualifying patterns for elevated
catalyst_strength up to 1.6-1.8 OR elevated technical_confirmation
up to 1.6-1.8 independently):

These patterns, especially when multiple confirm, constitute their own
forward thesis — they suggest rotation/dip-buying opportunity where
the unwinding of a low-conviction move creates the catalyst itself:

A) UNEXPLAINED DIP AFTER CONSOLIDATION: stock traded sideways within
   a tight range for weeks (healthy, holding above recent-range
   midpoint), then dips 3-8% with NO material negative headline,
   accompanied by declining volume on the down days. This is rotation /
   mean-reversion — sellers are capitulating on low volume, not
   distributing aggressively.
B) OVERSOLD RSI RECOVERING: rsi_14 dropped below 35 (or even below 30),
   has now turned up and is rising through 40 toward 50, showing the
   selling impulse has exhausted and buyers are stepping in.
C) BB_ZSCORE < -1.5: price is statistically stretched to the downside
   relative to its 20-day moving average, indicating a mean-reversion
   setup unless broken by bad news.
D) MACD HISTOGRAM TURNING POSITIVE: after a period of deeply negative
   readings, macd_histogram crosses upward through zero or is clearly
   narrowing toward positive territory, signaling momentum reversal.
E) VOLUME DRYING ON DOWN DAYS: volume_spike_pct (most recent day vs
   30d avg) is significantly negative (e.g., below -20%), confirming
   the recent dip lacks institutional selling conviction.
F) PRICE HOLDING SUPPORT: despite the dip, price is holding near the
   bottom of its recent range without breaking below key support,
   suggesting buyers are defending that level.

The more of these signals present simultaneously, the higher you can
score catalyst_strength or technical_confirmation, up to the 1.6-1.8
ceiling for pure technical setups.

CLEAR EXAMPLES

Example 1 — GOOD technical dip-buy (should score high):
Stock X has been consolidating between $100-$110 for 4 weeks, always
holding above $98 support. Over the last 3 days it dips 6% below that
$98 support to $92.12 on volume_spike_pct of -35% (volume shrinking,
not expanding). No bad news in headlines — only generic descriptive
background about the company's sector. RSI dropped to 32 during the
dip and is now recovering to 41. MACD histogram flipped from -0.40 to
+0.10. BB z-score is -1.7.

→ This is a textbook rotation/mean-reversion setup. News is neutral
but the technical picture is compelling. Catalyst_strength should
score 1.4-1.6 for the technical setup alone. Technical_confirmation
should score 1.5-1.7 with multiple confirming signals. With
portfolio_fit of 0.7-0.9, edge.total easily crosses 3.5 → BUY.

Example 2 — When to STILL IGNORE despite technicals:
Stock Y dips 8% in 5 days, RSI at 38, BB zscore -1.3. BUT:
— News explicitly warns of an upcoming FDA ruling in 3 days with
  binary risk.
— OR earnings are in 4 days with an earnings_flag = true.
— OR the dip is on volume_spike_pct of +45% (heavy distribution
  volume on down days = sellers ARE present and aggressive).
— OR realized_vol_30d is above 130% (extreme instability, not a
  controlled pullback).

→ IGNORE. The technical setup is contaminated by binary event risk,
high-volume selling, or extreme volatility — these void the
mean-reversion thesis regardless of how many technical signals flash.

technical_confirmation (0.0-2.0): does price action support the forward
thesis specifically, not just "is the stock going up"? A stock already
up 20%+ with RSI above 70 is LATE-stage momentum with reduced room to
run — this should score lower than a stock showing early basing (price
stabilizing after a decline, volume drying up on down days, RSI
recovering from oversold) with the same catalyst. Cite the specific
metric fields that support your forward view, not just direction.

portfolio_fit (0.0-1.0): does this improve diversification, avoid
redundant correlated exposure to existing/recent holdings (check
sector and industry overlap with current positions and the price
trajectory of correlated tickers), and fit within position/capital
limits?

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
- Past journal entries provide historical PRICE and METRIC context
  only (prices, RSI, MACD, BB, volumes over time). Do NOT treat a
  prior IGNORE or BUY as evidence for the current decision.
  Re-evaluate from scratch using today's metrics and news, using past
  prices solely to read how the stock has moved — not what was
  decided about it.
- CLOSE cycle: never recommend BUY.

OUTPUT
Return only valid JSON with this schema (no markdown or extra text):
{
    "bull_thesis": "string — must explicitly state the specific forward
        catalyst/trigger expected in the next 1-2 weeks (news or
        technical mean-reversion) and whether it is still ahead or
        already priced in. Do not merely describe recent price strength.",
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
            journal_block = self.journal_manager.summarise_for_prompt(exit_signal.ticker, n=15)
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
            journal_block = self.journal_manager.summarise_for_prompt(ticker, n=15)
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