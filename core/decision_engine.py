from __future__ import annotations

import json
import logging
from datetime import datetime, timezone

from models.decisions import DecisionType, TradeDecision, StructuredEdgeScore, CycleType
from models.market_data import MarketMetrics
from models.portfolio import Portfolio, PositionRecord
from utils.anthropic_client import AnthropicClient
from utils.journal_manager import JournalManager
from utils.earnings_calendar import EarningsCalendar
from core.position_manager import ExitSignal, PositionManager, position_snapshot


class DecisionEngine:
    """Unified decision engine for exit reviews and new-position recommendations."""

    SYSTEM_PROMPT_TEXT = """\
You are an institutional discretionary portfolio manager responsible for
an account. Your single job every time you are called: assess
whether a specific stock is likely to be meaningfully HIGHER in price
1-2 weeks from now than it is today, and size conviction accordingly.
You are not grading whether a stock has been strong recently â€" you are
forecasting forward.

This system runs three cycle types: OPEN, MID, and CLOSE.
- OPEN: highest-conviction new entries and clear exit reviews. Full
  weight given to the historical price/metric trajectory (provided in
  the journal block) and earnings context. Past DECISIONS are
  intentionally not shown â€" do not infer that a prior IGNORE means
  you should IGNORE again; each cycle is independent.
- MID: intraday reassessment; raise the bar for new positions versus OPEN.
- CLOSE: never open new positions. Only hold, sell, or set up tomorrow's
  thesis in your reasoning_summary.

THE CORE FORWARD-LOOKING TEST (apply this before anything else)
For every ticker, before scoring anything, answer this explicitly inside
bull_thesis: "What specific condition â€" news-driven OR purely technical
â€" would cause this stock's price to be higher in 1-2 weeks than it is
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
   accompanying negative news â€" this is a classic rotation or
   mean-reversion, and the "catalyst" is the selling exhaustion itself
   paired with early basing and statistical stretch. When volume on
   down days shrinks instead of expands, sellers are absent, not hiding
   â€" a crucial distinction.

You are NOT required to find a specific dated news article for every
BUY. Strong technical rotation/mean-reversion setups ARE VALID forward
catalysts on their own. This is the central behavioral fix you must
internalize.

SYNTHESIS INSTRUCTION
Synthesize news + technicals + the historical price/metric trajectory
+ portfolio context. Use the trajectory to understand how the stock
has moved (past prices, RSI/MACD/BB/volume trends) â€" NOT to
inherit prior decisions. Each cycle is a fresh evaluation.
Strong technical mean-reversion/rotation setups are valid even with
neutral or light news. The framework allows technicals to drive
high scores when they are compelling â€" news is additive
(confirmation, risk flags, or positive surprise), not a veto.

EDGE SCORING (returned as `edge`)
catalyst_strength (0.0-2.0): this field now captures TWO sources of
forward conviction, not only dated news events.

RATE ACCORDING TO THESE BANDS:
â€¢ Dated, specific, unresolved event (product date explicitly stated
  as coming in 1-2 weeks, product launch, contract win, regulatory
  ruling, guidance revision): up to 2.0, depending on clarity and
  proximity.
â€¢ A strong technical mean-reversion / rotation setup WITH NO MAJOR
  NEGATIVE NEWS (the pattern described below in the recognition
  section, backed by multiple confirming metrics): up to 1.6-1.8.
â€¢ Purely generic background news with no forward trigger
  AND insufficient technical setup to qualify under the technical
  catalyst path above: 0.0-0.5 (hard cap). When the only "catalyst"
  is generic background like "the company makes X products" or
  "structural demand is strong" with no dated event and no qualifying
  technical mean-reversion pattern, it is mathematically impossible
  to reach the {EDGE_THRESHOLD} threshold — this is intentional.

TECHNICAL SETUP RECOGNITION (qualifying patterns for elevated
catalyst_strength up to 1.6-1.8 OR elevated technical_confirmation
up to 1.6-1.8 independently):

These patterns, when multiple confirm, constitute their own
forward thesis — they suggest a rotation/dip-buying opportunity where
the unwind of a low-conviction move creates the catalyst itself:

A) UNEXPLAINED DIP AFTER CONSOLIDATION: stock traded sideways within
   a tight range for weeks (healthy, holding above recent-range
   midpoint), then dips 3-8% with NO meaningful negative headline,
   accompanied by declining volume on the down days. This is rotation /
   mean-reversion — sellers are capitulating on low volume, not
   distributing aggressively.
B) OVERSOLD RSI RECOVERING: rsi_14 dropped below 35 (or even below 30),
   has turned up and is rising through 40 toward 50, showing the
   selling impulse has exhausted and buyers are stepping in.
C) BB_ZSCORE < -1.5: price is statistically stretched to the downside
   relative to its 20-day moving average, indicating a mean-reversion
   setup unless invalidated by negative news.
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
$98 support to $92.12 on volume_spike_of -35% (volume shrinking,
not expanding). No bad news in headlines — only generic descriptive
background about the company's sector. RSI dropped to 32 during the
dip and is now recovering to 41. MACD histogram flipped from -0.40 to
+0.10. BB z-score is -1.7.

→ This is a textbook mean-reversion/rotation setup. News is neutral
but the technical picture is compelling. Catalyst_strength scores
1.4-1.6 for the technical setup alone. Technical_confirmation
should score 1.5-1.7 with multiple confirming signals. With
portfolio_fit of 0.7-0.9, edge.total easily clears {EDGE_THRESHOLD} →
BUY.

Example 2 — When to STILL IGNORE despite technicals:
Stock Y dips 8% in 5 days, RSI at 38, BB zscore -1.3. BUT:
— News explicitly warns of an upcoming binary regulatory ruling in 3 days with unclear outcome.
— OR the dip is on volume_spike_pct of +45% (heavy distribution
  volume on down days = sellers ARE present and aggressive).
— OR realized_vol_30d is above 130% (extreme instability, not a
  controlled pullback).

→ IGNORE. The technical setup is contaminated by high-volume selling, extreme volatility, or a true binary non-earnings risk — these void the mean-reversion thesis.

IMPORTANT about earnings: earnings_flag / the earnings window is INFORMATIONAL. It is a dated catalyst to evaluate (BUY into, HOLD through, or avoid with evidence), NOT an automatic IGNORE or automatic SELL by itself.

technical_confirmation (0.0-2.0): does price action support the forward
thesis specifically, not just "is the stock going up"? A stock already
up 20%+ with RSI above 70 is LATE-stage momentum with reduced room to
run — this should score lower than a stock showing early basing (price
stabilizing after a decline, volume drying up on down days, RSI
recovering from oversold) with the same catalyst. Cite the specific
metric fields that support your forward view, not just direction.

portfolio_fit (0.0-1.0): does this improve diversification, avoid
redundant correlated exposure to existing holdings (check
sector and industry overlap with current positions and the price
trajectory of correlated tickers), and fit within position/capital
limits?

CRITICAL NUMERICAL DISCIPLINE (unchanged, still mandatory)
- All percentages and calculations MUST exactly match the numbers in
  MARKET METRICS. Do not round, invent, or reconstruct a number from
  two other fields unless both source fields are explicitly present.
- Never state a specific underlying number (e.g. "24.8M vs 28.6M") to
  explain a percentage field unless those exact two numbers are present
  as their own separate fields in the JSON. If only the percent is
  given, cite only the percent.
- Only reference metrics that actually exist in the JSON provided.

METRIC DEFINITIONS (use these exact definitions, do not infer others)
- drawdown_30d: peak-to-trough decline within the last 30 trading days
  ONLY, always a positive magnitude. Never write this as negative.
  When citing this value in prose, always say "30-drawdown" — never
  attach any other day-count (e.g. "90-day drawdown") to this number,
  since no drawdown metric other than the 30-day window is computed
  or provided anywhere in this system.
- return_share_5d / return_share_30d / return_share_90d: signed
  percentage price change over that period. These are RETURNS, not
  drawdowns. A negative return_share_90d does not imply any
  "90-day drawdown" field exists — it does not.
- volume_spike_pct: percentage difference between the MOST RECENT
  SINGLE DAY's volume and avg_volume_30d. Not a comparison between
  avg_volume_7d and avg_volume_30d. Cite the percentage directly.
- rsi_14 on 14-day RSI: Below 30 = oversold, above 70 = overbought,
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
- Only recommend opening a position when the computed edge.total >= {EDGE_THRESHOLD}
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

    # Short exit-review rubric injected on all exit / held-review paths (~100 tokens).
    _EXIT_REVIEW_RULES = (
        "EXIT REVIEW RULES:\n"
        "- You MUST state entry vs current and unrealized % in reasoning.\n"
        "- SELL needs explicit evidence: thesis broken, technical failure vs plan, or capital better used elsewhere.\n"
        "- For EXIT decisions, score edge as EXIT CONVICTION (not buy quality): "
        "catalyst_strength = strength of the exit reason (0-2), "
        "technical_confirmation = price action supports exit (0-2), "
        "portfolio_fit = value of freeing capital / cutting risk (0-1).\n"
        "- HOLD is valid when unrealized is small and thesis still intact, including holding through earnings when the opportunity is sound.\n"
        "- STOP_LOSS trigger remains forced SELL (existing code path).\n"
        "- Earnings proximity is informational only — never auto-SELL solely because earnings are near."
    )

    def __init__(self, client: AnthropicClient, journal_manager: JournalManager, earnings_calendar: EarningsCalendar, settings: dict | None = None) -> None:
        self.client = client
        self.journal_manager = journal_manager
        self.earnings_calendar = earnings_calendar
        self._settings = settings or {}
        self._edge_threshold = float(self._settings.get("trading", {}).get("edge_score_threshold", 3.0))
        self.logger = logging.getLogger("robinhood-agent.core.decision_engine")
        # Build the system prompt with the live threshold interpolated
        _prompt_text = self.SYSTEM_PROMPT_TEXT.replace("{EDGE_THRESHOLD}", f"{self._edge_threshold:g}")
        self.system_prompt = [
            {
                "type": "text",
                "text": _prompt_text,
                "cache_control": {"type": "ephemeral"},
            }
        ]

    async def review_exit(
        self, exit_signal: ExitSignal, metrics: MarketMetrics, news: list[str], portfolio: Portfolio, cycle_type: CycleType,
        spy_context: str = "",
        position_record: PositionRecord | None = None,
        open_positions_pnl: dict[str, float] | None = None,
        effective_buying_power: float | None = None,
    ) -> TradeDecision:
        """Review an exit signal and return a SELL or HOLD recommendation.

        The `cycle_type` is included so the model knows OPEN/MID/CLOSE context.
        `position_record` (optional) enables the one-line POSITION snapshot block.
        `open_positions_pnl` (optional) is the {ticker: unrealized_pct} map for
        the whole book, computed ONCE per cycle by PositionManager and reused
        across every ticker's call — this is what makes each call "aware" of
        the other open names without re-sending their full data.
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

        user_prompt = self._build_user_prompt(
            exit_signal, metrics, news, portfolio, cycle_type, spy_context,
            position_record=position_record,
            open_positions_pnl=open_positions_pnl,
            effective_buying_power=effective_buying_power,
        )
        try:
            response = await self.client.complete(self.system_prompt, user_prompt)
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
                reasoning_summary="Fallback review due to error.",
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
        position_record: PositionRecord | None = None,
        unrealized_pnl_pct: float = 0.0,
        days_held: int = 0,
        open_positions_pnl: dict[str, float] | None = None,
    ) -> TradeDecision:
        """Produce a BUY/HOLD/IGNORE/ROTATE recommendation for the requested ticker."""
        user_prompt = self._build_user_prompt_for_buy(
            ticker, metrics, news, portfolio, current_price, effective_buying_power, existing_positions_count, cycle_type, spy_context,
            position_record=position_record, unrealized_pnl_pct=unrealized_pnl_pct, days_held=days_held,
            open_positions_pnl=open_positions_pnl,
        )
        try:
            response = await self.client.complete(self.system_prompt, user_prompt)
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

            # CLOSE-cycle override: do not open new BUYs (size resolved at execution)
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
                risk_notes="Fallback due to API failure.",
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

    def _build_user_prompt(
        self, exit_signal: ExitSignal, metrics: MarketMetrics, news: list[str], portfolio: Portfolio, cycle_type: CycleType, spy_context: str = "",
        position_record: PositionRecord | None = None,
        open_positions_pnl: dict[str, float] | None = None,
        effective_buying_power: float | None = None,
    ) -> str:
        journal_block = ""
        try:
            journal_block = self.journal_manager.summarise_for_prompt(exit_signal.ticker, n=15)
        except Exception:
            journal_block = ""

        earnings_note = ""
        try:
            earnings_note = self.earnings_calendar.earnings_window_for_prompt(exit_signal.ticker)
        except Exception:
            earnings_note = ""

        # One-line POSITION snapshot when we have an open record + current price
        pos_block = ""
        if position_record is not None and exit_signal.current_price > 0:
            try:
                pos_block = position_snapshot(position_record, exit_signal.current_price) + "\n"
            except Exception:
                pos_block = ""

        reason_line = exit_signal.reason
        if isinstance(reason_line, str) and "+" in reason_line:
            parts = [p.replace("_", " ") for p in reason_line.split("+")]
            reason_line = f"{exit_signal.reason} ({' and '.join(parts)})"

        # Compact account context: THIS ticker gets full detail (pos_block
        # above); every other open position gets name + live P&L only, via
        # a map computed once per cycle by PositionManager (positions.json
        # is the single source of truth for what's actually held).
        pnl_map = open_positions_pnl or {}
        roster_line = PositionManager.other_positions_line(pnl_map, exit_signal.ticker)
        bp = effective_buying_power if effective_buying_power is not None else portfolio.buying_power
        account_line = (
            f"ACCOUNT: total_value={portfolio.total_value:.2f} buying_power={bp:.2f} "
            f"cash_pct={portfolio.cash_pct:.1f}%\n{roster_line}\n"
        )

        return (
            (spy_context + "\n" if spy_context else "")
            + pos_block
            + f"CYCLE: {cycle_type.value if isinstance(cycle_type, CycleType) else str(cycle_type)}\n"
            + (earnings_note + "\n" if earnings_note else "")
            + (journal_block + "\n" if journal_block else "")
            + f"EXIT SIGNAL: {reason_line}\n"
            f"CURRENT PRICE: {exit_signal.current_price:.4f}\n"
            f"UNREALIZED P&L: {exit_signal.unrealized_pct * 100:.2f}%\n"
            f"NEWS HEADLINES:\n{json.dumps(news, indent=2)}\n\n"
            f"MARKET METRICS:\n{json.dumps(metrics.model_dump(mode='json'), indent=2)}\n\n"
            f"{account_line}\n"
            + self._EXIT_REVIEW_RULES + "\n"
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
        position_record: PositionRecord | None = None,
        unrealized_pnl_pct: float = 0.0,
        days_held: int = 0,
        open_positions_pnl: dict[str, float] | None = None,
    ) -> str:
        journal_block = ""
        try:
            journal_block = self.journal_manager.summarise_for_prompt(ticker, n=15)
        except Exception:
            journal_block = ""

        earnings_note = ""
        try:
            earnings_note = self.earnings_calendar.earnings_window_for_prompt(ticker)
        except Exception:
            earnings_note = ""

        # One-line POSITION snapshot when reviewing an open position + current price
        pos_block = ""
        if position_record is not None and current_price > 0:
            try:
                pos_block = position_snapshot(position_record, current_price) + "\n"
            except Exception:
                pos_block = ""

        # Single source of truth for "what else is held": positions.json via
        # PositionManager, not the raw live portfolio object. Full detail is
        # given only for `ticker` itself (EXISTING POSITION block below);
        # every other open name gets ticker + live P&L only.
        pnl_map = open_positions_pnl or {}
        roster_line = PositionManager.other_positions_line(pnl_map, ticker)
        weakest_ticker = min(pnl_map, key=pnl_map.get, default=None) if pnl_map else None

        return (
            (spy_context + "\n" if spy_context else "")
            + pos_block
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
            f"ACCOUNT: total_value={portfolio.total_value:.2f}, effective_buying_power={effective_buying_power:.2f}, "
            f"positions_count={existing_positions_count}, max_positions=12\n"
            f"{roster_line}\n\n"
            + (
                f"EXISTING POSITION: {position_record.ticker}\n"
                f"  Entry Price: ${position_record.entry_price:.2f}\n"
                f"  Quantity: {position_record.quantity:.6f}\n"
                f"  Unrealized P&L: {unrealized_pnl_pct * 100:.2f}%\n"
                f"  Days Held: {days_held}\n"
                f"  Stop-Loss: {position_record.stop_loss_pct * 100:.1f}%\n"
                f"  Take-Profit: {position_record.take_profit_pct * 100:.1f}%\n"
                if position_record is not None
                else "EXISTING POSITION: None\n"
            )
            + (f"WEAKEST CURRENT POSITION: {weakest_ticker} unrealized_pnl_pct={pnl_map[weakest_ticker]:.2f}%\n" if weakest_ticker else "WEAKEST CURRENT POSITION: None\n")
            + "Decide whether to BUY, HOLD, IGNORE, or ROTATE."
        )