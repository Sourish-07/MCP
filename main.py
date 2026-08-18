from __future__ import annotations

import asyncio
import json
import logging
import os
from datetime import datetime, timezone, time, timedelta
from pathlib import Path
from typing import Optional

import pytz
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from dotenv import load_dotenv

from core.data_ingest import DataIngestLayer
from core.decision_engine import DecisionEngine
from core.execution import ExecutionEngine
from core.metrics import MetricsEngine
from core.news_fetch import NewsFetcher
from models.news import NewsItem
from core.pnl_tracker import PnLTracker
from core.position_manager import PositionManager, ExitSignal, position_snapshot
from robinhood_mcp.robinhood_client import RobinhoodMCPClient
from models.decisions import DecisionType, CycleType, JournalEntry, TradeDecision, StructuredEdgeScore
from models.market_data import MarketMetrics
from models.portfolio import Portfolio, PositionRecord
from utils.journal_manager import JournalManager
from utils.earnings_calendar import EarningsCalendar
from utils.anthropic_client import AnthropicClient
from utils.cost_tracker import CostLimitExceededError, CostTracker
from utils.logger import configure_logging


# Short exit-review rubric injected on monitor / exit review paths (~100 tokens).
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


# Compact monitor-only system prompt for the 1-minute threshold path (~300 tokens).
# Replaces the long DecisionEngine system prompt on monitor complete() calls so
# repeated STOP_LOSS / PROFIT_TARGET / PROFIT_REVIEW reviews bill far less input.
_MONITOR_SYSTEM_PROMPT_TEXT = """\
You are reviewing an EXISTING open equity position after a price threshold fired.
Your only job: decide HOLD or SELL for this one ticker. Do not recommend BUY, IGNORE, or ROTATE.

Always ground reasoning in entry price vs current price and unrealized P&L %.
SELL requires explicit evidence (thesis broken, technical failure vs plan, or capital better deployed).
HOLD is valid when the thesis still holds and the move is noise or still unfolding.

For EXIT decisions, score edge as EXIT CONVICTION (not buy quality):
- catalyst_strength (0-2): strength of the exit reason
- technical_confirmation (0-2): price action supports exit
- portfolio_fit (0-1): value of freeing capital / cutting risk
Code will sum these; soft sells still need total >= the account edge threshold to execute.
STOP_LOSS is a hard risk rule: decision must be SELL.

Earnings proximity is informational only — not an automatic SELL.
Cite only metrics present in the user message. Do not invent numbers.

Return ONLY valid JSON (no markdown):
{
  "bull_thesis": "string",
  "bear_thesis": "string",
  "failure_conditions": ["string"],
  "decision": "HOLD|SELL",
  "action_type": "NONE|CLOSE",
  "rotate_from_ticker": null,
  "edge": {"catalyst_strength": 0.0, "technical_confirmation": 0.0, "portfolio_fit": 0.0},
  "risk_notes": "string",
  "reasoning_summary": "string — must mention entry vs current and unrealized %"
}
"""

_MONITOR_SYSTEM_PROMPT = [
    {
        "type": "text",
        "text": _MONITOR_SYSTEM_PROMPT_TEXT,
        "cache_control": {"type": "ephemeral"},
    }
]


# Per-ticker cooldowns for SOFT monitor signals so a stuck >=15% / >=20% gain
# does not re-query the LLM every 60 seconds after a HOLD. STOP_LOSS is a hard
# risk rule and is NEVER throttled.
MONITOR_SOFT_COOLDOWN = timedelta(minutes=15)   # PROFIT_REVIEW
MONITOR_TARGET_COOLDOWN = timedelta(minutes=5)  # PROFIT_TARGET (closer to exit)


class TradingAgent:
    """Anthropic-only trading agent orchestrating ingest, metrics, news, decisions,
    and execution.  Provides a three-cycle schedule plus a 60-second price monitor
    for real-time stop-loss / take-profit enforcement on held positions."""

    def __init__(self) -> None:
        self.logger = logging.getLogger("robinhood-agent.main")
        self.scheduler = AsyncIOScheduler(timezone="America/New_York")
        self.config = self._load_config()
        self.cost_tracker = CostTracker()
        self.mcp_client = RobinhoodMCPClient()
        self.ingest = DataIngestLayer(self.mcp_client)
        self.metrics_engine = MetricsEngine()
        self.news_fetcher = NewsFetcher(
            AnthropicClient(model=self.config["models"]["cheap"]),
            finnhub_api_key=os.getenv("FINNHUB_API_KEY", ""),
            settings=self.config.get("news", {}),
        )
        self.journal_manager = JournalManager()
        self.earnings_calendar = EarningsCalendar()
        self.earnings_calendar.set_client(self.mcp_client)
        self.decision_engine = DecisionEngine(
            AnthropicClient(model=self.config["models"]["decision"], max_tokens=4096),
            self.journal_manager,
            self.earnings_calendar,
            settings=self.config,
        )
        self.position_manager = PositionManager()
        self.pnl_tracker = PnLTracker(self.mcp_client)
        self.executor = ExecutionEngine(self.mcp_client, self.position_manager, self.config)
        self._cached_watchlist: list[str] = []
        self._cached_watchlist_date: str = ""

        # ---- 1-minute price monitor state (sections 3 / 5 / 6) ----
        self._monitor_locks: dict[str, asyncio.Lock] = {}
        self._monitor_cache: dict[str, tuple[float, MarketMetrics, list[str]]] = {}
        self._exit_in_progress: set[str] = set()
        # Per-ticker last soft-signal evaluation timestamp (UTC). Used to
        # throttle PROFIT_REVIEW / PROFIT_TARGET after a HOLD so the LLM is
        # not re-queried every tick. STOP_LOSS ignores this entirely.
        self._last_monitor_review: dict[str, datetime] = {}

    # ==================================================================
    #  CONFIG
    # ==================================================================

    def _load_config(self) -> dict:
        load_dotenv(".env", override=False)
        base = Path(__file__).resolve().parent
        with (base / "config" / "settings.json").open("r", encoding="utf-8") as handle:
            settings = json.load(handle)
        dry_run = os.getenv("DRY_RUN", str(settings.get("trading", {}).get("dry_run", True))).lower() != "false"
        settings["env"] = {
            "ANTHROPIC_API_KEY": os.getenv("ANTHROPIC_API_KEY", ""),
            "ROBINHOOD_MCP_URL": os.getenv("ROBINHOOD_MCP_URL", ""),
            "DRY_RUN": dry_run,
        }
        return settings

    async def _resolve_watchlist(self) -> list[str]:
        watchlist_name = self.config.get("watchlist", {}).get("robinhood_watchlist_name", "Default")
        live_tickers = await self.mcp_client.get_watchlist_tickers(watchlist_name)
        if live_tickers:
            tickers = live_tickers[:20]
            self.logger.info("watchlist_source=robinhood_mcp count=%d name=%s", len(tickers), watchlist_name)
            return tickers
        fallback = list(self.config["watchlist"]["default_tickers"])
        self.logger.info("watchlist_source=settings_json count=%d (MCP returned empty)", len(fallback))
        return fallback

    def _is_market_open(self) -> bool:
        ny_tz = pytz.timezone(self.config["schedule"]["timezone"])
        now = datetime.now(ny_tz)
        if now.weekday() >= 5:
            return False
        open_t = now.replace(hour=9, minute=30, second=0, microsecond=0)
        close_t = now.replace(hour=16, minute=0, second=0, microsecond=0)
        return open_t <= now <= close_t

    def _earnings_flag(self, ticker: str) -> bool:
        try:
            return self.earnings_calendar.is_in_earnings_window(ticker)
        except Exception:
            return False

    def _earnings_window_block(self, ticker: str) -> str:
        try:
            return self.earnings_calendar.earnings_window_for_prompt(ticker)
        except Exception:
            return ""

    # ==================================================================
    #  PRICE MONITOR — 1-MINUTE LIGHTWEIGHT LOOP (sects 3/4/5/6)
    # ==================================================================

    def _monitor_time_window(self, now_et: datetime) -> str:
        """Return 'MARKET' | 'PRE' | 'POST' | 'CLOSED'."""
        t = now_et.time()
        if time(9, 30) <= t <= time(16, 0):
            return "MARKET"
        if time(4, 0) <= t <= time(9, 29):
            return "PRE"
        if time(16, 1) <= t <= time(20, 0):
            return "POST"
        return "CLOSED"

    async def _fetch_monitor_metrics(self, ticker: str) -> MarketMetrics:
        """Fetch current metrics for a monitor threshold review."""
        try:
            bars = await self.mcp_client.get_equity_historicals(
                ticker, span="month", interval="day",
            )
            if bars:
                return await self.metrics_engine.compute(ticker, bars, self.mcp_client)
        except Exception:
            pass
        if ticker in self._monitor_cache:
            _, cached_m, _ = self._monitor_cache[ticker]
            return cached_m
        return MarketMetrics(ticker=ticker)

    async def _fetch_monitor_news(self, ticker: str) -> list[str]:
        """Fetch Finnhub news for a monitor review; falls back empty."""
        try:
            fetched = await self.news_fetcher.fetch_news([ticker], cycle_type="MONITOR")
            raw_items = fetched.get(ticker, [])
            if raw_items and isinstance(raw_items[0], NewsItem):
                return [
                    f"[{n.source}] {n.headline}"
                    + (f" — {n.summary}" if n.summary else "")
                    for n in raw_items
                ]
        except Exception:
            pass
        if ticker in self._monitor_cache:
            _, _, cached_news = self._monitor_cache[ticker]
            return cached_news
        return []

    async def _monitor_refresh_data(
        self, ticker: str, window: str,
    ) -> tuple[MarketMetrics, list[str]]:
        """Return (metrics, news) for a monitor trigger, respecting the
        out-of-cycle windows (PRE/POST → fetch MCP + Finnhub fresh).
        Falls back to stale cache; never invents numbers."""
        now_ts = datetime.now(timezone.utc).timestamp()

        if ticker in self._monitor_cache:
            ts, cached_m, cached_news = self._monitor_cache[ticker]
            if now_ts - ts < 900:  # 15 minutes
                return cached_m, cached_news

        if window in ("PRE", "POST"):
            m = await self._fetch_monitor_metrics(ticker)
            nl = await self._fetch_monitor_news(ticker)
        else:
            m = await self._fetch_monitor_metrics(ticker)
            nl = await self._fetch_monitor_news(ticker)

        self._monitor_cache[ticker] = (now_ts, m, nl)
        return m, nl

    async def _evaluate_threshold_signal(
        self,
        signal: ExitSignal,
        record: PositionRecord,
        cycle_label: str,
        metrics: MarketMetrics,
        news: list[str],
        portfolio: Portfolio,
        spy_context: str,
        open_positions_pnl: dict[str, float] | None = None,
    ) -> TradeDecision:
        """Compact re-evaluation prompt for a threshold hit (STOP_LOSS,
        PROFIT_TARGET, PROFIT_REVIEW).  STOP_LOSS is forced SELL; the
        model is consulted only for journal/logging.

        The other two are soft — model decides HOLD vs SELL.
        """
        journal_block = ""
        try:
            journal_block = self.journal_manager.summarise_for_prompt(signal.ticker, n=15)
        except Exception:
            journal_block = ""

        try:
            days_held = (datetime.now(timezone.utc).date()
                         - datetime.fromisoformat(record.entry_date).date()).days
        except Exception:
            days_held = 0

        unrealized_pct = signal.unrealized_pct

        # One-line POSITION snapshot + EARNINGS WINDOW block + EXIT REVIEW RULES
        pos_block = ""
        try:
            pos_block = position_snapshot(record, signal.current_price) + "\n"
        except Exception:
            pos_block = ""
        earnings_block = self._earnings_window_block(signal.ticker)

        # Compact account context: this ticker's full detail is already in
        # pos_block above; positions.json (via PositionManager) is the
        # single source of truth for everything else — name + live P&L only.
        pnl_map = open_positions_pnl or {}
        roster_line = PositionManager.other_positions_line(pnl_map, signal.ticker)
        account_line = f"ACCOUNT: total_value={portfolio.total_value:.2f} cash_pct={portfolio.cash_pct:.1f}%\n{roster_line}\n"

        user_prompt = (
            f"CYCLE: {cycle_label}\n"
            + pos_block
            + (earnings_block + "\n" if earnings_block else "")
            + f"TRIGGER: {signal.reason}\n"
            f"TICKER: {signal.ticker}\n"
            f"ENTRY_PRICE: {record.entry_price:.2f}\n"
            f"CURRENT_PRICE: {signal.current_price:.2f}\n"
            f"UNREALIZED_PNL_PCT: {unrealized_pct * 100:.2f}%\n"
            f"DAYS_HELD: {days_held}\n"
            f"STOP_LOSS_PCT: {record.stop_loss_pct * 100:.1f}%\n"
            f"TAKE_PROFIT_PCT: {record.take_profit_pct * 100:.1f}%\n"
            f"POSITION QUANTITY: {record.quantity:.6f}\n\n"
            f"JOURNAL HISTORY (last 15 entries - price & metric trajectory only):\n"
            f"{journal_block}\n\n"
            f"LATEST METRICS:\n{json.dumps(metrics.model_dump(mode='json'), indent=2)}\n\n"
            f"LATEST NEWS:\n{json.dumps(news, indent=2)}\n\n"
            f"{account_line}\n"
            + "You previously entered this position. A threshold has been crossed.\n"
            "Re-evaluate using ALL prior journal context plus the fresh data above.\n"
            "Decide strictly HOLD or SELL. Return only the JSON schema required by the system prompt.\n"
            + ("For STOP_LOSS the decision must be SELL.\n" if signal.reason == "STOP_LOSS" else "")
        )

        # ----- STOP_LOSS: forced SELL (model consulted for journal only) -----
        if signal.reason == "STOP_LOSS":
            model_explanation = f"Auto-exit: {signal.reason}"
            try:
                response = await self.decision_engine.client.complete(
                    _MONITOR_SYSTEM_PROMPT, user_prompt,
                )
                payload = json.loads(self.decision_engine._strip_fences(response))
                model_explanation = str(payload.get("reasoning_summary", model_explanation))
            except Exception as exc:
                self.logger.warning("monitor_stop_loss_model_failed ticker=%s error=%s", signal.ticker, exc)
            return TradeDecision(
                ticker=signal.ticker,
                bull_thesis="Auto exit due to stop-loss threshold.",
                bear_thesis="Risk budget exhausted.",
                failure_conditions=["Stop-loss breach"],
                decision=DecisionType.SELL,
                position_size_pct=1.0,
                action_type="CLOSE",
                replaced_ticker=None,
                edge=StructuredEdgeScore(catalyst_strength=2.0, technical_confirmation=2.0, portfolio_fit=1.0, total=5.0),
                risk_notes="Forced exit by rule.",
                reasoning_summary=f"MONITOR_STOP_LOSS: {model_explanation}",
                timestamp=datetime.now(timezone.utc).isoformat(),
                cycle_type="MONITOR",
            )

        # ----- Soft threshold: model decides HOLD vs SELL -----
        try:
            response = await self.decision_engine.client.complete(
                _MONITOR_SYSTEM_PROMPT, user_prompt,
            )
            payload = json.loads(self.decision_engine._strip_fences(response))
            decision_str = str(payload.get("decision", "HOLD"))
            edge_obj = payload.get("edge")
            if not isinstance(edge_obj, dict):
                edge_val = float(payload.get("edge_score", 0.0) or 0.0)
                edge = StructuredEdgeScore(catalyst_strength=0.0, technical_confirmation=0.0, portfolio_fit=0.0, total=edge_val)
            else:
                edge = StructuredEdgeScore(
                    catalyst_strength=float(edge_obj.get("catalyst_strength", 0.0)),
                    technical_confirmation=float(edge_obj.get("technical_confirmation", 0.0)),
                    portfolio_fit=float(edge_obj.get("portfolio_fit", 0.0)),
                )
                edge.compute_total()

            return TradeDecision(
                ticker=signal.ticker,
                bull_thesis=str(payload.get("bull_thesis", "")),
                bear_thesis=str(payload.get("bear_thesis", "")),
                failure_conditions=list(payload.get("failure_conditions", [])) if payload.get("failure_conditions") else [],
                decision=DecisionType(decision_str),
                edge=edge,
                action_type="CLOSE" if decision_str == "SELL" else "NONE",
                replaced_ticker=None,
                rotate_from_ticker=None,
                risk_notes=str(payload.get("risk_notes", "")),
                reasoning_summary=f"MONITOR_{signal.reason}: {payload.get('reasoning_summary', '')}",
                timestamp=datetime.now(timezone.utc).isoformat(),
                cycle_type="MONITOR",
            )
        except Exception as exc:
            self.logger.warning("monitor_eval_failed ticker=%s error=%s", signal.ticker, exc)
            return TradeDecision(
                ticker=signal.ticker,
                bull_thesis="Fallback monitor decision.",
                bear_thesis="Fallback monitor decision.",
                failure_conditions=["Monitor model unavailable"],
                decision=DecisionType.HOLD,
                position_size_pct=0.0,
                action_type="NONE",
                replaced_ticker=None,
                edge=StructuredEdgeScore(),
                risk_notes="Monitor fallback due to API failure.",
                reasoning_summary="MONITOR_FALLBACK: model unavailable",
                timestamp=datetime.now(timezone.utc).isoformat(),
                cycle_type="MONITOR",
            )

    async def run_price_monitor(self) -> None:
        """1-minute loop: pull quotes for held tickers, check thresholds, act.

        Only active during normal market hours (09:30–16:00 ET).
        Independent of the OPEN/MID/CLOSE scheduled cycles.
        """
        ny_tz = pytz.timezone(self.config["schedule"]["timezone"])
        now_et = datetime.now(ny_tz)
        window = self._monitor_time_window(now_et)

        if window != "MARKET":
            return

        try:
            open_records = self.position_manager.get_open_records()
        except Exception as exc:
            self.logger.warning("monitor_records_load_failed: %s", exc)
            return

        if not open_records:
            return

        open_tickers = [r.ticker for r in open_records]

        # cheap: quotes only
        try:
            raw_quotes = await self.mcp_client.get_equity_quotes(open_tickers)
        except Exception as exc:
            self.logger.warning("monitor_quote_fetch_failed: %s", exc)
            return

        quotes_map: dict[str, float] = {}
        for q in (raw_quotes or []):
            try:
                ticker = getattr(q, "ticker", "")
                price = float(getattr(q, "price", 0.0) or 0.0)
                if ticker and price > 0:
                    quotes_map[ticker] = price
            except Exception:
                continue

        # single-pass P&L map for the whole open book (positions.json is the
        # source of truth for which tickers are held; quotes_map supplies
        # live price) — computed once per tick, reused for every signal.
        pnl_map = self.position_manager.open_positions_pnl_map(quotes_map)

        signals: list[ExitSignal] = []
        for record in open_records:
            price = quotes_map.get(record.ticker)
            if price is None:
                self.logger.warning("monitor_quote_missing ticker=%s", record.ticker)
                continue
            if record.entry_price <= 0:
                continue
            unrealized_pct = (price - record.entry_price) / record.entry_price

            if unrealized_pct <= record.stop_loss_pct:
                signals.append(ExitSignal(record.ticker, "STOP_LOSS", price, unrealized_pct, record.quantity))
            elif unrealized_pct >= record.take_profit_pct:
                signals.append(ExitSignal(record.ticker, "PROFIT_TARGET", price, unrealized_pct, record.quantity))
            elif unrealized_pct >= PositionManager.REVIEW_GAIN_PCT:
                signals.append(ExitSignal(record.ticker, "PROFIT_REVIEW", price, unrealized_pct, record.quantity))

        self.logger.info("price_monitor tick open_positions=%d signals=%d", len(open_records), len(signals))

        if not signals:
            return

        # light portfolio snapshot for the compact prompt
        try:
            portfolio = await self.mcp_client.get_portfolio()
        except Exception:
            portfolio = Portfolio()

        for signal in signals:
            ticker = signal.ticker
            lock = self._monitor_locks.setdefault(ticker, asyncio.Lock())

            if ticker in self._exit_in_progress:
                self.logger.debug("monitor_skip_locked ticker=%s", ticker)
                continue

            # ---- Soft-signal cooldown (PROFIT_REVIEW / PROFIT_TARGET) ----
            # STOP_LOSS is a hard risk rule: never skipped for cooldown.
            # Combined soft reasons (e.g. "PROFIT_REVIEW+MAX_HOLDING_PERIOD")
            # are treated as soft and use MONITOR_SOFT_COOLDOWN.
            if signal.reason != "STOP_LOSS":
                now_utc = datetime.now(timezone.utc)
                if "PROFIT_TARGET" in signal.reason:
                    cooldown = MONITOR_TARGET_COOLDOWN
                else:
                    # PROFIT_REVIEW and any soft combined form fall back to the
                    # longer cooldown.
                    cooldown = MONITOR_SOFT_COOLDOWN
                last = self._last_monitor_review.get(ticker)
                if last is not None and (now_utc - last) < cooldown:
                    remaining = (cooldown - (now_utc - last)).total_seconds()
                    self.logger.info(
                        "monitor_skip_cooldown ticker=%s reason=%s remaining_s=%.0f",
                        ticker, signal.reason, remaining,
                    )
                    continue

            self._exit_in_progress.add(ticker)

            try:
                async with lock:
                    record = next((r for r in open_records if r.ticker == ticker), None)
                    if record is None:
                        continue

                    # Refresh data appropriate for the current window
                    metrics, news_list = await self._monitor_refresh_data(ticker, window)

                    decision = await self._evaluate_threshold_signal(
                        signal=signal,
                        record=record,
                        cycle_label="MONITOR",
                        metrics=metrics,
                        news=news_list,
                        portfolio=portfolio,
                        spy_context="",
                        open_positions_pnl=pnl_map,
                    )

                    # Stamp the soft-signal evaluation timestamp AFTER the
                    # model returns (HOLD or SELL, and also on the fallback
                    # HOLD path inside _evaluate_threshold_signal) so a failed
                    # call does not immediately retry every tick. STOP_LOSS
                    # is intentionally never stamped.
                    if signal.reason != "STOP_LOSS":
                        self._last_monitor_review[ticker] = datetime.now(timezone.utc)

                    # Journal the monitor decision with consistent fill/unrealized/earnings fields
                    try:
                        entry = JournalEntry(
                            timestamp=datetime.now(timezone.utc).isoformat(),
                            cycle_type="MONITOR",
                            bull_thesis=decision.bull_thesis,
                            bear_thesis=decision.bear_thesis,
                            decision=decision.decision.value,
                            decision_rationale=decision.reasoning_summary,
                            key_metrics=metrics.model_dump(mode="json") if hasattr(metrics, "model_dump") else {},
                            news_used=news_list,
                            fill_price=record.entry_price,
                            unrealized_pnl_pct=signal.unrealized_pct,
                            earnings_flag=self._earnings_flag(ticker),
                        )
                        self.journal_manager.append(ticker, entry)
                    except Exception:
                        pass

                    if decision.decision == DecisionType.SELL:
                        result = await self.executor.execute(
                            decision, portfolio, signal.current_price,
                            dry_run=self.config["env"]["DRY_RUN"],
                            vol_30d=getattr(metrics, "realized_vol_30d", 0.0) or 0.0,
                            avg_volume_30d=getattr(metrics, "avg_volume_30d", 0.0) or 0.0,
                            realized_vol_30d=getattr(metrics, "realized_vol_30d", 0.0) or 0.0,
                        )
                        # Executor already calls record_realized_pnl + record_exit on
                        # EXECUTED / SUBMITTED / WOULD_EXECUTE (dry-run). Do NOT call record_exit again.
                        if result.status in ("EXECUTED", "SUBMITTED", "WOULD_EXECUTE"):
                            pnl_map.pop(ticker, None)
                            # Position is closed: clear cooldown timestamp so a
                            # future re-entry starts clean (no stale throttle).
                            self._last_monitor_review.pop(ticker, None)
                            try:
                                self.position_manager.write_performance_snapshot(quotes_map)
                            except Exception:
                                pass
                            self.logger.info(
                                "monitor_sell_completed ticker=%s status=%s",
                                ticker, result.status,
                            )
            except Exception as exc:
                self.logger.error("monitor_signal_processing_error ticker=%s error=%s", ticker, exc)
            finally:
                self._exit_in_progress.discard(ticker)

    # ==================================================================
    #  MAIN CYCLE  (sections 1 / 2)
    # ==================================================================

    async def run_cycle(self, cycle_type: CycleType) -> None:
        """Run one full trading cycle (OPEN / MID / CLOSE)."""
        cycle_str = cycle_type.value if isinstance(cycle_type, CycleType) else str(cycle_type)
        self.logger.info("=== cycle_start cycle=%s dry_run=%s ===", cycle_str, self.config["env"]["DRY_RUN"])

        try:
            self.cost_tracker.check_limits(
                self.config["cost_limits"]["daily_usd_limit"],
                self.config["cost_limits"]["monthly_usd_limit"],
            )
        except CostLimitExceededError as exc:
            self.logger.error("cycle_aborted cost_limit: %s", exc)
            return

        if not self._is_market_open():
            self.logger.info("cycle_skipped market_closed")
            return

        today = datetime.now(timezone.utc).date().isoformat()
        if self._cached_watchlist_date == today and self._cached_watchlist:
            watchlist_tickers = self._cached_watchlist
        else:
            watchlist_tickers = await self._resolve_watchlist()
            self._cached_watchlist = watchlist_tickers
            self._cached_watchlist_date = today

        open_records = self.position_manager.get_open_records()
        existing_tickers = [r.ticker for r in open_records]

        all_fetch_tickers = list(dict.fromkeys(["SPY"] + watchlist_tickers + existing_tickers))

        # Ingest fresh data
        data = await self.ingest.ingest(all_fetch_tickers, [])
        portfolio = data["portfolio"]
        quotes = data["quotes"]
        historicals = data["historicals"]

        # ---- SECTION 1: reconcile live MCP portfolio ⟷ positions.json ----
        self.position_manager.reconcile_with_mcp(portfolio)
        open_records = self.position_manager.get_open_records()
        existing_tickers = [r.ticker for r in open_records]

        self.logger.info(
            "ingest_complete cycle=%s tickers=%d quotes_fetched=%d historicals_filled=%d",
            cycle_str, len(all_fetch_tickers), len(quotes),
            sum(1 for bars in historicals.values() if bars),
        )
        if cycle_type == CycleType.OPEN:
            try:
                await self.earnings_calendar.refresh(watchlist_tickers)
            except Exception:
                pass

        # Compute metrics
        metrics = {}
        metrics_tasks = {
            t: self.metrics_engine.compute(t, historicals.get(t, []), self.mcp_client)
            for t in all_fetch_tickers
        }
        results_list = await asyncio.gather(*metrics_tasks.values(), return_exceptions=True)
        for ticker, result in zip(metrics_tasks.keys(), results_list):
            if isinstance(result, Exception):
                self.logger.warning("metrics_compute_failed ticker=%s error=%s", ticker, result)
                metrics[ticker] = MarketMetrics(ticker=ticker)
            else:
                metrics[ticker] = result

        # fix stale current_price from metrics with live quote price
        for ticker, m in metrics.items():
            quote = quotes.get(ticker)
            if quote and quote.price > 0 and m.current_price > 0:
                sma_20 = m.current_price - (m.current_price * m.distance_from_20dma / 100.0)
                m.current_price = quote.price
                if sma_20 != 0:
                    m.distance_from_20dma = ((quote.price - sma_20) / sma_20) * 100.0

        spy_m = metrics.get("SPY")
        spy_context = ""
        if spy_m and spy_m.current_price > 0:
            spy_context = (
                f"MARKET CONTEXT (SPY): "
                f"5d={spy_m.return_5d:+.2f}% "
                f"30d={spy_m.return_30d:+.2f}% "
                f"vs_20dma={spy_m.distance_from_20dma:+.2f}% "
                f"rsi={spy_m.rsi_14:.1f} "
                f"vol_30d={spy_m.realized_vol_30d:.1f}%"
            )

        raw_news = await self.news_fetcher.fetch_news(
            all_fetch_tickers, cycle_str,
        )
        news: dict[str, list[str]] = {}
        for ticker, items in raw_news.items():
            if isinstance(items, list) and items and isinstance(items[0], NewsItem):
                news[ticker] = [
                    f"[{n.source}] {n.headline}"
                    + (f" — {n.summary}" if n.summary else "")
                    for n in items
                ]
            else:
                news[ticker] = items  # type: ignore[assignment]

        exit_signals = self.position_manager.check_exits(portfolio, quotes)
        effective_buying_power = float(portfolio.buying_power)
        # positions.json (via PositionManager) is the single source of truth
        # for the position count against the 12-slot cap — not the raw live
        # portfolio object.
        effective_positions_count = len(open_records)
        # {ticker: unrealized_pct} for the whole open book, computed ONCE
        # per cycle here and reused across every review_exit/make_decision
        # call below, instead of re-sending full detail on every position
        # into every single ticker's prompt.
        pnl_map = self.position_manager.open_positions_pnl_map(quotes)

        # Process exit signals (hard stops)
        for signal in exit_signals:
            ticker = signal.ticker
            quote = quotes.get(ticker)
            current_price = quote.price if quote else signal.current_price
            held_record = next((r for r in open_records if r.ticker == ticker), None)
            exit_decision = await self.decision_engine.review_exit(
                signal,
                metrics.get(ticker, MarketMetrics(ticker=ticker)),
                news.get(ticker, []),
                portfolio,
                cycle_type,
                spy_context=spy_context,
                position_record=held_record,
                open_positions_pnl=pnl_map,
                effective_buying_power=effective_buying_power,
            )
            try:
                mm = metrics.get(ticker)
                key_metrics = mm.model_dump(mode="json") if mm and hasattr(mm, "model_dump") else {}
                entry = JournalEntry(
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    cycle_type=cycle_str,
                    bull_thesis=exit_decision.bull_thesis,
                    bear_thesis=exit_decision.bear_thesis,
                    decision=exit_decision.decision.value,
                    decision_rationale=exit_decision.reasoning_summary,
                    key_metrics=key_metrics,
                    news_used=news.get(ticker, []),
                    fill_price=held_record.entry_price if held_record else None,
                    unrealized_pnl_pct=signal.unrealized_pct,
                    earnings_flag=self._earnings_flag(ticker),
                )
                self.journal_manager.append(ticker, entry)
            except Exception:
                pass
            if exit_decision.decision == DecisionType.SELL:
                result = await self.executor.execute(exit_decision, portfolio, current_price, dry_run=self.config["env"]["DRY_RUN"], vol_30d=metrics.get(ticker).realized_vol_30d if metrics.get(ticker) else 0.0, avg_volume_30d=metrics.get(ticker).avg_volume_30d if metrics.get(ticker) else 0.0, realized_vol_30d=metrics.get(ticker).realized_vol_30d if metrics.get(ticker) else 0.0)
                if result.status in ("EXECUTED", "SUBMITTED", "WOULD_EXECUTE"):
                    effective_buying_power += signal.quantity * current_price
                    effective_positions_count -= 1
                    pnl_map.pop(ticker, None)

        # ---- SECTION 2: Path A (held) + Path B (non-held) ----
        #
        # Path A: every existing position gets a routine re-evaluation.
        # Path B: new watchlist positions evaluated normally.

        # ---- Path A: held tickers (never skipped) ----
        for ticker in existing_tickers:
            record = next((r for r in open_records if r.ticker == ticker), None)
            if record is None:
                continue
            quote = quotes.get(ticker)
            current_price = quote.price if quote else 0.0
            if current_price <= 0 and record.entry_price > 0:
                current_price = record.entry_price

            unrealized_pct = ((current_price - record.entry_price)
                              / record.entry_price if record.entry_price else 0.0)
            try:
                days_held = (datetime.now(timezone.utc).date()
                             - datetime.fromisoformat(record.entry_date).date()).days
            except Exception:
                days_held = 0

            routine_signal = ExitSignal(
                ticker=ticker,
                reason="ROUTINE",
                current_price=current_price,
                unrealized_pct=unrealized_pct,
                quantity=record.quantity,
            )
            routine_decision = await self.decision_engine.review_exit(
                routine_signal,
                metrics.get(ticker, MarketMetrics(ticker=ticker)),
                news.get(ticker, []),
                portfolio,
                cycle_type,
                spy_context=spy_context,
                position_record=record,
                open_positions_pnl=pnl_map,
                effective_buying_power=effective_buying_power,
            )

            # Journal the held review (numbers consistent with positions.json)
            try:
                mm = metrics.get(ticker)
                key_metrics = mm.model_dump(mode="json") if mm and hasattr(mm, "model_dump") else {}
                entry = JournalEntry(
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    cycle_type=cycle_str,
                    bull_thesis=routine_decision.bull_thesis,
                    bear_thesis=routine_decision.bear_thesis,
                    decision=routine_decision.decision.value,
                    decision_rationale=routine_decision.reasoning_summary,
                    key_metrics=key_metrics,
                    news_used=news.get(ticker, []),
                    fill_price=record.entry_price,
                    unrealized_pnl_pct=unrealized_pct,
                    earnings_flag=self._earnings_flag(ticker),
                )
                self.journal_manager.append(ticker, entry)
            except Exception:
                pass

            if routine_decision.decision == DecisionType.SELL:
                result = await self.executor.execute(routine_decision, portfolio, current_price, dry_run=self.config["env"]["DRY_RUN"], vol_30d=metrics.get(ticker).realized_vol_30d if metrics.get(ticker) else 0.0, avg_volume_30d=metrics.get(ticker).avg_volume_30d if metrics.get(ticker) else 0.0, realized_vol_30d=metrics.get(ticker).realized_vol_30d if metrics.get(ticker) else 0.0)
                if result.status in ("EXECUTED", "SUBMITTED", "WOULD_EXECUTE"):
                    effective_buying_power += record.quantity * current_price
                    effective_positions_count -= 1
                    pnl_map.pop(ticker, None)
                    # record_exit is already performed inside ExecutionEngine.execute on SELL fill.
                    # Do not call record_exit again here.

        # ---- Path B: non-held watchlist tickers (unchanged) ----
        for ticker in [t for t in watchlist_tickers if t != "SPY"]:
            if ticker in existing_tickers:
                continue
            quote = quotes.get(ticker)
            current_price = quote.price if quote else 0.0
            decision = await self.decision_engine.make_decision(
                ticker,
                metrics.get(ticker, MarketMetrics(ticker=ticker)),
                news.get(ticker, []),
                portfolio,
                current_price,
                effective_buying_power,
                effective_positions_count,
                cycle_type,
                spy_context=spy_context,
                open_positions_pnl=pnl_map,
            )
            mm = metrics.get(ticker)
            result = await self.executor.execute(
                decision,
                portfolio,
                current_price,
                dry_run=self.config["env"]["DRY_RUN"],
                vol_30d=mm.realized_vol_30d if mm else 0.0,
                avg_volume_30d=mm.avg_volume_30d if mm else 0.0,
                realized_vol_30d=mm.realized_vol_30d if mm else 0.0,
            )
            # Journal AFTER execute so fill_price can come from the fill
            # (matching positions.json). For IGNORE/HOLD fill_price is null.
            try:
                key_metrics = mm.model_dump(mode="json") if mm and hasattr(mm, "model_dump") else {}
                fill_price_for_journal = None
                if result.status in ("EXECUTED", "SUBMITTED", "WOULD_EXECUTE") and result.fill_price:
                    fill_price_for_journal = float(result.fill_price)
                entry = JournalEntry(
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    cycle_type=cycle_str,
                    bull_thesis=decision.bull_thesis,
                    bear_thesis=decision.bear_thesis,
                    decision=decision.decision.value,
                    decision_rationale=decision.reasoning_summary,
                    key_metrics=key_metrics,
                    news_used=news.get(ticker, []),
                    fill_price=fill_price_for_journal,
                    unrealized_pnl_pct=None,
                    earnings_flag=self._earnings_flag(ticker),
                )
                self.journal_manager.append(ticker, entry)
            except Exception:
                pass
            if result.status in ("EXECUTED", "WOULD_EXECUTE"):
                if decision.decision in (DecisionType.BUY, DecisionType.ROTATE):
                    size_pct = ExecutionEngine._size_from_edge(decision.edge.total)
                    effective_buying_power -= size_pct * portfolio.total_value
                    effective_positions_count += 1
                    # newly opened this cycle — 0% unrealized, so the very
                    # next Path B ticker's roster line already knows about it
                    pnl_map[ticker] = 0.0

        # ---- End of cycle: write open-position unrealized marks (no extra API) ----
        try:
            self.position_manager.write_performance_snapshot(quotes)
        except Exception as exc:
            self.logger.warning("end_of_cycle_performance_snapshot_failed: %s", exc)

    async def _run_attribution_report(self) -> None:
        report_path = Path(__file__).resolve().parent / "logs" / f"attribution_report_{datetime.now(timezone.utc).date().isoformat()}.json"
        if report_path.exists():
            return

        trades_path = Path(__file__).resolve().parent / "logs" / "trades.json"
        trades = []
        if trades_path.exists():
            try:
                trades = json.loads(trades_path.read_text(encoding="utf-8"))
            except Exception:
                trades = []

        threshold = float(self.config.get("trading", {}).get("edge_score_threshold", 3.0))
        th_u = 0.5
        edge_distribution = {
            f"<{threshold:g}": 0,
            f"{threshold:g}-{threshold+th_u:g}": 0,
            f"{threshold+th_u:g}-{threshold+2*th_u:g}": 0,
            f">={threshold+2*th_u:g}": 0,
        }
        for record in trades:
            edge = float(record.get("edge_score", 0.0) or 0.0)
            if edge < threshold:
                edge_distribution[f"<{threshold:g}"] += 1
            elif edge < threshold + th_u:
                edge_distribution[f"{threshold:g}-{threshold+th_u:g}"] += 1
            elif edge < threshold + 2 * th_u:
                edge_distribution[f"{threshold+th_u:g}-{threshold+2*th_u:g}"] += 1
            else:
                edge_distribution[f">={threshold+2*th_u:g}"] += 1

        real_pnl = {}
        try:
            real_pnl = await self.pnl_tracker.build_report()
        except Exception:
            pass

        payload = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "total_trades": len(trades),
            "would_execute_count": sum(1 for item in trades if str(item.get("status", "")).upper() == "WOULD_EXECUTE"),
            "edge_score_distribution": edge_distribution,
            "real_robinhood_pnl": real_pnl,
        }
        report_path.parent.mkdir(exist_ok=True)
        report_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    async def start(self) -> None:
        configure_logging()
        self.logger.info("agent_start dry_run=%s", self.config["env"]["DRY_RUN"])

        schedule = self.config.get("schedule", {})
        open_h = int(schedule.get("open_hour", 9))
        open_m = int(schedule.get("open_minute", 35))
        mid_h = int(schedule.get("mid_hour", 12))
        mid_m = int(schedule.get("mid_minute", 30))
        close_h = int(schedule.get("close_hour", 15))
        close_m = int(schedule.get("close_minute", 30))
        tz_name = schedule.get("timezone", "America/New_York")

        self.scheduler.add_job(
            self.run_cycle, "cron", args=[CycleType.OPEN],
            day_of_week="mon-fri", hour=open_h, minute=open_m, timezone=tz_name,
        )
        self.scheduler.add_job(
            self.run_cycle, "cron", args=[CycleType.MID],
            day_of_week="mon-fri", hour=mid_h, minute=mid_m, timezone=tz_name,
        )
        self.scheduler.add_job(
            self.run_cycle, "cron", args=[CycleType.CLOSE],
            day_of_week="mon-fri", hour=close_h, minute=close_m, timezone=tz_name,
        )

        # --- 1-minute price monitor (section 7) ---
        self.scheduler.add_job(
            self.run_price_monitor, "cron",
            day_of_week="mon-fri", hour="9-15", minute="*",
            timezone=tz_name,
            max_instances=1, coalesce=True,
        )
        self.scheduler.add_job(
            self.run_price_monitor, "cron",
            day_of_week="mon-fri", hour=16, minute="0",
            timezone=tz_name,
            max_instances=1, coalesce=True,
        )
        self.scheduler.start()

        # --- Reconcile at agent start ---
        try:
            portfolio = await self.mcp_client.get_portfolio()
            # Never reconcile against an empty / failed portfolio pull
            if portfolio is None or not getattr(portfolio, "positions", None):
                self.logger.warning(
                    "agent_start_reconcile_skipped: empty or missing portfolio.positions"
                )
            else:
                self.position_manager.reconcile_with_mcp(portfolio)
        except Exception as exc:
            self.logger.warning("agent_start_reconcile_failed: %s", exc)

        # --- Smart startup catch-up ---
        tz = pytz.timezone(tz_name)
        now_et = datetime.now(tz)
        current_time = now_et.time()
        weekday = now_et.weekday()

        open_time = time(open_h, open_m)
        mid_time = time(mid_h, mid_m)
        close_time = time(close_h, close_m)
        after_hours = time(16, 0)

        if weekday >= 5:
            self.logger.info("outside market hours, waiting for next trading day (ET=%s weekday=%d)",
                             current_time.strftime("%H:%M"), weekday)
        elif current_time >= after_hours:
            self.logger.info("outside_after_hour waiting for next trading day (ET=%s)", current_time.strftime("%H:%M"))
        elif current_time < open_time:
            self.logger.info("waiting_for_open open_at=%s (ET=%s)", open_time.strftime("%H:%M"), current_time.strftime("%H:%M"))
        elif current_time < mid_time:
            self.logger.info("catch_up: running OPEN now (ET=%s)", current_time.strftime("%H:%M"))
            await self.run_cycle(CycleType.OPEN)
        elif current_time < close_time:
            self.logger.info("catch_up: running MID now (ET=%s)", current_time.strftime("%H:%M"))
            await self.run_cycle(CycleType.MID)
        else:
            self.logger.info("catch_up: running CLOSE now (ET=%s)", current_time.strftime("%H:%M"))
            await self.run_cycle(CycleType.CLOSE)

        try:
            while True:
                await asyncio.sleep(60)
        except (KeyboardInterrupt, SystemExit):
            self.logger.info("agent_shutdown")
            self.scheduler.shutdown(wait=False)


if __name__ == "__main__":
    agent = TradingAgent()
    asyncio.run(agent.start())