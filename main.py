from __future__ import annotations

import asyncio
import json
import logging
import os
from datetime import datetime, timezone, time
from pathlib import Path

import pytz
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from dotenv import load_dotenv

from core.data_ingest import DataIngestLayer
from core.decision_engine import DecisionEngine
from core.execution import ExecutionEngine
from core.metrics import MetricsEngine
from core.news_fetch import NewsFetcher
from core.pnl_tracker import PnLTracker
from core.position_manager import PositionManager
from robinhood_mcp.robinhood_client import RobinhoodMCPClient
from models.decisions import DecisionType, CycleType, JournalEntry
from models.market_data import MarketMetrics
from utils.journal_manager import JournalManager
from utils.earnings_calendar import EarningsCalendar
from utils.anthropic_client import AnthropicClient
from utils.cost_tracker import CostLimitExceededError, CostTracker
from utils.logger import configure_logging


class TradingAgent:
    """Anthropic-only trading agent orchestrating ingest, metrics, news, decisions, and execution."""

    def __init__(self) -> None:
        self.logger = logging.getLogger("robinhood-agent.main")
        self.scheduler = AsyncIOScheduler(timezone="America/New_York")
        self.config = self._load_config()
        self.cost_tracker = CostTracker()
        self.mcp_client = RobinhoodMCPClient()
        self.ingest = DataIngestLayer(self.mcp_client)
        self.metrics_engine = MetricsEngine()
        self.news_fetcher = NewsFetcher(AnthropicClient(model="claude-haiku-4-5"))
        # initialize shared managers and pass into decision engine
        self.journal_manager = JournalManager()
        self.earnings_calendar = EarningsCalendar()
        self.earnings_calendar.set_client(self.mcp_client)
        self.decision_engine = DecisionEngine(AnthropicClient(model="claude-sonnet-5"), self.journal_manager, self.earnings_calendar)
        self.position_manager = PositionManager()
        self.pnl_tracker = PnLTracker(self.mcp_client)
        self.executor = ExecutionEngine(self.mcp_client, self.position_manager, self.config)
        self._cached_watchlist: list[str] = []
        self._cached_watchlist_date: str = ""

    def _load_config(self) -> dict:
        """Load settings and environment variables for the trading cycle."""
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
        """Load watchlist from Robinhood MCP, fall back to settings.json."""
        watchlist_name = self.config.get("watchlist", {}).get(
            "robinhood_watchlist_name", "Default")
        live_tickers = await self.mcp_client.get_watchlist_tickers(watchlist_name)
        if live_tickers:
            tickers = live_tickers[:20]
            self.logger.info("watchlist_source=robinhood_mcp count=%d name=%s",
                             len(tickers), watchlist_name)
            return tickers
        fallback = list(self.config["watchlist"]["default_tickers"])
        self.logger.info(
            "watchlist_source=settings_json count=%d (MCP returned empty)",
            len(fallback))
        return fallback

    async def run_cycle(self, cycle_type: CycleType) -> None:
        """Run one full trading cycle for the specified `cycle_type` (OPEN/MID/CLOSE)."""
        self.logger.info("=== cycle_start cycle=%s dry_run=%s ===", cycle_type.value if isinstance(cycle_type, CycleType) else str(cycle_type), self.config["env"]["DRY_RUN"])

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
        existing_tickers = [record.ticker for record in open_records]

        all_fetch_tickers = list(dict.fromkeys(
            ["SPY"] + watchlist_tickers + existing_tickers))

        # Every cycle always fetches fresh portfolio, quotes, and historicals.
        # No in-memory caching between cycles — data changes throughout the
        # trading day and must be refetched every time, including OPEN, MID,
        # and CLOSE alike, regardless of whether this is a fresh process or
        # a continuation of one already running.
        data = await self.ingest.ingest(all_fetch_tickers, [])
        portfolio = data["portfolio"]
        quotes = data["quotes"]
        historicals = data["historicals"]
        self.logger.info(
            "ingest_complete cycle=%s tickers=%d quotes_fetched=%d historicals_filled=%d",
            cycle_type.value if isinstance(cycle_type, CycleType) else str(cycle_type),
            len(all_fetch_tickers), len(quotes),
            sum(1 for bars in historicals.values() if bars),
        )
        if cycle_type == CycleType.OPEN:
            try:
                await self.earnings_calendar.refresh(watchlist_tickers)
            except Exception:
                pass

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
        spy_m = metrics.get("SPY")
        if spy_m and spy_m.current_price > 0:
            spy_context = (
                f"MARKET CONTEXT (SPY): "
                f"5d={spy_m.return_5d:+.2f}% "
                f"30d={spy_m.return_30d:+.2f}% "
                f"vs_20dma={spy_m.distance_from_20dma:+.2f}% "
                f"rsi={spy_m.rsi_14:.1f} "
                f"vol_30d={spy_m.realized_vol_30d:.1f}%"
            )
        else:
            spy_context = ""

        news = await self.news_fetcher.fetch_news(all_fetch_tickers, cycle_type.value if isinstance(cycle_type, CycleType) else str(cycle_type))

        exit_signals = self.position_manager.check_exits(portfolio, quotes)
        effective_buying_power = float(portfolio.buying_power)
        effective_positions_count = len(portfolio.positions)

        for signal in exit_signals:
            ticker = signal.ticker
            quote = quotes.get(ticker)
            current_price = quote.price if quote else signal.current_price
            exit_decision = await self.decision_engine.review_exit(
                signal,
                metrics.get(ticker, await self.metrics_engine.compute(ticker, [], self.mcp_client)),
                news.get(ticker, []),
                portfolio,
                cycle_type,
                spy_context=spy_context,
            )
            # append journal entry for the review decision
            try:
                mm = metrics.get(ticker)
                key_metrics = mm.model_dump(mode="json") if mm and hasattr(mm, "model_dump") else {}
                entry = JournalEntry(
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    cycle_type=cycle_type.value if isinstance(cycle_type, CycleType) else str(cycle_type),
                    bull_thesis=exit_decision.bull_thesis,
                    bear_thesis=exit_decision.bear_thesis,
                    decision=exit_decision.decision.value,
                    decision_rationale=exit_decision.reasoning_summary,
                    key_metrics=key_metrics,
                    news_used=news.get(ticker, []),
                )
                self.journal_manager.append(ticker, entry)
            except Exception:
                pass
            if exit_decision.decision == DecisionType.SELL:
                result = await self.executor.execute(exit_decision, portfolio, current_price, dry_run=self.config["env"]["DRY_RUN"], vol_30d=metrics.get(ticker).realized_vol_30d if metrics.get(ticker) else 0.0, avg_volume_30d=metrics.get(ticker).avg_volume_30d if metrics.get(ticker) else 0.0, realized_vol_30d=metrics.get(ticker).realized_vol_30d if metrics.get(ticker) else 0.0)
                if result.status in ("EXECUTED", "WOULD_EXECUTE"):
                    effective_buying_power += signal.quantity * current_price
                    effective_positions_count -= 1

        for ticker in [t for t in watchlist_tickers if t != "SPY"]:
            if ticker in existing_tickers:
                continue
            quote = quotes.get(ticker)
            current_price = quote.price if quote else 0.0
            decision = await self.decision_engine.make_decision(
                ticker,
                metrics.get(ticker, await self.metrics_engine.compute(ticker, [], self.mcp_client)),
                news.get(ticker, []),
                portfolio,
                current_price,
                effective_buying_power,
                effective_positions_count,
                cycle_type,
                spy_context=spy_context,
            )
            # append journal entry for the new-position decision
            try:
                mm = metrics.get(ticker)
                key_metrics = mm.model_dump(mode="json") if mm and hasattr(mm, "model_dump") else {}
                entry = JournalEntry(
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    cycle_type=cycle_type.value if isinstance(cycle_type, CycleType) else str(cycle_type),
                    bull_thesis=decision.bull_thesis,
                    bear_thesis=decision.bear_thesis,
                    decision=decision.decision.value,
                    decision_rationale=decision.reasoning_summary,
                    key_metrics=key_metrics,
                    news_used=news.get(ticker, []),
                    earnings_flag=bool(self.earnings_calendar.get_upcoming(ticker, within_days=2)),
                )
                self.journal_manager.append(ticker, entry)
            except Exception:
                pass
            result = await self.executor.execute(decision, portfolio, current_price, dry_run=self.config["env"]["DRY_RUN"], vol_30d=metrics.get(ticker).realized_vol_30d if metrics.get(ticker) else 0.0, avg_volume_30d=metrics.get(ticker).avg_volume_30d if metrics.get(ticker) else 0.0, realized_vol_30d=metrics.get(ticker).realized_vol_30d if metrics.get(ticker) else 0.0)
            if result.status in ("EXECUTED", "WOULD_EXECUTE"):
                if decision.decision in (DecisionType.BUY, DecisionType.ROTATE):
                    # compute the size used based on the decision's computed edge; execution resolved exact size
                    size_pct = ExecutionEngine._size_from_edge(decision.edge.total)
                    effective_buying_power -= size_pct * portfolio.total_value
                    effective_positions_count += 1

        if datetime.now(pytz.timezone(self.config["schedule"]["timezone"])).weekday() == 4:
            await self._run_attribution_report()

    async def _run_attribution_report(self) -> None:
        """Write a minimal attribution summary based on the trade log for the current day."""
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

        edge_distribution = {"<3.5": 0, "3.5-4.0": 0, "4.0-4.5": 0, ">=4.5": 0}
        for record in trades:
            edge = float(record.get("edge_score", 0.0) or 0.0)
            if edge < 3.5:
                edge_distribution["<3.5"] += 1
            elif edge < 4.0:
                edge_distribution["3.5-4.0"] += 1
            elif edge < 4.5:
                edge_distribution["4.0-4.5"] += 1
            else:
                edge_distribution[">=4.5"] += 1

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

    def _is_market_open(self) -> bool:
        """Return True if the current time falls within NYSE market hours."""
        ny_tz = pytz.timezone(self.config["schedule"]["timezone"])
        now = datetime.now(ny_tz)
        if now.weekday() >= 5:
            return False
        open_time = now.replace(hour=9, minute=30, second=0, microsecond=0)
        close_time = now.replace(hour=16, minute=0, second=0, microsecond=0)
        return open_time <= now <= close_time

    async def start(self) -> None:
        """Register cron jobs and begin the run loop with smart startup logic."""
        configure_logging()
        self.logger.info("agent_start dry_run=%s", self.config["env"]["DRY_RUN"])

        # --- Register three cron jobs exactly as configured (mon-fri, America/New_York) ---
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
        self.scheduler.start()

        # --- Smart startup catch-up logic (all boundaries use datetime.time objects) ---
        tz = pytz.timezone(tz_name)
        now_et = datetime.now(tz)
        current_time = now_et.time()
        weekday = now_et.weekday()  # 0=Mon … 4=Fri, 5=Sat, 6=Sun

        # cron fire times as datetime.time boundaries
        open_time = time(open_h, open_m)        # e.g. 09:35
        mid_time = time(mid_h, mid_m)            # e.g. 12:30
        close_time = time(close_h, close_m)      # e.g. 15:30
        after_hours = time(16, 0)                # 16:00

        if weekday >= 5:
            self.logger.info(
                "outside market hours, waiting for next trading day "
                "(current ET: %s, weekday=%d)",
                current_time.strftime("%H:%M"), weekday,
            )
        elif current_time >= after_hours:
            self.logger.info(
                "outside market hours, waiting for next trading day "
                "(current ET: %s, after 16:00)",
                current_time.strftime("%H:%M"),
            )
        elif current_time < open_time:
            self.logger.info(
                "waiting for market open at %s (current ET: %s) — "
                "no catch-up needed, OPEN cron will fire normally",
                open_time.strftime("%H:%M"), current_time.strftime("%H:%M"),
            )
        elif current_time < mid_time:
            self.logger.info(
                "catch-up: current ET %s is between %s and %s — "
                "running OPEN now, MID and CLOSE will fire via cron",
                current_time.strftime("%H:%M"),
                open_time.strftime("%H:%M"), mid_time.strftime("%H:%M"),
            )
            await self.run_cycle(CycleType.OPEN)
        elif current_time < close_time:
            self.logger.info(
                "catch-up: current ET %s is between %s and %s — "
                "running MID now, CLOSE will fire via cron",
                current_time.strftime("%H:%M"),
                mid_time.strftime("%H:%M"), close_time.strftime("%H:%M"),
            )
            await self.run_cycle(CycleType.MID)
        else:
            # between close_time and after_hours
            self.logger.info(
                "catch-up: current ET %s is between %s and %s — "
                "running CLOSE now",
                current_time.strftime("%H:%M"),
                close_time.strftime("%H:%M"), after_hours.strftime("%H:%M"),
            )
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
