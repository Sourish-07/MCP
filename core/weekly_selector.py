"""Weekly ticker selection orchestrator."""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import pytz
from apscheduler.schedulers.asyncio import AsyncIOScheduler

from core.sentiment_model import get_sentiment_model
from core.weekly_news import WeeklyNewsFetcher
from core.ticker_selector import select_universe
from core.position_manager import PositionManager

logger = logging.getLogger("robinhood-agent.core.weekly_selector")


class WeeklySelector:
    """Orchestrates weekly ticker selection based on sentiment analysis."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.logger = logger
        self.enabled = config.get("sentiment_selection", {}).get("enabled", False)

        ss = config.get("sentiment_selection", {})
        self.universe_size = int(ss.get("universe_size", 10))
        self.max_positions = int(config.get("trading", {}).get("max_positions", 12))
        self.lookback_days = int(ss.get("lookback_days", 7))
        self.min_sentiment_score = float(ss.get("min_sentiment_score", -1.0))

        self.data_dir = Path(__file__).resolve().parent.parent / "data"
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.selected_universe_path = self.data_dir / "selected_universe.json"
        self.candidate_universe_path = self.data_dir / "candidate_universe.json"

        self.position_manager = PositionManager()
        self.news_fetcher = WeeklyNewsFetcher(lookback_days=self.lookback_days)
        self.sentiment_model = None  # Lazy load

        self.candidate_universe: list[str] = []
        self.selected_universe: list[str] = []

    async def initialize(self) -> None:
        """Initialize components and load data."""
        self.logger.info("weekly_selector_initializing")
        self.candidate_universe = await self._load_candidate_universe()
        self.selected_universe = await self._load_selected_universe()
        # Lazy-load the sentiment model once
        self.sentiment_model = get_sentiment_model()
        self.logger.info(
            "weekly_selector_initialized candidates=%d previous_selection=%d",
            len(self.candidate_universe), len(self.selected_universe),
        )

    async def _load_candidate_universe(self) -> list[str]:
        """Load the candidate universe from file."""
        try:
            with open(self.candidate_universe_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                tickers = data.get("tickers", [])
                self.logger.info("Loaded %d candidate tickers", len(tickers))
                return tickers
        except FileNotFoundError:
            self.logger.warning("Candidate universe file not found, using defaults")
            return self._get_default_universe()
        except Exception as exc:
            self.logger.error("Failed to load candidate universe: %s", exc)
            return self._get_default_universe()

    def _get_default_universe(self) -> list[str]:
        """Default liquid universe if the candidate file is missing."""
        return [
            "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "NVDA",
            "JPM", "V", "JNJ", "WMT", "PG", "MA", "HD", "DIS",
            "NFLX", "ADBE", "INTC", "AMD", "CAT", "BA", "XOM", "COST",
            "UNH", "AVGO", "MU", "TSM", "GS", "MS", "SBUX", "MCD",
            "NKE", "CRM", "PYPL", "QCOM", "TXN", "HON", "IBM", "ORCL",
        ]

    async def _load_selected_universe(self) -> list[str]:
        """Load previously selected universe."""
        try:
            with open(self.selected_universe_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                tickers = data.get("tickers", [])
                self.logger.info("Loaded previous selection from %s", data.get("timestamp", ""))
                return tickers
        except FileNotFoundError:
            self.logger.info("No previous selection found")
            return []
        except Exception as exc:
            self.logger.error("Failed to load selected universe: %s", exc)
            return []

    async def _save_selected_universe(self, tickers: list[str], scores: dict[str, float]) -> None:
        """Save selected universe to file."""
        try:
            data = {
                "tickers": tickers,
                "scores": scores,
                "timestamp": datetime.now(pytz.UTC).isoformat(),
                "universe_size": self.universe_size,
            }
            with open(self.selected_universe_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            self.logger.info("Saved selection of %d tickers", len(tickers))
        except Exception as exc:
            self.logger.error("Failed to save selected universe: %s", exc)

    def get_current_holdings(self) -> set[str]:
        """Get currently held tickers from position manager."""
        try:
            open_records = self.position_manager.get_open_records()
            held_tickers = {record.ticker for record in open_records if record.quantity > 0}
            self.logger.info("current_holdings count=%d tickers=%s", len(held_tickers), sorted(held_tickers))
            return held_tickers
        except Exception as exc:
            self.logger.error("Failed to get current holdings: %s", exc)
            return set()

    async def fetch_news_and_scores(self, tickers: list[str]) -> dict[str, float]:
        """Fetch news and score tickers.

        Per-ticker aggregation (required contract):
          1. Collect ALL headlines for the ticker from the lookback window.
          2. Score EVERY headline individually with the ModernBERT model.
          3. Aggregate to a single ticker score using the MEAN of the
             individual headline scores (not first, not max).
          4. A ticker with zero headlines gets a neutral score of 0.0.
          5. A scoring failure for a ticker falls back to 0.0 and is logged;
             it never aborts the whole scoring pass.
        """
        self.logger.info("Fetching news for %d tickers...", len(tickers))
        news_dict = await self.news_fetcher.fetch_news_for_universe(tickers)

        self.logger.info("Scoring news...")
        scores: dict[str, float] = {}

        for ticker, news_items in news_dict.items():
            if not news_items:
                scores[ticker] = 0.0
                continue

            headlines = [item.headline for item in news_items if item.headline]
            if not headlines:
                scores[ticker] = 0.0
                continue

            try:
                scores_list = self.sentiment_model.score_headlines(headlines)
                # Ticker score = MEAN of per-headline sentiment scores
                scores[ticker] = sum(scores_list) / len(scores_list) if scores_list else 0.0
            except Exception as exc:
                self.logger.error("Failed to score %s: %s", ticker, exc)
                scores[ticker] = 0.0

        self.logger.info("Scored %d tickers", len(scores))
        return scores

    async def _fallback_universe(self) -> list[str]:
        """Previous week's universe, or the static default as a last resort.

        Never returns empty unless there is genuinely nothing configured --
        the bot is never left with an empty/missing universe after a failure.
        """
        if self.selected_universe:
            self.logger.info("fallback=previous_selection size=%d", len(self.selected_universe))
            return list(self.selected_universe)
        # Try reloading the persisted universe from disk (in-memory state may
        # be empty on a fresh process even though a previous selection exists).
        try:
            if self.selected_universe_path.exists():
                with open(self.selected_universe_path, "r", encoding="utf-8") as f:
                    tickers = json.load(f).get("tickers", [])
                if tickers:
                    self.logger.info("fallback=persisted_file size=%d", len(tickers))
                    self.selected_universe = list(tickers)
                    return list(tickers)
        except Exception as exc:
            self.logger.error("fallback_reload_failed: %s", exc)
        default = self._get_default_universe()[: self.universe_size]
        self.logger.warning("fallback=static_default_universe size=%d", len(default))
        return default

    async def run_selection(self) -> list[str]:
        """Run the weekly selection pipeline. Returns the selected tickers.

        The ENTIRE pipeline (holdings lookup -> news fetch -> scoring ->
        selection -> persistence) is wrapped in a broad try/except. On ANY
        failure -- Finnhub error, model error, ValueError, network issue,
        empty candidates, etc. -- we log a clear ERROR and keep the previous
        week's universe untouched (or the static default if none exists).
        The bot is never left with an empty or missing universe.
        """
        self.logger.info("weekly_selection_starting")

        try:
            current_holdings = self.get_current_holdings()
            self.logger.info("protected_tickers=%s", sorted(current_holdings))

            all_candidates = sorted(set(self.candidate_universe) | set(self.selected_universe))
            self.logger.info("scoring %d candidates", len(all_candidates))

            scores = await self.fetch_news_and_scores(all_candidates)

            selected = select_universe(
                held_tickers=current_holdings,
                scored_candidates=scores,
                universe_size=self.universe_size,
                max_positions=self.max_positions,
                min_sentiment_score=self.min_sentiment_score,
            )

            if not selected:
                raise ValueError("selection produced an empty universe")

            self.logger.info("selection_complete selected=%s", selected)
            await self._save_selected_universe(selected, scores)
            self.selected_universe = selected
            return selected
        except Exception as exc:
            self.logger.error(
                "weekly_selection_failed, keeping previous universe: %s: %s",
                type(exc).__name__, exc,
            )
            return await self._fallback_universe()

    def start_scheduler(self, scheduler: AsyncIOScheduler) -> None:
        """Register the weekly selection job on the shared scheduler.

        Note: uses the main agent's scheduler rather than creating its own,
        so all jobs live in one place and shut down together.
        """
        if not self.enabled:
            self.logger.info("weekly_selector_disabled")
            return

        schedule_config = self.config.get("sentiment_selection", {})
        day_of_week = schedule_config.get("schedule_day", "sun")
        hour = int(schedule_config.get("schedule_hour", 20))
        minute = int(schedule_config.get("schedule_minute", 0))
        timezone_str = schedule_config.get("schedule_timezone", "America/New_York")

        try:
            scheduler.add_job(
                self.run_selection,
                "cron",
                day_of_week=day_of_week,
                hour=hour,
                minute=minute,
                timezone=timezone_str,
                id="weekly_sentiment_selection",
                max_instances=1,
                coalesce=True,
            )
            self.logger.info(
                "weekly_scheduler_registered day=%s hour=%d minute=%d tz=%s",
                day_of_week, hour, minute, timezone_str,
            )
        except Exception as exc:
            self.logger.error("weekly_scheduler_registration_failed: %s", exc)

    async def run_now(self) -> list[str]:
        """Manually trigger selection (for testing / startup catch-up)."""
        return await self.run_selection()

    def get_selected_universe(self) -> list[str]:
        """Get the currently selected universe (falls back to config watchlist)."""
        if self.selected_universe:
            return list(self.selected_universe)
        fallback = self.config.get("watchlist", {}).get("default_tickers", [])
        return list(fallback)

    def get_candidate_universe(self) -> list[str]:
        """Get the candidate universe."""
        return list(self.candidate_universe)


# Module-level singleton
_weekly_selector: WeeklySelector | None = None


async def get_weekly_selector(config: dict[str, Any]) -> WeeklySelector:
    """Get or create the weekly selector singleton."""
    global _weekly_selector
    if _weekly_selector is None:
        _weekly_selector = WeeklySelector(config)
        await _weekly_selector.initialize()
    return _weekly_selector


async def run_weekly_selection(config: dict[str, Any]) -> list[str]:
    """Convenience function to run weekly selection."""
    selector = await get_weekly_selector(config)
    return await selector.run_selection()