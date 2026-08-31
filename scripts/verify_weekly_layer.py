"""Manual verification script for the weekly sentiment selection layer.

Run from the repo root: python scripts/verify_weekly_layer.py

Checks performed (offline-deterministic; news + model are stubbed):
  1. Manually trigger the weekly selector and inspect the resulting list
  2. Simulate 3 open positions -> those 3 stay protected, only 7 new chosen
  3. Simulate selling one to zero -> it is eligible for replacement next run
  4. Force Finnhub/model failure -> previous universe kept, error logged
  5. H > universe_size -> ALL held tickers kept, universe temporarily larger
  6. Feature flag off -> no scheduler job, universe falls back to watchlist
  7. Live Finnhub fetch (only if FINNHUB_API_KEY is set; otherwise skipped)
  8. Confirm daily schedule and max_positions are untouched
  9. Confirm no hard-coded secrets in the new modules
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
os.chdir(REPO_ROOT)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
log = logging.getLogger("verify")


async def main() -> int:
    # --- 0. Load config ---
    cfg_path = REPO_ROOT / "config" / "settings.json"
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    log.info("config loaded: sentiment_selection.enabled=%s universe_size=%s max_positions=%s",
             cfg["sentiment_selection"]["enabled"],
             cfg["sentiment_selection"]["universe_size"],
             cfg["trading"]["max_positions"])

    # --- 1. Trigger weekly selector manually (with stubbed news+model so no API calls) ---
    from core.weekly_selector import WeeklySelector
    selector = WeeklySelector(cfg)
    log.info("selector created: enabled=%s universe_size=%d", selector.enabled, selector.universe_size)

    # Stub news fetcher and scoring so verification is deterministic offline.
    # SIM_* tickers get very negative scores so a sold position is never
    # re-selected as a fresh candidate.
    async def fake_scores(tickers):
        return {t: (-0.9 if t.startswith("SIM") else 0.5 - i * 0.01)
                for i, t in enumerate(tickers)}
    selector.fetch_news_and_scores = fake_scores

    await selector.initialize()
    selected = await selector.run_now()
    log.info("1) manual weekly selection -> %s", selected)
    assert len(selected) == cfg["sentiment_selection"]["universe_size"], "universe size mismatch"

    # --- 2. Simulate 3 open positions ---
    from models.portfolio import PositionRecord

    def holdings(*tickers: str):
        class FakePM:
            def get_open_records(self):
                return [PositionRecord(ticker=t, entry_price=100.0, quantity=1.0)
                        for t in tickers]
        return FakePM()

    selector.position_manager = holdings("SIM_A", "SIM_B", "SIM_C")
    selected3 = await selector.run_selection()
    for t in ("SIM_A", "SIM_B", "SIM_C"):
        assert t in selected3, f"{t} must be protected"
    assert len(selected3) == cfg["sentiment_selection"]["universe_size"]
    log.info("2) 3 holdings protected + 7 new -> %s", selected3)

    # --- 3. Sell one to zero -> eligible for replacement next run ---
    selector.position_manager = holdings("SIM_A", "SIM_B")
    selected2 = await selector.run_selection()
    assert "SIM_C" not in selected2, "sold position should be replaceable"
    log.info("3) after selling SIM_C -> %s", selected2)

    # --- 4. Force failure -> previous universe kept ---
    prev = selector.selected_universe

    async def boom(tickers):
        raise TimeoutError("forced failure")
    selector.fetch_news_and_scores = boom
    kept = await selector.run_selection()
    assert kept == prev or kept == selected2, "failure must keep previous universe"
    assert kept, "bot must never be left with an empty universe"
    log.info("4) failure kept previous universe -> %s", kept)

    # --- 5. H > universe_size -> ALL held tickers kept, none dropped ---
    selector.fetch_news_and_scores = fake_scores
    many = tuple(f"SIM_H{i}" for i in range(12))  # 12 held > universe_size 10
    selector.position_manager = holdings(*many)
    selected_big = await selector.run_selection()
    assert len(selected_big) == 12, "universe may temporarily exceed universe_size"
    assert all(t in selected_big for t in many), "no holding may ever be dropped"
    log.info("5) H=12 > universe_size=10 -> all 12 kept -> %s", selected_big)

    # --- 6. Feature flag off -> no scheduler job, watchlist fallback ---
    import copy
    from unittest.mock import MagicMock

    cfg_off = copy.deepcopy(cfg)
    cfg_off["sentiment_selection"]["enabled"] = False
    selector_off = WeeklySelector(cfg_off)
    mock_scheduler = MagicMock()
    selector_off.start_scheduler(mock_scheduler)
    mock_scheduler.add_job.assert_not_called()
    assert selector_off.get_selected_universe() == cfg["watchlist"]["default_tickers"], \
        "flag off must fall back to watchlist.default_tickers"
    log.info("6) feature flag off -> scheduler untouched, watchlist fallback OK")

    # --- 7. Live Finnhub fetch (only when key present) ---
    if os.getenv("FINNHUB_API_KEY"):
        from core.weekly_news import WeeklyNewsFetcher
        fetcher = WeeklyNewsFetcher(lookback_days=7)
        live = await fetcher.fetch_news_for_universe(["AAPL", "MSFT"])
        for sym, items in live.items():
            log.info("7) live Finnhub %s -> %d headlines", sym, len(items))
            for item in items[:2]:
                score = selector.sentiment_model.score_single(item.headline) \
                    if selector.sentiment_model else 0.0
                log.info("   score=%+.3f %s", score, item.headline[:80])
        assert any(live.values()), "live fetch returned no headlines at all"
    else:
        log.info("7) FINNHUB_API_KEY not set -> live fetch SKIPPED, mocked/e2e path verified")

    # --- 8. Daily schedule + max positions untouched ---
    sched = cfg["schedule"]
    assert (sched["open_hour"], sched["open_minute"]) == (9, 35)
    assert (sched["mid_hour"], sched["mid_minute"]) == (12, 30)
    assert (sched["close_hour"], sched["close_minute"]) == (15, 30)
    assert sched["timezone"] == "America/New_York"
    assert cfg["trading"]["max_positions"] == 12
    log.info("8) schedule OPEN 9:35 / MID 12:30 / CLOSE 15:30 ET, max_positions=12 unchanged")

    # --- 9. Secrets check ---
    import re
    for mod in ("core/sentiment_model.py", "core/weekly_news.py",
                "core/weekly_selector.py", "core/ticker_selector.py"):
        text = (REPO_ROOT / mod).read_text(encoding="utf-8")
        # Look for hardcoded string literals assigned to sensitive variable names
        for match in re.finditer(
            r"(?i)(api_key|apikey|secret|token|password)\s*=\s*['\"][^'\"]{8,}['\"]", text
        ):
            raise AssertionError(f"possible hardcoded secret in {mod}: {match.group(0)[:40]}")
    log.info("9) no hardcoded secrets in new modules")

    log.info("=========================== ALL CHECKS PASSED ===========================")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))