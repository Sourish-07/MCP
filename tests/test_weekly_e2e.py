"""End-to-end integration tests for the weekly sentiment selection layer.

Covers the Checkpoint 6 verification requirements with mocked I/O.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.weekly_selector import WeeklySelector
from models.portfolio import PositionRecord


@pytest.fixture
def base_config() -> dict:
    return {
        "trading": {"max_positions": 12},
        "watchlist": {"default_tickers": ["JPM", "NVDA", "AVGO", "CAT", "MU",
                                          "TSM", "UNH", "XOM", "AMD", "COST"]},
        "sentiment_selection": {
            "enabled": True,
            "universe_size": 10,
            "lookback_days": 7,
            "min_sentiment_score": -1.0,
            "schedule_day": "sun",
            "schedule_hour": 20,
        },
    }


def _make_selector(config: dict, tmp_path) -> WeeklySelector:
    """Build a WeeklySelector pointed at temp paths without loading the model."""
    import logging
    selector = WeeklySelector.__new__(WeeklySelector)
    selector.config = config
    selector.logger = logging.getLogger("test.weekly_selector")
    ss = config.get("sentiment_selection", {})
    selector.enabled = ss.get("enabled", False)
    selector.universe_size = ss.get("universe_size", 10)
    selector.max_positions = config.get("trading", {}).get("max_positions", 12)
    selector.lookback_days = ss.get("lookback_days", 7)
    selector.min_sentiment_score = ss.get("min_sentiment_score", -1.0)
    selector.data_dir = tmp_path
    selector.selected_universe_path = tmp_path / "selected_universe.json"
    selector.candidate_universe_path = tmp_path / "candidate_universe.json"
    selector.position_manager = None
    selector.news_fetcher = AsyncMock()
    selector.sentiment_model = None
    selector.candidate_universe = []
    selector.selected_universe = []
    return selector


def _fake_scores(tickers: list[str]) -> dict[str, float]:
    """Deterministic fake sentiment scores (descending)."""
    return {t: 1.0 - i * 0.05 for i, t in enumerate(tickers)}


def _stub_holdings(tickers_with_qty: list[tuple[str, float]]):
    """Return a stub position_manager whose get_open_records yields the records."""
    records = [
        PositionRecord(ticker=t, entry_price=100.0, quantity=q)
        for t, q in tickers_with_qty
    ]
    return type("PM", (), {"get_open_records": staticmethod(lambda: records)})()


@pytest.mark.asyncio
async def test_e2e_fresh_selection_no_holdings(base_config, tmp_path):
    """Checkpoint 6.1: H=0 -> refresh all 10 from sentiment ranking; persist result."""
    selector = _make_selector(base_config, tmp_path)
    candidates = [f"CAND{i}" for i in range(30)]
    selector.candidate_universe = candidates
    selector.position_manager = _stub_holdings([])

    async def fake_scores(tickers):
        return _fake_scores(tickers)
    selector.fetch_news_and_scores = fake_scores

    selected = await selector.run_selection()

    assert len(selected) == 10
    assert selected[0] == "CAND0"
    saved = json.loads(selector.selected_universe_path.read_text(encoding="utf-8"))
    assert saved["tickers"] == selected


@pytest.mark.asyncio
async def test_e2e_three_holdings_protected(base_config, tmp_path):
    """Checkpoint 6.2: H=3 -> 3 stay + 7 new chosen, even with terrible held sentiment."""
    selector = _make_selector(base_config, tmp_path)
    selector.candidate_universe = [f"CAND{i}" for i in range(30)]

    held = {"HELD1", "HELD2", "HELD3"}
    selector.position_manager = _stub_holdings([(t, 1.0) for t in held])

    async def fake_scores(tickers):
        scores = _fake_scores(tickers)
        for h in held:
            scores[h] = -0.9  # terrible sentiment, still protected
        return scores
    selector.fetch_news_and_scores = fake_scores

    selected = await selector.run_selection()

    assert len(selected) == 10
    for h in held:
        assert h in selected
    new_ones = [t for t in selected if t not in held]
    assert len(new_ones) == 7
    assert all(t.startswith("CAND") for t in new_ones)


@pytest.mark.asyncio
async def test_e2e_replacement_after_selling(base_config, tmp_path):
    """Checkpoint 6.3: Sell to zero -> ticker eligible for replacement next run."""
    selector = _make_selector(base_config, tmp_path)
    selector.candidate_universe = [f"CAND{i}" for i in range(30)]

    # First run: HELD1 held
    selector.position_manager = _stub_holdings([("HELD1", 1.0)])

    async def fake_scores(tickers):
        scores = _fake_scores(tickers)
        scores["HELD1"] = -0.9
        return scores
    selector.fetch_news_and_scores = fake_scores

    selected_run1 = await selector.run_selection()
    assert "HELD1" in selected_run1

    # Sell to zero
    selector.position_manager = _stub_holdings([("HELD1", 0.0)])
    selected_run2 = await selector.run_selection()
    assert "HELD1" not in selected_run2


@pytest.mark.asyncio
async def test_e2e_finnhub_failure_keeps_previous(base_config, tmp_path):
    """Checkpoint 6.4: fetch failure -> keep previous universe + log error."""
    selector = _make_selector(base_config, tmp_path)
    selector.candidate_universe = [f"CAND{i}" for i in range(15)]

    previous = [f"PREV{i}" for i in range(10)]
    selector.selected_universe = previous
    selector.selected_universe_path.write_text(
        json.dumps({"tickers": previous, "scores": {}, "timestamp": "2026-08-24T00:00:00"}),
        encoding="utf-8",
    )
    selector.position_manager = _stub_holdings([])

    async def failing_fetch(tickers):
        raise ConnectionError("finnhub down")
    selector.fetch_news_and_scores = failing_fetch

    selected = await selector.run_selection()
    assert selected == previous


@pytest.mark.asyncio
async def test_e2e_all_held_equals_universe(base_config, tmp_path):
    """H=10 edge case: all holdings -> universe stays as-is."""
    selector = _make_selector(base_config, tmp_path)
    selector.candidate_universe = [f"CAND{i}" for i in range(30)]

    held = {f"HELD{i}" for i in range(10)}
    selector.position_manager = _stub_holdings([(t, 1.0) for t in held])

    async def fake_scores(tickers):
        return {t: 0.0 for t in tickers}
    selector.fetch_news_and_scores = fake_scores

    selected = await selector.run_selection()
    assert len(selected) == 10
    assert set(selected) == held


@pytest.mark.asyncio
async def test_e2e_holdings_exceeding_universe_size(base_config, tmp_path):
    """H > universe_size: ALL 12 held tickers kept, universe temporarily 12."""
    selector = _make_selector(base_config, tmp_path)
    selector.candidate_universe = [f"CAND{i}" for i in range(30)]

    held = {f"HELD{i}" for i in range(12)}  # 12 held > universe_size 10
    selector.position_manager = _stub_holdings([(t, 1.0) for t in held])

    async def fake_scores(tickers):
        return _fake_scores(tickers)
    selector.fetch_news_and_scores = fake_scores

    selected = await selector.run_selection()
    assert len(selected) == 12  # Universe temporarily exceeds size
    assert held.issubset(set(selected))  # Every single holding is protected


@pytest.mark.asyncio
async def test_e2e_mean_aggregation(base_config, tmp_path):
    """Per-ticker score = MEAN of its headline scores (not first/max)."""
    selector = _make_selector(base_config, tmp_path)
    selector.candidate_universe = ["AAA", "BBB"]
    selector.position_manager = _stub_holdings([])

    from models.news import NewsItem

    def _item(headline: str) -> NewsItem:
        return NewsItem(id=1, headline=headline, source="test", datetime=0, related="AAA")

    selector.news_fetcher.fetch_news_for_universe = AsyncMock(
        return_value={
            "AAA": [_item("h1"), _item("h2"), _item("h3")],  # 3 headlines
            "BBB": [],  # zero headlines -> neutral
        }
    )

    class _FakeModel:
        def score_headlines(self, headlines):
            assert headlines == ["h1", "h2", "h3"]  # every headline is scored
            return [0.9, 0.3, -0.3]

    selector.sentiment_model = _FakeModel()

    scores = await selector.fetch_news_and_scores(["AAA", "BBB"])
    assert scores["AAA"] == pytest.approx(0.3)  # mean of [0.9, 0.3, -0.3]
    assert scores["BBB"] == 0.0  # zero headlines -> neutral


@pytest.mark.asyncio
async def test_e2e_empty_news_neutral_scores(base_config, tmp_path):
    """Tickers with no news get neutral 0.0, no crash."""
    selector = _make_selector(base_config, tmp_path)
    selector.candidate_universe = ["AAPL", "MSFT", "GOOGL"]
    selector.position_manager = _stub_holdings([])

    # Real fetch_news_and_scores with mocked fetcher returning empty news
    from models.news import NewsItem
    selector.news_fetcher.fetch_news_for_universe = AsyncMock(
        return_value={"AAPL": [], "MSFT": [], "GOOGL": []}
    )
    selector.sentiment_model = None

    scores = await selector.fetch_news_and_scores(["AAPL", "MSFT", "GOOGL"])
    assert scores == {"AAPL": 0.0, "MSFT": 0.0, "GOOGL": 0.0}


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))