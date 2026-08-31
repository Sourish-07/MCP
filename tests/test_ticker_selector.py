"""Unit tests for ticker selector logic."""

import pytest
from core.ticker_selector import select_universe, filter_candidates


class TestSelectUniverse:
    """Test cases for the select_universe function."""

    def test_empty_held_empty_candidates(self):
        """Test with no held tickers and no candidates."""
        held = set()
        candidates = {}
        result = select_universe(held, candidates, universe_size=10)
        assert result == []

    def test_empty_held_with_candidates(self):
        """Test with no held tickers but available candidates."""
        held = set()
        candidates = {
            "AAPL": 0.8,
            "MSFT": 0.6,
            "GOOGL": 0.4,
            "AMZN": 0.2,
            "TSLA": -0.1,
        }
        result = select_universe(held, candidates, universe_size=3)
        assert len(result) == 3
        assert result[0] == "AAPL"  # Highest score
        assert result[1] == "MSFT"
        assert result[2] == "GOOGL"

    def test_held_tickers_protected(self):
        """Test that held tickers are always protected."""
        held = {"AAPL", "MSFT"}
        candidates = {
            "AAPL": 0.9,  # Held
            "MSFT": 0.8,  # Held
            "GOOGL": 0.7,
            "AMZN": 0.6,
            "TSLA": 0.5,
        }
        result = select_universe(held, candidates, universe_size=3)
        assert len(result) == 3
        assert "AAPL" in result  # Protected
        assert "MSFT" in result  # Protected
        assert "GOOGL" in result  # One new ticker

    def test_held_tickers_exceeding_universe(self, caplog):
        """H > universe_size: ALL held tickers are kept, none are dropped."""
        held = {"AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"}  # 5 held > size 3
        candidates = {
            "AAPL": 0.9,  # Held
            "MSFT": 0.8,  # Held
            "GOOGL": 0.7,  # Held
            "AMZN": 0.6,  # Held
            "TSLA": 0.5,  # Held
            "NVDA": 0.4,
            "META": 0.3,
        }
        with caplog.at_level("WARNING"):
            result = select_universe(held, candidates, universe_size=3)
        # Universe temporarily exceeds target size: all 5 held, 0 new
        assert len(result) == 5
        assert set(result) == held  # No held ticker is ever dropped
        # Held tickers are ordered by sentiment (descending)
        assert result == ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]
        assert any(
            "holdings_exceed_universe_size" in rec.getMessage()
            for rec in caplog.records
        )

    def test_never_raises_on_insufficient_candidates(self):
        """Not enough candidates -> return best valid list, never raise."""
        held = {"AAPL", "MSFT"}
        candidates = {"AAPL": 0.5, "MSFT": 0.4, "GOOGL": 0.3}  # only 1 new
        result = select_universe(held, candidates, universe_size=10)
        assert set(result) == {"AAPL", "MSFT", "GOOGL"}
        assert len(result) == 3

    def test_sentiment_score_filtering(self):
        """Test that candidates below min_sentiment_score are filtered out."""
        held = {"AAPL"}
        candidates = {
            "AAPL": 0.9,  # Held
            "MSFT": 0.5,
            "GOOGL": -0.6,  # Below threshold
            "AMZN": 0.3,
        }
        result = select_universe(
            held, candidates, universe_size=3, min_sentiment_score=-0.5
        )
        assert len(result) == 3
        assert "AAPL" in result  # Protected
        assert "MSFT" in result
        assert "AMZN" in result
        assert "GOOGL" not in result  # Filtered out by sentiment

    def test_min_sentiment_score_zero(self):
        """Test with min_sentiment_score=0."""
        held = set()
        candidates = {
            "AAPL": 0.1,
            "MSFT": -0.1,
            "GOOGL": 0.0,
            "AMZN": 0.1,
        }
        result = select_universe(
            held, candidates, universe_size=3, min_sentiment_score=0.0
        )
        assert len(result) == 3
        assert "AAPL" in result
        assert "GOOGL" in result
        assert "AMZN" in result
        assert "MSFT" not in result

    def test_max_positions_never_drops_held(self):
        """12 held with universe_size=10/max_positions=12: ALL 12 held kept.

        max_positions/universe_size never cause a live holding to be dropped;
        the universe temporarily exceeds the configured size instead.
        """
        held = {f"TICK{i}" for i in range(12)}  # 12 held tickers
        candidates = {
            # Held tickers with explicit descending scores so ordering is deterministic
            **{f"TICK{i}": 1.0 - i * 0.01 for i in range(12)},
            # New candidates (excluded because all 12 slots are held)
            "AAPL": 0.9,
            "MSFT": 0.8,
            "GOOGL": 0.7,
        }
        result = select_universe(held, candidates, universe_size=10, max_positions=12)
        assert len(result) == 12  # Universe exceeds size -- nothing dropped
        assert set(result) == held
        assert result[0] == "TICK0"  # Highest-scoring held ticker first
        assert result[11] == "TICK11"  # Lowest-scoring held ticker still kept

    def test_case_insensitivity(self):
        """Test that ticker symbols are case-insensitive."""
        held = {"aapl", "MSFT"}
        candidates = {
            "AAPL": 0.9,
            "MSFT": 0.8,
            "GOOGL": 0.7,
        }
        result = select_universe(held, candidates, universe_size=3)
        assert len(result) == 3
        assert "AAPL" in result
        assert "MSFT" in result
        assert "GOOGL" in result


class TestFilterCandidates:
    """Test cases for the filter_candidates function."""

    def test_basic_filtering(self):
        """Test basic candidate filtering."""
        candidates = ["AAPL", "MSFT", "GOOGL", "AMZN"]
        held = {"AAPL", "MSFT"}
        result = filter_candidates(candidates, held)
        assert result == ["GOOGL", "AMZN"]

    def test_empty_candidates(self):
        """Test with empty candidates list."""
        candidates = []
        held = {"AAPL"}
        result = filter_candidates(candidates, held)
        assert result == []

    def test_all_held(self):
        """Test when all candidates are held."""
        candidates = ["AAPL", "MSFT"]
        held = {"AAPL", "MSFT"}
        result = filter_candidates(candidates, held)
        assert result == []