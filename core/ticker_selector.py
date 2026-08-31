"""Ticker selection logic for weekly sentiment-based universe selection."""

from __future__ import annotations

import logging
from typing import List, Set

logger = logging.getLogger("robinhood-agent.core.ticker_selector")


def select_universe(
    held_tickers: Set[str],
    scored_candidates: dict[str, float],
    universe_size: int = 10,
    max_positions: int = 12,
    min_sentiment_score: float = -1.0,
) -> List[str]:
    """Select the weekly universe with absolute holding protection.

    Non-negotiable rules:
      * EVERY currently held ticker (open position > 0) is ALWAYS included in
        the returned universe, for as long as it is held. A live holding is
        never dropped under any circumstance -- not for low sentiment and not
        even when holdings outnumber `universe_size` / `max_positions`.
      * If H > universe_size, ALL held tickers are returned (the universe
        temporarily exceeds the configured size) and a WARNING is logged.
      * If H <= universe_size, all held tickers are kept and the remaining
        slots are filled with the highest-scoring non-held candidates at or
        above `min_sentiment_score`.
      * If there are not enough candidates to reach `universe_size`, the best
        valid list available is returned. This function NEVER raises for
        "not enough candidates" and never invents tickers.

    Args:
        held_tickers: Set of currently held tickers (always protected)
        scored_candidates: Dict of ticker -> sentiment score [-1, +1]
        universe_size: Target universe size (default 10)
        max_positions: Maximum allowed positions (default 12; informational --
            holdings alone may exceed this; they are still never dropped)
        min_sentiment_score: Minimum sentiment score for NEW candidates only
            (default -1.0). Never applied to held tickers.

    Returns:
        List of tickers: all held tickers first (sorted by sentiment,
        descending, purely for deterministic output), followed by the top
        new candidates. May be smaller than `universe_size` when candidates
        are scarce, or larger when holdings exceed it.
    """
    # Ensure held_tickers is a set of uppercase strings
    held = {str(ticker).upper() for ticker in held_tickers}
    num_held = len(held)

    if num_held > universe_size:
        logger.warning(
            "holdings_exceed_universe_size held=%d universe_size=%d: keeping ALL held "
            "tickers (universe temporarily exceeds the configured size); no holding is dropped",
            num_held, universe_size,
        )

    # Held tickers are ranked by sentiment purely for deterministic ordering.
    # Ordering never causes a held ticker to be dropped.
    sorted_held = sorted(
        held, key=lambda t: scored_candidates.get(t, 0.0), reverse=True
    )

    # Filter candidates: non-held, score >= min_sentiment_score. Sort by score
    # (descending) and fill whatever slots remain after the held tickers.
    available_candidates = sorted(
        (
            (ticker.upper(), score)
            for ticker, score in scored_candidates.items()
            if ticker.upper() not in held and score >= min_sentiment_score
        ),
        key=lambda x: x[1],
        reverse=True,
    )

    num_needed = max(universe_size - num_held, 0)
    new_tickers = [ticker for ticker, _ in available_candidates[:num_needed]]
    selected = sorted_held + new_tickers

    if len(selected) != universe_size:
        logger.warning(
            "selection_size_mismatch selected=%d universe_size=%d held=%d available_candidates=%d "
            "(returning the best valid list instead of raising)",
            len(selected), universe_size, num_held, len(available_candidates),
        )
    else:
        logger.info(
            "Selected universe: %d held + %d new = %d total tickers",
            num_held, len(new_tickers), len(selected),
        )

    return selected


def filter_candidates(
    candidates: list[str],
    held_tickers: Set[str],
    min_price: float = 5.0,
    min_volume: float = 1_000_000,
) -> List[str]:
    """Filter candidate tickers based on basic criteria.

    Args:
        candidates: List of candidate tickers
        held_tickers: Set of currently held tickers (to exclude)
        min_price: Minimum price per share
        min_volume: Minimum average daily volume

    Returns:
        Filtered list of tickers
    """
    # This is a placeholder - in real implementation, you'd get price/volume data
    # For now, just exclude held tickers
    return [ticker for ticker in candidates if ticker.upper() not in held_tickers]