from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone

from robinhood_mcp.robinhood_client import RobinhoodMCPClient
from models.market_data import EquityQuote, OHLCVBar
from models.portfolio import Portfolio


class DataIngestLayer:
    """Phase 1 ingestion layer using the Robinhood MCP client."""

    def __init__(self, client: RobinhoodMCPClient | None = None) -> None:
        self.client = client or RobinhoodMCPClient()
        self.logger = logging.getLogger("robinhood-agent.core.data_ingest")

    async def ingest(self, tickers: list[str], existing_tickers: list[str]) -> dict:
        """Fetch portfolio, quotes, and historical bars for the union of watchlist and open positions."""
        deduped = list(dict.fromkeys([*tickers, *existing_tickers]))[:20]
        try:
            portfolio = await self.client.get_portfolio()
        except Exception as exc:
            self.logger.warning("portfolio_fetch_failed: %s", exc)
            portfolio = Portfolio()

        try:
            quotes = await self.client.get_equity_quotes(deduped)
            quote_map = {item.ticker: item for item in quotes}
        except Exception as exc:
            self.logger.warning("quote_fetch_failed: %s", exc)
            quote_map = {}

        tasks = [self.client.get_equity_historicals(ticker, span="3month") for ticker in deduped]
        historical_results = await asyncio.gather(*tasks, return_exceptions=True)

        historical_map: dict[str, list[OHLCVBar]] = {}
        for ticker, result in zip(deduped, historical_results, strict=False):
            if isinstance(result, Exception):
                self.logger.warning("historical_fetch_failed ticker=%s error=%s", ticker, result)
                historical_map[ticker] = []
            else:
                historical_map[ticker] = list(result)

        return {
            "portfolio": portfolio,
            "quotes": quote_map,
            "historicals": historical_map,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    async def ingest_quotes_only(self, tickers: list[str], existing_tickers: list[str] = None) -> dict:
        """Fetch only portfolio and quotes for the union of watchlist and open positions.

        This intentionally skips fetching historical bars to save API/time.
        """
        existing_tickers = existing_tickers or []
        deduped = list(dict.fromkeys([*tickers, *existing_tickers]))[:20]
        try:
            portfolio = await self.client.get_portfolio()
        except Exception as exc:
            self.logger.warning("portfolio_fetch_failed: %s", exc)
            portfolio = Portfolio()

        try:
            quotes = await self.client.get_equity_quotes(deduped)
            quote_map = {item.ticker: item for item in quotes}
        except Exception as exc:
            self.logger.warning("quote_fetch_failed: %s", exc)
            quote_map = {}

        return {
            "portfolio": portfolio,
            "quotes": quote_map,
            "historicals": {},
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
