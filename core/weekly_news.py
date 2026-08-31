"""Weekly news fetcher with Finnhub integration, rate limiting, and caching."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import random
from datetime import date, datetime, timedelta
from pathlib import Path

import httpx

from models.news import NewsItem

logger = logging.getLogger("robinhood-agent.core.weekly_news")

_FINNHUB_COMPANY_NEWS = "https://finnhub.io/api/v1/company-news"


class RateLimiter:
    """Simple sliding-window rate limiter (default: Finnhub free tier 60 calls/min)."""

    def __init__(self, max_calls_per_minute: int = 60):
        self.max_calls = max_calls_per_minute
        self.calls: list[datetime] = []
        self.lock = asyncio.Lock()
        self.logger = logger

    async def acquire(self) -> None:
        """Acquire a rate limit token, waiting if necessary."""
        async with self.lock:
            now = datetime.now()
            self.calls = [c for c in self.calls if (now - c).total_seconds() < 60]
            if len(self.calls) >= self.max_calls:
                oldest = min(self.calls)
                wait_time = 60 - (now - oldest).total_seconds() + 0.1
                if wait_time > 0:
                    self.logger.info("rate_limit_wait %.1fs", wait_time)
                    await asyncio.sleep(wait_time)
                    now = datetime.now()
                    self.calls = [c for c in self.calls if (now - c).total_seconds() < 60]
            self.calls.append(datetime.now())


class WeeklyNewsFetcher:
    """Fetch weekly news from Finnhub with rate limiting, retries, and caching."""

    def __init__(self, lookback_days: int = 7, max_retries: int = 3):
        self.logger = logger
        self.lookback_days = lookback_days
        self.max_retries = max_retries
        self._api_key = os.getenv("FINNHUB_API_KEY", "")
        self._cache_dir = Path(__file__).resolve().parent.parent / "data"
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._cache_file = self._cache_dir / "weekly_news_cache.json"
        self._cache = self._load_cache()
        self._rate_limiter = RateLimiter(max_calls_per_minute=55)  # under 60 limit

    def _load_cache(self) -> dict:
        """Load cached news data."""
        if self._cache_file.exists():
            try:
                with open(self._cache_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as exc:
                self.logger.warning("weekly_news_cache_load_failed: %s", exc)
        return {}

    def _save_cache(self) -> None:
        """Save cache to disk."""
        try:
            with open(self._cache_file, "w", encoding="utf-8") as f:
                json.dump(self._cache, f, indent=2)
        except Exception as exc:
            self.logger.warning("weekly_news_cache_save_failed: %s", exc)

    def _is_cache_fresh(self, cached_entry: dict, max_age_hours: int = 24) -> bool:
        """Check whether a cache entry is still fresh."""
        try:
            cached_at = datetime.fromisoformat(cached_entry.get("cached_at", ""))
            age_hours = (datetime.now() - cached_at).total_seconds() / 3600
            return age_hours < max_age_hours
        except Exception:
            return False

    async def fetch_company_news(
        self,
        symbol: str,
        from_date: date | None = None,
        to_date: date | None = None,
    ) -> list[NewsItem]:
        """Fetch company news for a symbol with rate limiting, retries, and caching.

        Returns an empty list (not an exception) on failure so callers can
        treat the ticker as neutral (score 0.0).
        """
        if not self._api_key:
            self.logger.warning("weekly_news no FINNHUB_API_KEY set, returning empty")
            return []

        to_date = to_date or date.today()
        from_date = from_date or (to_date - timedelta(days=self.lookback_days))
        cache_key = f"{symbol.upper()}_{from_date.isoformat()}_{to_date.isoformat()}"

        cached = self._cache.get(cache_key)
        if cached and self._is_cache_fresh(cached):
            self.logger.debug("weekly_news cache hit symbol=%s", symbol)
            return [NewsItem.model_validate(item) for item in cached.get("items", [])]

        url = (
            f"{_FINNHUB_COMPANY_NEWS}?symbol={symbol.upper()}"
            f"&from={from_date.isoformat()}&to={to_date.isoformat()}"
            f"&token={self._api_key}"
        )

        for attempt in range(1, self.max_retries + 1):
            await self._rate_limiter.acquire()
            try:
                async with httpx.AsyncClient() as client:
                    response = await client.get(url, timeout=30.0)
                    if response.status_code == 429:
                        backoff = (2 ** attempt) + random.uniform(0, 1)
                        self.logger.warning(
                            "weekly_news rate_limited symbol=%s attempt=%d backoff=%.1fs",
                            symbol, attempt, backoff,
                        )
                        await asyncio.sleep(backoff)
                        continue
                    response.raise_for_status()
                    data = response.json()
                    items = self._parse_items(symbol, data)
                    self._cache[cache_key] = {
                        "cached_at": datetime.now().isoformat(),
                        "items": [item.model_dump(mode="json") for item in items],
                    }
                    self._save_cache()
                    self.logger.info("weekly_news_fetched symbol=%s count=%d", symbol, len(items))
                    return items
            except httpx.HTTPStatusError as exc:
                self.logger.warning(
                    "weekly_news_http_error symbol=%s status=%s", symbol, exc.response.status_code
                )
                await asyncio.sleep((2 ** attempt) + random.uniform(0, 1))
            except Exception as exc:
                self.logger.warning(
                    "weekly_news_error symbol=%s attempt=%d error=%s", symbol, attempt, exc
                )
                await asyncio.sleep((2 ** attempt) + random.uniform(0, 1))

        self.logger.error("weekly_news_failed symbol=%s after %d attempts", symbol, self.max_retries)
        return []

    def _parse_items(self, symbol: str, data: list) -> list[NewsItem]:
        """Parse raw Finnhub response items into NewsItem models."""
        news_items: list[NewsItem] = []
        for item in data if isinstance(data, list) else []:
            try:
                news_items.append(
                    NewsItem(
                        id=int(item.get("id", 0)),
                        headline=str(item.get("headline", "")),
                        summary=item.get("summary"),
                        source=str(item.get("source", "finnhub")),
                        url=item.get("url"),
                        related=symbol.upper(),
                        category=str(item.get("category", "")),
                        datetime=int(item.get("datetime", 0)),
                    )
                )
            except Exception as exc:
                self.logger.warning("weekly_news_parse_failed symbol=%s error=%s", symbol, exc)
                continue
        return news_items

    async def fetch_news_for_universe(
        self, symbols: list[str], max_concurrent: int = 5
    ) -> dict[str, list[NewsItem]]:
        """Fetch news for multiple symbols with concurrency control."""
        semaphore = asyncio.Semaphore(max_concurrent)

        async def fetch_with_semaphore(symbol: str) -> tuple[str, list[NewsItem]]:
            async with semaphore:
                news = await self.fetch_company_news(symbol)
                return symbol, news

        tasks = [fetch_with_semaphore(symbol) for symbol in symbols]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        news_dict: dict[str, list[NewsItem]] = {}
        for symbol, result in zip(symbols, results):
            if isinstance(result, Exception):
                self.logger.error("weekly_news universe fetch failed symbol=%s: %s", symbol, result)
                news_dict[symbol] = []
            else:
                _, news = result
                news_dict[symbol] = news

        self.logger.info("weekly_news universe complete symbols=%d", len(news_dict))
        return news_dict


async def fetch_weekly_news(symbols: list[str], lookback_days: int = 7) -> dict[str, list[NewsItem]]:
    """Convenience function: fetch weekly news for multiple symbols."""
    fetcher = WeeklyNewsFetcher(lookback_days=lookback_days)
    return await fetcher.fetch_news_for_universe(symbols)