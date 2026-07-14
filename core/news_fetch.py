from __future__ import annotations

import asyncio
import json
import logging
import re
import xml.etree.ElementTree as ET
from datetime import datetime, timezone, timedelta
from pathlib import Path

import httpx

from utils.anthropic_client import AnthropicClient


class NewsFetcher:
    """Fetch RSS headlines and supplement with Haiku summaries when needed.

    Adds headline deduplication across cycles using data/seen_headlines.json.
    """

    def __init__(self, anthropic_client: AnthropicClient) -> None:
        self.client = anthropic_client
        self.logger = logging.getLogger("robinhood-agent.core.news_fetch")
        self._seen_path = Path(__file__).resolve().parent.parent / "data" / "seen_headlines.json"
        self._seen_path.parent.mkdir(parents=True, exist_ok=True)

    async def fetch_news(self, tickers: list[str], cycle_type: str = "OPEN") -> dict[str, list[str]]:
        """Return ticker -> headline list using RSS and optional Haiku summaries.

        cycle_type influences Haiku fallback prompts and deduplication ensures MID/CLOSE
        cycles only surface headlines not already seen today.
        """
        tickers = list(dict.fromkeys(tickers))
        results: dict[str, list[str]] = {ticker: [] for ticker in tickers}

        today = datetime.now(timezone.utc).date().isoformat()
        try:
            raw = self._seen_path.read_text(encoding="utf-8") if self._seen_path.exists() else ""
            cache = json.loads(raw) if raw else {"date": today, "seen": []}
        except Exception:
            cache = {"date": today, "seen": []}

        if cache.get("date") != today:
            cache = {"date": today, "seen": []}

        seen_set = set(cache.get("seen", []))
        new_seen = set()

        try:
            rss_items = await self._fetch_rss_items(tickers)
            for ticker, items in rss_items.items():
                # filter out headlines already seen today
                fresh = [t for t in items if t not in seen_set]
                # limit to first 5 fresh
                results[ticker] = fresh[:5]
                new_seen.update(results[ticker])

            # For tickers lacking sufficient fresh headlines, invoke fallback summaries
            fallbacks = [ticker for ticker in tickers if len(results[ticker]) < 2]
            tasks = [self._generate_fallback_summary(ticker, cycle_type) for ticker in fallbacks]
            if tasks:
                generated = await asyncio.gather(*tasks, return_exceptions=True)
                for ticker, item in zip(fallbacks, generated, strict=False):
                    if isinstance(item, Exception):
                        self.logger.warning("fallback_news_failed ticker=%s error=%s", ticker, item)
                        continue
                    # include fallback lines even if they were seen previously; they are model-generated
                    results[ticker] = (results[ticker] + item)[:5]
        except Exception as exc:
            self.logger.warning("news_fetch_failed: %s", exc)

        # persist updated seen cache
        try:
            cache_out = {"date": today, "seen": list(seen_set.union(new_seen))}
            self._seen_path.write_text(json.dumps(cache_out, indent=2), encoding="utf-8")
        except Exception as exc:
            self.logger.warning("seen_cache_write_failed: %s", exc)

        return {ticker: results.get(ticker, [])[:5] for ticker in tickers}

    async def _fetch_rss_items(self, tickers: list[str]) -> dict[str, list[str]]:
        """Fetch Yahoo headlines and WSJ headlines, then filter them for relevant tickers."""
        async with httpx.AsyncClient(timeout=20.0) as client:
            all_tasks = [client.get(f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}&region=US&lang=en-US") for ticker in tickers]
            all_tasks.append(client.get("https://feeds.a.dj.com/rss/RSSWSJD.xml"))
            all_responses = await asyncio.gather(*all_tasks, return_exceptions=True)
            yahoo_responses = all_responses[:-1]
            wsj_response = all_responses[-1]
            wsj_items = self._parse_wsj_items(wsj_response)

        items_by_ticker: dict[str, list[str]] = {ticker: [] for ticker in tickers}
        cutoff = datetime.now(timezone.utc) - timedelta(hours=24)

        for ticker in tickers:
            seen: set[str] = set()
            for response in yahoo_responses:
                if isinstance(response, Exception):
                    continue
                if response.status_code != 200:
                    continue
                try:
                    root = ET.fromstring(response.text)
                except ET.ParseError:
                    continue
                for item in root.findall('.//item'):
                    title = (item.findtext('title') or '').strip()
                    pub_date_raw = item.findtext('pubDate') or ''
                    if not title or not self._is_recent(pub_date_raw, cutoff):
                        continue
                    if self._mentions_ticker(title, ticker):
                        if title not in seen:
                            seen.add(title)
                            items_by_ticker[ticker].append(title)
            for title in wsj_items:
                if self._mentions_ticker(title, ticker) and title not in seen:
                    seen.add(title)
                    items_by_ticker[ticker].append(title)

        return items_by_ticker

    @staticmethod
    def _parse_wsj_items(response: httpx.Response) -> list[str]:
        """Parse the WSJ RSS feed and return relevant titles."""
        try:
            if response.status_code != 200:
                return []
            root = ET.fromstring(response.text)
        except Exception:
            return []
        titles = []
        for item in root.findall('.//item'):
            title = (item.findtext('title') or '').strip()
            if title:
                titles.append(title)
        return titles

    @staticmethod
    def _mentions_ticker(title: str, ticker: str) -> bool:
        """Return True when a headline explicitly mentions the ticker as a standalone token."""
        pattern = rf"(?<![A-Za-z0-9]){re.escape(ticker.upper())}(?![A-Za-z0-9])"
        return bool(re.search(pattern, title.upper()))

    @staticmethod
    def _is_recent(pub_date_raw: str, cutoff: datetime) -> bool:
        """Return True when the item is within 24 hours."""
        try:
            dt = datetime.strptime(pub_date_raw, "%a, %d %b %Y %H:%M:%S %z")
        except Exception:
            return False
        return dt.astimezone(timezone.utc) >= cutoff

    async def _generate_fallback_summary(self, ticker: str, cycle_type: str) -> list[str]:
        """Use Haiku to generate two brief factual statements when RSS is sparse.

        The cycle_type is included to bias the statements toward opening catalysts or closing risks.
        """
        try:
            system = (
                "You are a financial analyst. Given a ticker, write 2 brief factual statements about what is "
                "currently relevant for this stock based on your knowledge. Be specific to the company, not generic. "
                "No markdown."
            )
            user = f"Ticker: {ticker}. Cycle: {cycle_type}. Date: {datetime.now(timezone.utc).date().isoformat()}. Two key facts:"
            text = await self.client.complete(system, user)
            return [line.strip(" -•") for line in text.splitlines() if line.strip()][:2]
        except Exception as exc:
            self.logger.warning("fallback_summary_failed ticker=%s error=%s", ticker, exc)
            return []
