"""Fetch news from Finnhub REST API with structured models and id-based
deduplication across cycles.

Replaces the prior Yahoo/WSJ RSS + Haiku-fabrication pipeline.

Features:
- Finnhub company-news + general market news
- ID-based deduplication (cycles only show articles once per day)
- Freshness window filter + per-ticker cap
- Optional Haiku fallback for tickers with zero news
- Optional Haiku summarizer that condenses N articles per ticker into a
  single digest (separate cheap LLM call via Anthropic Haiku)
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from datetime import UTC, date, datetime, timedelta, timezone
from pathlib import Path

import httpx

from models.news import NewsItem
from utils.anthropic_client import AnthropicClient

_FINNHUB_COMPANY_NEWS = "https://finnhub.io/api/v1/company-news"
_FINNHUB_GENERAL_NEWS = "https://finnhub.io/api/v1/news"


class NewsFetcher:
    """Fetch per-ticker and general-market news from Finnhub REST.

    Each fetched article is parsed into a ``NewsItem``.  Deduplication
    uses Finnhub's ``id`` field, persisted to ``data/seen_headlines.json``
    so MID and CLOSE cycles don't re-surface the same articles within
    the same calendar day.

    Optional Haiku-fallback is only invoked when a ticker receives zero
    items **and** the config flag ``haiku_fallback.enabled`` is True.
    """

    def __init__(
        self,
        anthropic_client: AnthropicClient | None = None,
        finnhub_api_key: str = "",
        settings: dict | None = None,
    ) -> None:
        self.client = anthropic_client
        self.logger = logging.getLogger("robinhood-agent.core.news_fetch")
        self._seen_path = Path(__file__).resolve().parent.parent / "data" / "seen_headlines.json"
        self._seen_path.parent.mkdir(parents=True, exist_ok=True)

        cfg = settings or {}
        self._api_key: str = finnhub_api_key or os.getenv("FINNHUB_API_KEY", "")
        self._lookback_days: int = int(cfg.get("lookback_days", 3))
        self._max_per_ticker: int = int(cfg.get("max_per_ticker", 5))
        self._fresh_window_hours: int = int(cfg.get("fresh_window_hours", 24))
        gm = cfg.get("general_market", {})
        self._gm_enabled: bool = bool(gm.get("enabled", True))
        self._gm_category: str = str(gm.get("category", "general"))
        self._gm_count: int = int(gm.get("count", 8))
        hb = cfg.get("haiku_fallback", {})
        self._haiku_fallback_enabled: bool = bool(hb.get("enabled", False))
        hs = cfg.get("haiku_summary", {})
        self._haiku_summary_enabled: bool = bool(hs.get("enabled", False))
        self._haiku_summary_concurrent: int = min(int(hs.get("default_concurrent", 10)), 20)

    # ------------------------------------------------------------------
    # Public API — same signature shape as before  (dict[str, …])
    # ------------------------------------------------------------------

    async def fetch_news(
        self, tickers: list[str], cycle_type: str = "OPEN"
    ) -> dict[str, list[NewsItem]]:
        """Return ``{ticker: [NewsItem, …]}`` for every supplied ticker.

        * Builds a per-day seen-set of Finnhub ``id`` values.
        * Fetches company-news for each ticker concurrently.
        * Optionally fetches general-market news and attaches it to
          ``"MARKET"`` and merges into ``"SPY"``.
        * Applies 24‑h freshness filter and ``max_per_ticker`` cap.
        """
        tickers = list(dict.fromkeys(tickers))  # de-dupe, preserve order
        results: dict[str, list[NewsItem]] = {ticker: [] for ticker in tickers}

        today = date.today()
        today_iso = today.isoformat()

        # ── load / initialise dedup state ──────────────────────────
        try:
            raw = self._seen_path.read_text(encoding="utf-8") if self._seen_path.exists() else ""
            cache = json.loads(raw) if raw else {"date": today_iso, "seen": []}
        except Exception:
            cache = {"date": today_iso, "seen": []}

        if cache.get("date") != today_iso:
            cache = {"date": today_iso, "seen": []}

        seen_ids: set[int] = set(cache.get("seen", []))
        new_ids: set[int] = set()

        # ── time boundaries for fresh-window filter ────────────────
        now_epoch = int(datetime.now(timezone.utc).timestamp())
        cutoff_epoch = now_epoch - (self._fresh_window_hours * 3600)

        # ── concurrent company-news for every ticker ───────────────
        from_str = (today - timedelta(days=self._lookback_days)).isoformat()
        to_str = today.isoformat()

        async with httpx.AsyncClient(timeout=30.0) as client:
            # Limit concurrent Finnhub requests (free-tier rate limits)
            sem = asyncio.Semaphore(3)

            async def _bounded_company_news(ticker: str):
                async with sem:
                    return await self._fetch_company_news(client, ticker, from_str, to_str)

            co_tasks = [
                _bounded_company_news(ticker)
                for ticker in tickers
            ]

            gm_task: asyncio.Task | None = None
            if self._gm_enabled:
                gm_task = asyncio.create_task(
                    self._fetch_general_news(client)
                )

            co_results = await asyncio.gather(*co_tasks, return_exceptions=True)

            for ticker, co_result in zip(tickers, co_results):
                if isinstance(co_result, Exception):
                    self.logger.warning(
                        "company_news_failed ticker=%s error=%r type=%s",
                        ticker, co_result, type(co_result).__name__
                    )
                    continue
                articles: list[NewsItem] = co_result  # type: ignore[assignment]
                filtered = self._filter_fresh(articles, seen_ids, new_ids, cutoff_epoch)
                results[ticker] = filtered

            # ── general-market news ─────────────────────────────────
            if self._gm_enabled and gm_task is not None:
                try:
                    gm_articles = await gm_task
                except Exception as exc:
                    self.logger.warning(
                        "general_news_failed: %r type=%s", exc, type(exc).__name__
                    )
                    gm_articles = []

                gm_filtered = self._filter_fresh(
                    gm_articles, seen_ids, new_ids, cutoff_epoch, cap=self._gm_count
                )
                if gm_filtered:
                    results["MARKET"] = gm_filtered
                    # merge into SPY so the SPY context includes broad market news
                    current_spy = results.get("SPY", [])
                    current_spy = (current_spy + gm_filtered)[: self._max_per_ticker]
                    results["SPY"] = current_spy

            # ── optional Haiku fallback (off by default) ───────────
            if self._haiku_fallback_enabled and self.client is not None:
                empty_tickers = [t for t in tickers if len(results[t]) == 0]
                if empty_tickers:
                    fb_tasks = [
                        self._generate_fallback_summary(ticker, cycle_type)
                        for ticker in empty_tickers
                    ]
                    fb_results = await asyncio.gather(*fb_tasks, return_exceptions=True)
                    for ticker, fb_result in zip(empty_tickers, fb_results):
                        if isinstance(fb_result, Exception):
                            self.logger.warning(
                                "haiku_fallback_failed ticker=%s error=%s", ticker, fb_result
                            )
                            continue
                        if fb_result:
                            results[ticker] = fb_result  # type: ignore[list-item]

            # ── optional Haiku summarizer — one summary per ticker from all its news ──
            if self._haiku_summary_enabled and self.client is not None:
                summary_tickers = [
                    t for t in tickers if len(results.get(t, [])) > 0
                ]
                if summary_tickers:
                    # batch to respect default_concurrent limit
                    sem = asyncio.Semaphore(self._haiku_summary_concurrent)
                    async def _summarize(ticker: str) -> list[NewsItem]:
                        async with sem:
                            return await self._generate_ticker_summary(
                                ticker, results.get(ticker, [])
                            )
                    hs_tasks = [_summarize(t) for t in summary_tickers]
                    hs_results = await asyncio.gather(*hs_tasks, return_exceptions=True)
                    for ticker, hs_result in zip(summary_tickers, hs_results):
                        if isinstance(hs_result, Exception):
                            self.logger.warning(
                                "haiku_summary_failed ticker=%s error=%s", ticker, hs_result
                            )
                            continue
                        if hs_result:
                            # Append the AI summary as a marked entry alongside
                            # the raw articles — downstream renderer picks it up.
                            results[ticker] = hs_result + results[ticker]

            # ── persist updated seen-set ───────────────────────────
            try:
                all_seen = sorted(seen_ids | new_ids)
                self._seen_path.write_text(
                    json.dumps({"date": today_iso, "seen": all_seen}, indent=2),
                    encoding="utf-8",
                )
            except Exception as exc:
                self.logger.warning("seen_cache_write_failed: %s", exc)

            return results

    # ------------------------------------------------------------------
    # Internal fetch helpers
    # ------------------------------------------------------------------

    async def _fetch_company_news(
        self, client: httpx.AsyncClient, ticker: str, from_date: str, to_date: str
    ) -> list[NewsItem]:
        params = {
            "symbol": ticker,
            "from": from_date,
            "to": to_date,
            "token": self._api_key,
        }
        response = await client.get(_FINNHUB_COMPANY_NEWS, params=params)
        if response.status_code != 200:
            self.logger.warning(
                "finnhub_http_error endpoint=company-news symbol=%s status=%s body=%s",
                ticker, response.status_code, response.text[:300]
            )
        response.raise_for_status()
        raw_list = response.json()
        return self._parse_articles(raw_list, related=ticker)

    async def _fetch_general_news(self, client: httpx.AsyncClient) -> list[NewsItem]:
        params = {
            "category": self._gm_category,
            "token": self._api_key,
        }
        response = await client.get(_FINNHUB_GENERAL_NEWS, params=params)
        if response.status_code != 200:
            self.logger.warning(
                "finnhub_http_error endpoint=general-news status=%s body=%s",
                response.status_code, response.text[:300]
            )
        response.raise_for_status()
        raw_list = response.json()
        return self._parse_articles(raw_list)

    # ------------------------------------------------------------------
    # Parse & filter
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_articles(
        raw: list[dict], related: str = ""
    ) -> list[NewsItem]:
        articles: list[NewsItem] = []
        for obj in raw or []:
            try:
                item = NewsItem(
                    id=int(obj.get("id", 0)),
                    headline=str(obj.get("headline", "")).strip(),
                    summary=(
                        str(obj["summary"]).strip()
                        if obj.get("summary") is not None
                        else None
                    ),
                    source=str(obj.get("source", "")),
                    url=obj.get("url"),
                    image=obj.get("image"),
                    related=str(obj.get("related", related)) or related,
                    category=str(obj.get("category", "")),
                    datetime=int(obj.get("datetime", 0)),
                )
                if item.headline:
                    articles.append(item)
            except Exception as exc:
                logging.getLogger("robinhood-agent.core.news_fetch").debug(
                    "skip_malformed_article raw=%s error=%s",
                    obj.get("id", "<no-id>"),
                    exc,
                )
        return articles

    def _filter_fresh(
        self,
        articles: list[NewsItem],
        seen_ids: set[int],
        new_ids: set[int] | None = None,
        cutoff_epoch: int = 0,
        cap: int | None = None,
    ) -> list[NewsItem]:
        cap = cap if cap is not None else self._max_per_ticker
        fresh: list[NewsItem] = []
        for item in articles:
            if item.id in seen_ids:
                continue
            if cutoff_epoch > 0 and item.datetime < cutoff_epoch:
                continue
            fresh.append(item)
            if new_ids is not None:
                new_ids.add(item.id)
        # Most-recent-first before capping (Finnhub already returns
        # roughly desc by datetime, but enforce explicitly for safety).
        fresh.sort(key=lambda x: x.datetime, reverse=True)
        return fresh[:cap]

    # ------------------------------------------------------------------
    # Haiku helpers — fallback (disabled by default) and summarizer
    # ------------------------------------------------------------------

    async def _generate_ticker_summary(
        self, ticker: str, articles: list[NewsItem]
    ) -> list[NewsItem]:
        """Summarize all articles for `ticker` into one concise digest entry.

        Returns a list with a synthetic "summary" ``NewsItem``, always
        exactly one entry. Does not modify original articles.
        """
        if not self.client or not articles:
            return []
        headlines = "\n".join(
            f"- [{n.source}] {n.headline}" for n in articles
        )
        # merge sources into a short set for labelling
        sources = ", ".join(
            sorted({n.source for n in articles if n.source} or ["market"])
        )
        system = [
            {
                "type": "text",
                "text": (
                    "You synthesise financial news for a systematic trading agent. "
                    "You receive a list of recent headlines for one ticker. "
                    "Summarise them into 1-3 short bullet points, focusing on impact "
                    "to fundamentals, sentiment, or price action. "
                    "Do NOT invent prices, earnings numbers, or trade recommendations. "
                    "Use plain language with no markdown except * around ticker names."
                ),
                "cache_control": {"type": "ephemeral"},
            }
        ]
        user_text = f"Ticker: {ticker}. Headlines:\n{headlines}"
        try:
            digest = await self.client.complete(system, user_text)
        except Exception:
            return []  # summarization is best-effort; skip on error

        lines = [l.strip(" -•") for l in digest.splitlines() if l.strip()]
        if not lines:
            return []
        return [
            NewsItem(
                id=abs(hash(f"{ticker}_summary_{lines[0]}")),
                headline="; ".join(lines),
                source=f"Haiku summary ({sources})" if sources else "Haiku summary",
                summary=None,
                related=ticker,
                category="llm_summary",
                datetime=int(datetime.now(UTC).timestamp()),
            )
        ]

    async def _generate_fallback_summary(
        self, ticker: str, cycle_type: str
    ) -> list[NewsItem] | None:
        """Generate ONE summary headline when no live news was found.

        This path is only entered when ``haiku_fallback.enabled`` is True
        in settings and is meant as a safety net, not a replacement for
        real news.  The output is clearly labelled ``source``.
        """
        if not self.client:
            return None
        try:
            system = [
                {
                    "type": "text",
                    "text": (
                        "You are the fallback responder in a financial news pipeline. "
                        "You are given a ticker for which the live news API returned zero "
                        "results. Based on the latest facts you know about this company, "
                        "write ONE very brief factual statement (1-2 sentences) of what is "
                        "currently relevant. Do not speculate, do not invent prices, do not "
                        "use markdown. If you are unsure, respond with 'No additional context "
                        "available for this ticker.'"
                    ),
                    "cache_control": {"type": "ephemeral"},
                }
            ]
            user = (
                f"Ticker: {ticker}. Cycle: {cycle_type}. "
                f"Date: {date.today().isoformat()}. One key fact:"
            )
            text = await self.client.complete(system, user)
            lines = [
                line.strip(" -•")
                for line in text.splitlines()
                if line.strip()
            ]
            if not lines:
                return None
            # return as a single synthetic NewsItem for uniform rendering
            return [
                NewsItem(
                    id=abs(hash(f"{ticker}_{lines[0]}")),
                    headline=lines[0],
                    source="Haiku fallback summary",
                    summary=None,
                    related=ticker,
                    category="fallback",
                    datetime=int(datetime.now(UTC).timestamp()),
                )
            ]
        except Exception as exc:
            self.logger.warning(
                "haiku_fallback_failed ticker=%s error=%s", ticker, exc
            )
            return None