import json
import logging
from datetime import datetime, timezone, date
from pathlib import Path


class EarningsCalendar:
    """Fetches earnings dates from the Robinhood MCP get_earnings_calendar tool."""

    def __init__(self) -> None:
        self.cache_path = (
            Path(__file__).resolve().parent.parent / "data" / "earnings_calendar.json"
        )
        self.cache_path.parent.mkdir(exist_ok=True)
        self.mcp_client = None
        self.logger = logging.getLogger("robinhood-agent.earnings")

    def set_client(self, client) -> None:
        """Wire in the RobinhoodMCPClient after construction."""
        self.mcp_client = client

    async def refresh(self, tickers: list[str]) -> None:
        """Fetch upcoming earnings from Robinhood MCP and cache to disk."""
        if not self.mcp_client:
            self.logger.warning("earnings_calendar no mcp_client set, skipping refresh")
            return
        try:
            today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            parsed = await self.mcp_client.get_earnings_calendar_window(
                start_date=today, days=14
            )
            events = parsed.get("data", {}).get("results", [])
            watchlist_upper = {t.upper() for t in tickers}
            entries = []
            for ev in events:
                symbol = (ev.get("symbol") or "").upper()
                if symbol not in watchlist_upper:
                    continue
                report = ev.get("report", {})
                eps = ev.get("eps", {})
                entries.append({
                    "ticker": symbol,
                    "date": report.get("date", ""),
                    "time": "AMC" if report.get("timing") == "pm" else "BMO",
                    "verified": report.get("verified", False),
                    "eps_estimate": eps.get("estimate"),
                    "eps_actual": eps.get("actual"),
                })
            cache = {
                "refreshed_at": datetime.now(timezone.utc).isoformat(),
                "source": "robinhood_mcp_direct",
                "entries": entries,
            }
            self.cache_path.write_text(json.dumps(cache, indent=2), encoding="utf-8")
            self.logger.info("earnings_calendar refreshed count=%d", len(entries))
        except Exception as exc:
            self.logger.warning("earnings_calendar_refresh_failed: %s", exc)

    def get_upcoming(self, ticker: str, within_days: int = 2) -> dict | None:
        """Return earnings entry if ticker has earnings within within_days, else None."""
        try:
            cache = json.loads(self.cache_path.read_text(encoding="utf-8"))
            today = datetime.now(timezone.utc).date()
            for entry in cache.get("entries", []):
                if entry.get("ticker", "").upper() != ticker.upper():
                    continue
                try:
                    ev_date = date.fromisoformat(entry["date"])
                    days_away = (ev_date - today).days
                    if 0 <= days_away <= within_days:
                        return {**entry, "days_away": days_away}
                except (ValueError, KeyError):
                    continue
        except Exception:
            pass
        return None

    def earnings_warning_for_prompt(self, ticker: str) -> str:
        """Return a warning string to inject into Claude prompt if earnings imminent."""
        near = self.get_upcoming(ticker, within_days=2)
        if near:
            eps_estimate = near.get("eps_estimate")
            verified = near.get("verified", False)
            date_str = near.get("date", "?")
            time_str = near.get("time", "?")
            eps_str = f"{eps_estimate:.2f}" if eps_estimate else "not yet available"
            confirmed = "Confirmed date." if verified else "Date not yet confirmed — treat as tentative."
            return (
                f"EARNINGS WARNING: {ticker} reports in {near['days_away']} day(s) "
                f"({date_str}, {time_str}). "
                f"EPS estimate: {eps_str}. "
                f"{confirmed} "
                f"New BUY positions are BLOCKED. Review existing positions carefully."
            )
        far = self.get_upcoming(ticker, within_days=5)
        if far:
            eps_estimate = far.get("eps_estimate")
            eps_str = f"{eps_estimate:.2f}" if eps_estimate else "not yet available"
            return (
                f"Note: {ticker} earnings in {far['days_away']} day(s) "
                f"({far.get('date', '?')}, {far.get('time', '?')}). "
                f"EPS estimate: {eps_str}. "
                f"Factor into thesis."
            )
        return ""
