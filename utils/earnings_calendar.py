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

    def get_upcoming(self, ticker: str, within_days: int = 5, after_days: int = 3) -> dict | None:
        """Return earnings entry if ticker has earnings within the configured window, else None.

        Window covers `within_days` days BEFORE the report through `after_days`
        days AFTER the report (inclusive). This makes the earnings_flag true for
        the full catalyst window, not only the run-up.
        """
        try:
            cache = json.loads(self.cache_path.read_text(encoding="utf-8"))
            today = datetime.now(timezone.utc).date()
            for entry in cache.get("entries", []):
                if entry.get("ticker", "").upper() != ticker.upper():
                    continue
                try:
                    ev_date = date.fromisoformat(entry["date"])
                    days_away = (ev_date - today).days
                    # treat as in-window from `after_days` after the report (negative days_away)
                    # back through `within_days` before it (positive days_away).
                    if -after_days <= days_away <= within_days:
                        return {**entry, "days_away": days_away}
                except (ValueError, KeyError):
                    continue
        except Exception:
            pass
        return None

    def is_in_earnings_window(self, ticker: str, within_days: int = 5, after_days: int = 3) -> bool:
        """Cheap boolean check: is this ticker inside the earnings window right now?

        INFORMATIONAL ONLY — never used as a hard BUY/SELL gate.
        """
        return self.get_upcoming(ticker, within_days=within_days, after_days=after_days) is not None

    def earnings_window_for_prompt(self, ticker: str) -> str:
        """Return a short opportunity-review block when the ticker is in the
        earnings window. INFORMATIONAL — never a ban. ~80 tokens.

        Returns "" when the ticker is outside the window.
        """
        near = self.get_upcoming(ticker, within_days=5, after_days=3)
        if not near:
            return ""
        date_str = near.get("date", "?")
        time_str = near.get("time", "?")
        days_away = near.get("days_away", 0)
        eps_estimate = near.get("eps_estimate")
        iff_eps = f" EPS est {eps_estimate:.2f}." if eps_estimate else ""
        if days_away >= 0:
            head = f"EARNINGS WINDOW: next_earnings={date_str} ({time_str}, {days_away}d away)."
        else:
            head = f"EARNINGS WINDOW: reported {date_str} ({abs(days_away)}d ago, {time_str})."
        return (
            head + iff_eps + " Evaluate as a dated catalyst, NOT a ban.\n"
            "- Decide: BUY into / HOLD through if setup + expected move is favorable with edge, OR IGNORE/SELL if risk/reward is poor.\n"
            "- State in reasoning whether earnings is a positive catalyst, a risk, or neutral.\n"
            "- Do NOT reject solely because earnings are near."
        )

    def earnings_warning_for_prompt(self, ticker: str) -> str:
        """Backward-compatible alias for earnings_window_for_prompt.

        Kept so any legacy call sites still emit the (now informational) block.
        """
        return self.earnings_window_for_prompt(ticker)
