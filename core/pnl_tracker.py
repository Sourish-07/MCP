from __future__ import annotations

import logging
from datetime import datetime, timezone


class PnLTracker:
    """Pull real realized P&L and trade history directly from Robinhood MCP,
    replacing any locally-estimated performance tracking."""

    def __init__(self, client) -> None:
        self.client = client
        self.logger = logging.getLogger("robinhood-agent.pnl")

    async def build_report(self) -> dict:
        """Return a dict combining realized P&L and trade history from Robinhood.

        Each section is independently try/excepted so partial failures are
        safe — the caller gets what it can.
        """
        report = {"generated_at": datetime.now(timezone.utc).isoformat()}

        try:
            realized = await self.client.get_realized_pnl(span="month")
            report["realized_pnl_month"] = realized.get("data", {})
        except Exception as exc:
            self.logger.warning("realized_pnl_fetch_failed: %s", exc)
            report["realized_pnl_month"] = {}

        try:
            trades = await self.client.get_pnl_trade_history(span="month")
            report["trade_history_month"] = trades.get("data", {}).get("trades", [])
        except Exception as exc:
            self.logger.warning("trade_history_fetch_failed: %s", exc)
            report["trade_history_month"] = []

        return report