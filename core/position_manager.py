from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import List

from models.decisions import TradeDecision
from models.portfolio import Portfolio, PositionRecord
from models.market_data import EquityQuote


@dataclass
class ExitSignal:
    """Signal to exit an open position."""

    ticker: str
    reason: str
    current_price: float
    unrealized_pct: float
    quantity: float


class PositionManager:
    """Manage position entry/exit persistence and exit checks."""

    REVIEW_GAIN_PCT = 0.15  # 15% gain triggers a soft review

    def __init__(self, path: str | None = None) -> None:
        self.logger = logging.getLogger("robinhood-agent.core.position_manager")
        self._path = Path(path) if path else Path(__file__).resolve().parent.parent / "logs" / "positions.json"
        self._path.parent.mkdir(exist_ok=True)
        if not self._path.exists():
            self._path.write_text("[]", encoding="utf-8")

    @staticmethod
    def _vol_adjusted_stop_loss(realized_vol_30d: float) -> float:
        """Widen the stop-loss for higher-volatility names instead of a
        flat -7% that gets triggered by routine noise on growth stocks.
        Bands, based on 30-day annualized realized volatility:
          < 30%  -> -7%  (calm names, tight stop is appropriate)
          30-60% -> -10% (moderate vol, e.g. AVGO/NEM today)
          60-90% -> -13% (high vol, e.g. AVGO at 66%)
          > 90%  -> -16% (extreme vol, e.g. MU historically)
        """
        if realized_vol_30d < 30.0:
            return -0.07
        if realized_vol_30d < 60.0:
            return -0.10
        if realized_vol_30d < 90.0:
            return -0.13
        return -0.16

    def record_entry(self, result: TradeDecision | object, decision: TradeDecision, realized_vol_30d: float = 0.0) -> None:
        """Persist a position entry record for executed BUY/ROTATE orders."""
        # This method expects an ExecutionResult-like object for result
        try:
            if decision.decision not in (decision.decision.__class__.BUY, decision.decision.__class__.ROTATE):
                return
        except Exception:
            pass

        try:
            entry_price = float(getattr(result, "fill_price", 0.0))
            quantity = float(getattr(result, "quantity", 0.0))
        except Exception:
            entry_price = 0.0
            quantity = 0.0

        record = PositionRecord(
            ticker=decision.ticker,
            entry_price=entry_price,
            entry_date=datetime.now(timezone.utc).date().isoformat(),
            entry_edge_score=decision.edge_score if hasattr(decision, "edge_score") else 0.0,
            stop_loss_pct=self._vol_adjusted_stop_loss(realized_vol_30d),
            take_profit_pct=0.20,
            quantity=quantity,
            order_id=getattr(result, "order_id", ""),
        )
        entries = self.get_open_records()
        entries = [item for item in entries if item.ticker != record.ticker]
        entries.append(record)
        try:
            self._path.write_text(json.dumps([item.model_dump(mode="json") for item in entries], indent=2), encoding="utf-8")
        except Exception as exc:
            self.logger.warning("position_write_failed: %s", exc)

    def record_exit(self, ticker: str) -> None:
        """Remove a ticker from the open-position log."""
        entries = [item for item in self.get_open_records() if item.ticker != ticker]
        try:
            self._path.write_text(json.dumps([item.model_dump(mode="json") for item in entries], indent=2), encoding="utf-8")
        except Exception as exc:
            self.logger.warning("position_write_failed: %s", exc)

    def get_open_records(self) -> List[PositionRecord]:
        """Return all persisted open positions."""
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
            return [PositionRecord.model_validate(item) for item in data]
        except Exception as exc:
            self.logger.warning("position_load_failed: %s", exc)
            return []

    def check_exits(self, portfolio: Portfolio, quotes: dict[str, EquityQuote]) -> List[ExitSignal]:
        """Check open records against current price and holding rules.

        Emits signals with reason types:
          - STOP_LOSS: forced exit (hard rule)
          - PROFIT_REVIEW: advisory review when gain exceeds REVIEW_GAIN_PCT
          - MAX_HOLDING_PERIOD: advisory when days held > configured max
        """
        signals: List[ExitSignal] = []
        for record in self.get_open_records():
            position = next((item for item in portfolio.positions if item.ticker == record.ticker), None)
            if position is None:
                continue
            current_price = quotes.get(record.ticker).price if record.ticker in quotes else position.current_price
            if current_price <= 0:
                continue
            unrealized_pct = (current_price - record.entry_price) / record.entry_price if record.entry_price else 0.0

            # Hard stop-loss exit
            if unrealized_pct <= record.stop_loss_pct:
                signals.append(ExitSignal(record.ticker, "STOP_LOSS", current_price, unrealized_pct, position.quantity))
                continue

            # Soft profit review (advisory only) and max holding period advisory.
            profit_review = unrealized_pct >= self.REVIEW_GAIN_PCT
            try:
                days_held = (datetime.now(timezone.utc).date() - datetime.fromisoformat(record.entry_date).date()).days
            except Exception:
                days_held = 0
            max_holding = days_held > 15

            # Combine signals: if both conditions apply, emit a single combined signal.
            if profit_review and max_holding:
                signals.append(ExitSignal(record.ticker, "PROFIT_REVIEW+MAX_HOLDING_PERIOD", current_price, unrealized_pct, position.quantity))
            elif profit_review:
                signals.append(ExitSignal(record.ticker, "PROFIT_REVIEW", current_price, unrealized_pct, position.quantity))
            elif max_holding:
                signals.append(ExitSignal(record.ticker, "MAX_HOLDING_PERIOD", current_price, unrealized_pct, position.quantity))

        return signals
