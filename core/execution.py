from __future__ import annotations

import json
import logging
from datetime import datetime, timezone, time
import pytz
from pathlib import Path

from robinhood_mcp.robinhood_client import RobinhoodMCPClient
from models.decisions import DecisionType, ExecutionResult, TradeDecision, StructuredEdgeScore
from models.portfolio import Portfolio
from core.position_manager import PositionManager


class ExecutionEngine:
    """Execution engine with dry-run enforcement and logging."""

    EDGE_THRESHOLD = 3.5

    def __init__(self, client: RobinhoodMCPClient, position_manager: PositionManager, settings: dict) -> None:
        self.client = client
        self.position_manager = position_manager
        self.settings = settings
        self.logger = logging.getLogger("robinhood-agent.core.execution")
        self._trades_path = Path(__file__).resolve().parent.parent / "logs" / "trades.json"
        self._trades_path.parent.mkdir(exist_ok=True)
        if not self._trades_path.exists():
            self._trades_path.write_text("[]", encoding="utf-8")

    @staticmethod
    def _size_from_edge(total: float) -> float:
        """Map computed edge total into a position size percentage.

        Ranges:
          3.5 <= total < 4.0 -> 4%
          4.0 <= total < 4.5 -> 6%
          4.5 <= total       -> 8%
        """
        try:
            t = float(total or 0.0)
        except Exception:
            return 0.0
        if t >= 4.5:
            return 0.08
        if t >= 4.0:
            return 0.06
        if t >= 3.5:
            return 0.04
        return 0.0

    def _is_safe_execution_window(self) -> bool:
        """Return True when current NY time is safe for execution (not in blocked windows).

        Blocked windows: 09:30-09:34 (inclusive) and 15:50-15:59 (inclusive) ET.
        """
        try:
            tz_name = self.settings.get("schedule", {}).get("timezone", "America/New_York") if isinstance(self.settings, dict) else "America/New_York"
            ny_tz = pytz.timezone(tz_name)
            now = datetime.now(ny_tz)
            if now.weekday() >= 5:
                return True
            t = now.time()
            if (t >= time(9, 30) and t < time(9, 35)) or (t >= time(15, 50) and t < time(16, 0)):
                return False
        except Exception:
            return True
        return True

    async def execute(self, decision: TradeDecision, portfolio: Portfolio, current_price: float, dry_run: bool, vol_30d: float = 0.0, avg_volume_30d: float = 0.0, realized_vol_30d: float = 0.0) -> ExecutionResult:
        """Execute or simulate a decision after validating tradeability and risk limits."""
        effective_dry_run = bool(self.settings.get("env", {}).get("DRY_RUN", True)) or bool(dry_run)
        # Block execution during known unsafe time windows
        if not self._is_safe_execution_window():
            result = ExecutionResult(ticker=decision.ticker, status="SKIPPED_TIME_WINDOW", dry_run=effective_dry_run, timestamp=datetime.now(timezone.utc).isoformat())
            self._log_trade(decision, result, 0.0)
            return result

        if decision.decision in (DecisionType.IGNORE, DecisionType.HOLD):
            result = ExecutionResult(ticker=decision.ticker, status="SKIPPED", dry_run=True, timestamp=datetime.now(timezone.utc).isoformat())
            self._log_trade(decision, result, 0.0)
            return result
        # determine numeric edge total (structured or legacy)
        try:
            edge_total = float(getattr(decision, "edge", StructuredEdgeScore()).total)
        except Exception:
            edge_total = float(getattr(decision, "edge_score", 0.0) or 0.0)

        self.logger.info(
            "EXECUTION_START ticker=%s decision=%s edge_total=%.4f dry_run=%s",
            decision.ticker, decision.decision.value, edge_total, effective_dry_run
        )
        self.logger.info(
            "EXECUTION_GATECHECK ticker=%s raw_decision=%s edge_total=%.4f threshold=%.1f",
            decision.ticker, decision.decision.value, edge_total, self.EDGE_THRESHOLD
        )

        if edge_total < self.EDGE_THRESHOLD:
            result = ExecutionResult(ticker=decision.ticker, status="REJECTED_LOW_EDGE", dry_run=effective_dry_run, timestamp=datetime.now(timezone.utc).isoformat())
            self._log_trade(decision, result, 0.0)
            return result

        if decision.decision in (DecisionType.BUY, DecisionType.ROTATE):
            if vol_30d > 60.0:
                self.logger.info(
                    "execution_rejected ticker=%s reason=HIGH_VOLATILITY vol_30d=%.1f edge=%.4f",
                    decision.ticker, vol_30d, edge_total)
                result = ExecutionResult(
                    ticker=decision.ticker,
                    status="REJECTED_HIGH_VOLATILITY",
                    dry_run=effective_dry_run,
                    timestamp=datetime.now(timezone.utc).isoformat())
                self._log_trade(decision, result, 0.0)
                return result

        if decision.decision in (DecisionType.BUY, DecisionType.ROTATE):
            if avg_volume_30d > 0 and current_price > 0:
                dollar_volume = avg_volume_30d * current_price
                if dollar_volume < 75_000_000:
                    self.logger.info(
                        "execution_rejected ticker=%s reason=LOW_LIQUIDITY dollar_vol=%.0f edge=%.4f",
                        decision.ticker, dollar_volume, edge_total)
                    result = ExecutionResult(
                        ticker=decision.ticker,
                        status="REJECTED_LOW_LIQUIDITY",
                        dry_run=effective_dry_run,
                        timestamp=datetime.now(timezone.utc).isoformat())
                    self._log_trade(decision, result, 0.0)
                    return result

        try:
            tradability = await self.client.get_equity_tradability(decision.ticker)
            self.logger.info("TRADABILITY_RESPONSE ticker=%s full=%s", decision.ticker, tradability)
            
            # Robust parsing for actual MCP structure
            tradable = False
            if isinstance(tradability, dict):
                results = tradability.get("data", {}).get("results", [])
                if results and isinstance(results, list):
                    item = results[0] if results else {}
                    tradable = bool(item.get("tradeable")) or bool(item.get("tradable"))
                else:
                    # fallback top-level checks
                    tradable = bool(tradability.get("tradeable")) or bool(tradability.get("tradable"))
            
            self.logger.info("tradability_check ticker=%s tradable=%s", decision.ticker, tradable)
            
            if not tradable:
                return ExecutionResult(
                    ticker=decision.ticker, 
                    status="REJECTED_NOT_TRADABLE", 
                    dry_run=effective_dry_run, 
                    timestamp=datetime.now(timezone.utc).isoformat()
                )
        except Exception as exc:
            self.logger.error("tradability_check_failed ticker=%s error=%s", decision.ticker, exc)
            # Allow trade in OPEN cycle when data is available but check fails
            self.logger.warning("Proceeding with trade despite tradability check error in OPEN cycle")

        if decision.decision == DecisionType.SELL:
            record = next((r for r in self.position_manager.get_open_records() if r.ticker == decision.ticker), None)
            quantity = record.quantity if record and record.quantity > 0 else 0.0
            quantity = round(quantity, 6)
            if quantity <= 0:
                result = ExecutionResult(
                    ticker=decision.ticker,
                    status="REJECTED_NO_POSITION",
                    dry_run=effective_dry_run,
                    timestamp=datetime.now(timezone.utc).isoformat(),
                )
                self._log_trade(decision, result, 0.0)
                return result
            side = "sell"
        else:
            # compute position size from edge total
            position_size_pct = self._size_from_edge(edge_total)
            position_size_pct = min(position_size_pct, float(self.settings.get("trading", {}).get("max_position_pct", 0.08)))
            if position_size_pct <= 0:
                self.logger.info(
                    "execution_rejected ticker=%s reason=REJECTED_ZERO_SIZE edge=%.4f position_size_pct=%.4f",
                    decision.ticker, edge_total, position_size_pct)
                result = ExecutionResult(
                    ticker=decision.ticker,
                    status="REJECTED_ZERO_SIZE",
                    dry_run=effective_dry_run,
                    timestamp=datetime.now(timezone.utc).isoformat(),
                )
                self._log_trade(decision, result, 0.0)
                return result

            dollar_amount = position_size_pct * float(portfolio.total_value)
            quantity = dollar_amount / current_price if current_price > 0 else 0.0
            quantity = round(quantity, 6)
            available_qty = (portfolio.buying_power / current_price) if current_price > 0 else 0.0
            quantity = min(quantity, available_qty * 0.99) if available_qty > 0 else 0.0

            if quantity <= 0:
                self.logger.info(
                    "execution_rejected ticker=%s reason=REJECTED_INSUFFICIENT_CAPITAL edge=%.4f quantity=%.6f buying_power=%.2f",
                    decision.ticker, edge_total, quantity, portfolio.buying_power)
                result = ExecutionResult(
                    ticker=decision.ticker,
                    status="REJECTED_INSUFFICIENT_CAPITAL",
                    dry_run=effective_dry_run,
                    timestamp=datetime.now(timezone.utc).isoformat(),
                )
                self._log_trade(decision, result, 0.0)
                return result
            side = "buy" if decision.decision in (DecisionType.BUY, DecisionType.ROTATE) else "sell"

        # compute actual position size percent for this attempted trade (quantity * price / portfolio)
        try:
            position_size_pct_local = (quantity * current_price) / float(portfolio.total_value) if float(portfolio.total_value) > 0 else 0.0
        except Exception:
            position_size_pct_local = 0.0

        if effective_dry_run:
            result = ExecutionResult(
                ticker=decision.ticker,
                order_id="DRY_RUN",
                status="WOULD_EXECUTE",
                fill_price=current_price,
                quantity=quantity,
                timestamp=datetime.now(timezone.utc).isoformat(),
                dry_run=effective_dry_run,
                side=side,
            )
            self._log_trade(decision, result, position_size_pct_local)
            if decision.decision in (DecisionType.BUY, DecisionType.ROTATE):
                self.position_manager.record_entry(result, decision, realized_vol_30d)
            if decision.decision in (DecisionType.SELL, DecisionType.ROTATE):
                exit_ticker = decision.rotate_from_ticker if decision.decision == DecisionType.ROTATE else decision.ticker
                self.position_manager.record_exit(exit_ticker)
            return result

        try:
            preview = await self.client.review_equity_order(decision.ticker, side, quantity, "market")
            warnings = str(preview.get("warnings", "") or "").lower()
            if "insufficient" in warnings or "not allowed" in warnings:
                result = ExecutionResult(ticker=decision.ticker, status="REJECTED", dry_run=effective_dry_run, timestamp=datetime.now(timezone.utc).isoformat(), side=side)
                self._log_trade(decision, result, position_size_pct_local)
                return result
            placed = await self.client.place_equity_order(decision.ticker, side, quantity, "market", dry_run=False)
            result = ExecutionResult(
                ticker=decision.ticker,
                order_id=getattr(placed, "order_id", ""),
                status="EXECUTED" if getattr(placed, "status", "") else "SUBMITTED",
                fill_price=getattr(placed, "fill_price", current_price),
                quantity=quantity,
                timestamp=datetime.now(timezone.utc).isoformat(),
                dry_run=False,
                side=side,
            )
            self._log_trade(decision, result, position_size_pct_local)
            if result.status == "EXECUTED":
                if decision.decision in (DecisionType.BUY, DecisionType.ROTATE):
                    self.position_manager.record_entry(result, decision, realized_vol_30d)
                if decision.decision in (DecisionType.SELL, DecisionType.ROTATE):
                    exit_ticker = decision.rotate_from_ticker if decision.decision == DecisionType.ROTATE else decision.ticker
                    self.position_manager.record_exit(exit_ticker)
            return result
        except Exception as exc:
            self.logger.warning("execution_failed ticker=%s error=%s", decision.ticker, exc)
            return ExecutionResult(ticker=decision.ticker, status="REJECTED_EXCEPTION", dry_run=effective_dry_run, timestamp=datetime.now(timezone.utc).isoformat(), side=side)

    def _log_trade(self, decision: TradeDecision, result: ExecutionResult, position_size_pct: float) -> None:
        """Append a trade result to logs/trades.json."""
        try:
            existing = json.loads(self._trades_path.read_text(encoding="utf-8"))
            existing.append({
                "timestamp": result.timestamp,
                "ticker": decision.ticker,
                "decision": decision.decision.value,
                "edge_score": decision.edge_score,
                "position_size_pct": position_size_pct,
                "action_type": decision.action_type,
                "status": result.status,
                "fill_price": result.fill_price,
                "quantity": result.quantity,
                "dry_run": result.dry_run,
                "reasoning_summary": decision.reasoning_summary,
            })
            self._trades_path.write_text(json.dumps(existing, indent=2), encoding="utf-8")
        except Exception as exc:
            self.logger.warning("trade_logging_failed: %s", exc)
