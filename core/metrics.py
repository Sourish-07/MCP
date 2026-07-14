from __future__ import annotations

import json
import logging
import math
from datetime import datetime, timedelta, timezone
from typing import Any

from models.market_data import MarketMetrics, OHLCVBar


class MetricsEngine:
    """Market metrics engine that extracts Robinhood-computed values directly
    via MCP calls rather than recomputing indicators locally. Pure-arithmetic
    fields (returns, drawdown, current_price, realized volatility) remain
    computed from OHLCV bars as before."""

    def __init__(self) -> None:
        self.logger = logging.getLogger("robinhood-agent.core.metrics")

    async def compute(
        self, ticker: str, bars: list[OHLCVBar], client: Any,
    ) -> MarketMetrics:
        """Async — pulls RSI, MACD, Bollinger, ATR, SMA, and fundamentals
        directly from Robinhood's MCP tools, keeping only return/drawdown
        calculations as local arithmetic on the bars parameter."""
        if not bars:
            return MarketMetrics(ticker=ticker, window_start="", window_end="")

        closes = [bar.close for bar in bars]
        latest = bars[-1]

        # ---------- local arithmetic (unchanged logic) ----------
        base_5d = closes[-6] if len(closes) >= 6 else closes[0]
        base_30d = closes[-31] if len(closes) >= 31 else closes[0]
        base_90d = closes[-91] if len(closes) >= 91 else closes[0]

        return_5d = ((latest.close - base_5d) / base_5d) * 100.0 if base_5d else 0.0
        return_30d = ((latest.close - base_30d) / base_30d) * 100.0 if base_30d else 0.0
        return_90d = ((latest.close - base_90d) / base_90d) * 100.0 if base_90d else 0.0

        window = closes[-30:] if len(closes) >= 30 else closes
        peak = max(window) if window else 0.0
        trough = min(window) if window else 0.0
        drawdown_30d = ((peak - trough) / peak) * 100.0 if peak > 0 else 0.0

        # Realized volatility — still local arithmetic on bars
        returns_7d = [
            bars[index].close / bars[index - 1].close - 1.0
            for index in range(max(1, len(bars) - 7), len(bars))
        ]
        returns_30d = [
            bars[index].close / bars[index - 1].close - 1.0
            for index in range(max(1, len(bars) - 30), len(bars))
        ]
        realized_vol_7d = MetricsEngine._std(returns_7d) * math.sqrt(252) * 100.0
        realized_vol_30d = MetricsEngine._std(returns_30d) * math.sqrt(252) * 100.0

        metrics = MarketMetrics(
            ticker=ticker,
            current_price=latest.close,
            return_5d=return_5d,
            return_30d=return_30d,
            return_90d=return_90d,
            drawdown_30d=drawdown_30d,
            realized_vol_7d=realized_vol_7d,
            realized_vol_30d=realized_vol_30d,
            window_start=bars[0].timestamp,
            window_end=latest.timestamp,
        )

        # ---------- direct Robinhood extraction ----------
        end_time = datetime.now(timezone.utc)
        start_time_90d = (end_time - timedelta(days=90)).strftime("%Y-%m-%dT%H:%M:%SZ")
        end_time_str = end_time.strftime("%Y-%m-%dT%H:%M:%SZ")

        # RSI — direct from Robinhood
        rsi_result = await client.get_technical_indicator(
            ticker, "rsi", "day", start_time_90d, end_time_str,
            output="latest", period=14,
        )
        metrics.rsi_14 = self._extract_latest_indicator_value(rsi_result, "rsi")

        # MACD histogram — direct from Robinhood
        macd_result = await client.get_technical_indicator(
            ticker, "macd", "day", start_time_90d, end_time_str,
            output="latest",
        )
        metrics.macd_histogram = self._extract_macd_histogram(macd_result)

        # Bollinger — direct bands, z-score is (price - middle) / sigma
        bb_result = await client.get_technical_indicator(
            ticker, "bollinger_bands", "day", start_time_90d, end_time_str,
            output="latest", num_std=2,
        )
        metrics.bb_zscore = self._extract_bollinger_zscore(bb_result, metrics.current_price)

        # ATR — direct from Robinhood
        atr_result = await client.get_technical_indicator(
            ticker, "atr", "day", start_time_90d, end_time_str,
            output="latest", period=7,
        )
        metrics.atr_7d = self._extract_latest_indicator_value(atr_result, "atr")

        # 20-day SMA — distance is (price - sma) / sma (unit conversion)
        sma_result = await client.get_technical_indicator(
            ticker, "sma", "day", start_time_90d, end_time_str,
            output="latest", period=20,
        )
        sma_20 = self._extract_latest_indicator_value(sma_result, "sma")
        metrics.distance_from_20dma = (
            ((metrics.current_price - sma_20) / sma_20) * 100.0 if sma_20 else 0.0
        )

        # closes_above_20dma — needs the SMA series aligned with real closes
        sma_series_result = await client.get_technical_indicator(
            ticker, "sma", "day", start_time_90d, end_time_str,
            output="series", period=20,
        )
        metrics.closes_above_20dma = self._count_closes_above_series(bars, sma_series_result)

        # Fundamentals — volume averages and new fields direct from Robinhood
        fundamentals = await client.get_equity_fundamentals([ticker])
        fdata = fundamentals.get(ticker, {})
        metrics.avg_volume_7d = float(fdata.get("average_volume_2_weeks") or 0)
        metrics.avg_volume_30d = float(fdata.get("average_volume_30_days") or 0)
        if metrics.avg_volume_30d > 0 and bars:
            metrics.volume_spike_pct = (
                (bars[-1].volume - metrics.avg_volume_30d) / metrics.avg_volume_30d
            ) * 100.0
        metrics.market_cap = float(fdata.get("market_cap") or 0)
        metrics.pe_ratio = float(fdata.get("pe_ratio") or 0)
        metrics.dividend_yield = float(fdata.get("dividend_yield") or 0)
        metrics.sector = fdata.get("sector", "")
        metrics.industry = fdata.get("industry", "")

        return metrics

    # ------------------------------------------------------------------ #
    #  Helper methods — defensive extraction with fallback chains
    # ------------------------------------------------------------------ #

    def _extract_latest_indicator_value(self, result: dict, key_hint: str) -> float:
        try:
            indicators = result.get("data", {}).get("indicators", [])
            if not indicators:
                self.logger.warning("indicator_extract_no_indicators raw_data=%s", result)
                return 0.0
            series = indicators[0].get("series", [])
            if not series:
                self.logger.warning("indicator_extract_empty_series raw_data=%s", result)
                return 0.0
            latest_point = series[-1]
            value = latest_point.get("value")
            if value is None:
                self.logger.warning(
                    "indicator_extract_no_value key_hint=%s point=%s", key_hint, latest_point
                )
                return 0.0
            return float(value)
        except Exception as exc:
            self.logger.warning("indicator_extract_exception key_hint=%s error=%s raw_data=%s", key_hint, exc, result)
            return 0.0

    def _extract_macd_histogram(self, result: dict) -> float:
        try:
            indicators = result.get("data", {}).get("indicators", [])
            if not indicators:
                self.logger.warning("macd_extract_no_indicators raw_data=%s", result)
                return 0.0
            series = indicators[0].get("series", [])
            if not series:
                self.logger.warning("macd_extract_empty_series raw_data=%s", result)
                return 0.0
            latest_point = series[-1]
            if "histogram" in latest_point:
                return float(latest_point["histogram"])
            if "macd" in latest_point and "signal" in latest_point:
                return float(latest_point["macd"]) - float(latest_point["signal"])
            self.logger.warning("macd_extract_no_match point=%s", latest_point)
            return 0.0
        except Exception as exc:
            self.logger.warning("macd_extract_exception error=%s raw_data=%s", exc, result)
            return 0.0

    def _extract_bollinger_zscore(self, result: dict, current_price: float) -> float:
        try:
            indicators = result.get("data", {}).get("indicators", [])
            if not indicators:
                self.logger.warning("bollinger_extract_no_indicators raw_data=%s", result)
                return 0.0
            series = indicators[0].get("series", [])
            if not series:
                self.logger.warning("bollinger_extract_empty_series raw_data=%s", result)
                return 0.0
            latest_point = series[-1]
            upper = latest_point.get("upper")
            middle = latest_point.get("middle")
            if upper is None or middle is None:
                self.logger.warning("bollinger_extract_no_match point=%s", latest_point)
                return 0.0
            sigma = (float(upper) - float(middle)) / 2.0
            if sigma == 0:
                return 0.0
            return (current_price - float(middle)) / sigma
        except Exception as exc:
            self.logger.warning("bollinger_extract_exception error=%s raw_data=%s", exc, result)
            return 0.0

    def _count_closes_above_series(self, bars: list, sma_series_result: dict) -> int:
        try:
            indicators = sma_series_result.get("data", {}).get("indicators", [])
            if not indicators:
                self.logger.warning("sma_series_extract_no_indicators raw_data=%s", sma_series_result)
                return 0
            series = indicators[0].get("series", [])
            if not series:
                self.logger.warning("sma_series_extract_empty raw_data=%s", sma_series_result)
                return 0

            sma_by_date = {}
            for point in series:
                date_str = point.get("begins_at", "")[:10]
                value = point.get("value")
                if date_str and value is not None:
                    sma_by_date[date_str] = float(value)

            count = 0
            for bar in bars[-20:]:
                bar_date = bar.timestamp[:10]
                sma_value = sma_by_date.get(bar_date)
                if sma_value is not None and bar.close > sma_value:
                    count += 1
            return count
        except Exception as exc:
            self.logger.warning("sma_series_extract_exception error=%s raw_data=%s", exc, sma_series_result)
            return 0

    # ------------------------------------------------------------------ #
    #  Static helper — kept from original for realized volatility
    # ------------------------------------------------------------------ #

    @staticmethod
    def _std(values: list[float]) -> float:
        """Compute sample standard deviation for a list of returns."""
        if len(values) < 2:
            return 0.0
        mean = sum(values) / len(values)
        variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
        return variance ** 0.5