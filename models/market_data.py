from __future__ import annotations

from pydantic import BaseModel, Field


class OHLCVBar(BaseModel):
    """A single OHLCV bar for a ticker."""

    ticker: str = ""
    timestamp: str = Field(..., min_length=1)
    open: float = Field(..., ge=0)
    high: float = Field(..., ge=0)
    low: float = Field(..., ge=0)
    close: float = Field(..., ge=0)
    volume: float = Field(..., ge=0)


class MarketMetrics(BaseModel):
    """Derived market metrics for a ticker."""

    ticker: str = Field(..., min_length=1)
    current_price: float = 0.0
    return_5d: float = 0.0
    return_30d: float = 0.0
    return_90d: float = 0.0
    drawdown_30d: float = 0.0
    closes_above_20dma: int = 0
    distance_from_20dma: float = 0.0
    avg_volume_7d: float = 0.0
    avg_volume_30d: float = 0.0
    volume_spike_pct: float = 0.0
    atr_7d: float = 0.0
    realized_vol_7d: float = 0.0
    realized_vol_30d: float = 0.0
    rsi_14: float = 0.0
    bb_zscore: float = 0.0
    macd_histogram: float = 0.0
    market_cap: float = 0.0
    pe_ratio: float = 0.0
    dividend_yield: float = 0.0
    sector: str = ""
    industry: str = ""
    window_start: str = ""
    window_end: str = ""


class EquityQuote(BaseModel):
    """Latest quote snapshot for a ticker."""

    ticker: str = Field(..., min_length=1)
    price: float = Field(..., ge=0)
    bid: float = Field(..., ge=0)
    ask: float = Field(..., ge=0)
    volume: float = Field(..., ge=0)
    prev_close: float = Field(..., ge=0)
    change_pct: float = 0.0
