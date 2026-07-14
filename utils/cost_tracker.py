from __future__ import annotations

import json
import logging
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import DefaultDict

from pydantic import BaseModel


class CostLimitExceededError(RuntimeError):
    """Raised when cost limits are exceeded."""


class CostRecord(BaseModel):
    """Simple record of per-call cost data."""

    model: str
    input_tokens: int
    cache_creation_tokens: int = 0
    cache_read_tokens: int = 0
    output_tokens: int
    estimated_usd: float
    timestamp: str


class CostTracker:
    """Anthropic-only cost tracker for model usage and budget guardrails."""

    _instance: "CostTracker | None" = None

    # Base rates (per token, already divided by 1_000_000)
    HAIKU_BASE_INPUT = 1.00 / 1_000_000
    HAIKU_CACHE_WRITE_5M = 1.25 / 1_000_000
    HAIKU_CACHE_READ = 0.10 / 1_000_000
    HAIKU_OUTPUT = 5.00 / 1_000_000

    SONNET_BASE_INPUT = 3.00 / 1_000_000
    SONNET_CACHE_WRITE_5M = 3.75 / 1_000_000
    SONNET_CACHE_READ = 0.30 / 1_000_000
    SONNET_OUTPUT = 15.00 / 1_000_000

    def __new__(cls) -> "CostTracker":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.logger = logging.getLogger("robinhood-agent.cost")
            cls._instance.records: list[CostRecord] = []
            cls._instance._daily_totals: DefaultDict[str, float] = defaultdict(float)
            cls._instance._monthly_totals: DefaultDict[str, float] = defaultdict(float)
            cls._instance._load()
        return cls._instance

    def _storage_path(self) -> Path:
        """Return the dedicated cost tracker persistence path."""
        return Path(__file__).resolve().parent.parent / "logs" / "cost_tracker.json"

    def _load(self) -> None:
        """Restore previously persisted cost totals and records."""
        path = self._storage_path()
        if not path.exists():
            return

        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            self.records = [CostRecord.model_validate(item) for item in payload.get("records", [])]
            self._daily_totals = defaultdict(float, {key: float(value) for key, value in payload.get("daily_totals", {}).items()})
            self._monthly_totals = defaultdict(float, {key: float(value) for key, value in payload.get("monthly_totals", {}).items()})
        except Exception as exc:
            self.logger.warning("cost_tracker_load_failed: %s", exc)
            self.records = []
            self._daily_totals = defaultdict(float)
            self._monthly_totals = defaultdict(float)

    def record(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
        cache_creation_tokens: int = 0,
        cache_read_tokens: int = 0,
    ) -> float:
        """Record token usage and return the estimated USD cost.

        input_tokens corresponds to base-rate (non-cached) input tokens.
        cache_creation_tokens are billed at the cache-write rate.
        cache_read_tokens are billed at the cache-read rate.
        """
        base_in, cache_write, cache_read, out_rate = self._pricing_for(model)

        # Regular (non-cached) input tokens = total input minus cache_creation minus cache_read
        regular_input = max(0, input_tokens - cache_creation_tokens - cache_read_tokens)

        estimated_usd = (
            regular_input * base_in
            + cache_creation_tokens * cache_write
            + cache_read_tokens * cache_read
            + output_tokens * out_rate
        )

        record = CostRecord(
            model=model,
            input_tokens=input_tokens,
            cache_creation_tokens=cache_creation_tokens,
            cache_read_tokens=cache_read_tokens,
            output_tokens=output_tokens,
            estimated_usd=estimated_usd,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )
        self.records.append(record)

        today = datetime.now(timezone.utc).date().isoformat()
        month = datetime.now(timezone.utc).strftime("%Y-%m")
        self._daily_totals[today] += estimated_usd
        self._monthly_totals[month] += estimated_usd

        self.logger.info(
            "cost_record model=%s input=%d cache_create=%d cache_read=%d output=%d usd=%.6f",
            model,
            input_tokens,
            cache_creation_tokens,
            cache_read_tokens,
            output_tokens,
            estimated_usd,
        )
        self._persist()
        return estimated_usd

    def _pricing_for(self, model: str) -> tuple[float, float, float, float]:
        """Return (base_input, cache_write, cache_read, output) rates for the model."""
        model_name = model.lower()
        if "haiku" in model_name:
            return self.HAIKU_BASE_INPUT, self.HAIKU_CACHE_WRITE_5M, self.HAIKU_CACHE_READ, self.HAIKU_OUTPUT
        if "sonnet" in model_name:
            return self.SONNET_BASE_INPUT, self.SONNET_CACHE_WRITE_5M, self.SONNET_CACHE_READ, self.SONNET_OUTPUT
        # Default to Sonnet pricing for unknown models
        return self.SONNET_BASE_INPUT, self.SONNET_CACHE_WRITE_5M, self.SONNET_CACHE_READ, self.SONNET_OUTPUT

    def daily_total(self) -> float:
        """Return the current daily estimated spend in USD."""
        today = datetime.now(timezone.utc).date().isoformat()
        return self._daily_totals.get(today, 0.0)

    def monthly_total(self) -> float:
        """Return the current monthly estimated spend in USD."""
        month = datetime.now(timezone.utc).strftime("%Y-%m")
        return self._monthly_totals.get(month, 0.0)

    def check_limits(self, daily: float, monthly: float) -> bool:
        """Check configured budget limits and raise warnings/fail-fast on excess usage."""
        daily_total = self.daily_total()
        monthly_total = self.monthly_total()
        self.logger.info("cost_limits daily=%.4f monthly=%.4f current_daily=%.4f current_monthly=%.4f", daily, monthly, daily_total, monthly_total)

        if daily_total > daily:
            self.logger.warning("daily cost limit exceeded: %.4f > %.4f", daily_total, daily)
        if monthly_total > monthly:
            self.logger.warning("monthly cost limit exceeded: %.4f > %.4f", monthly_total, monthly)
        if daily_total >= daily * 2:
            raise CostLimitExceededError("Daily spend reached 2x configured limit.")
        return daily_total <= daily and monthly_total <= monthly

    def _persist(self) -> None:
        path = self._storage_path()
        path.parent.mkdir(exist_ok=True)
        payload = {
            "records": [record.model_dump(mode="json") for record in self.records],
            "daily_totals": dict(self._daily_totals),
            "monthly_totals": dict(self._monthly_totals),
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")