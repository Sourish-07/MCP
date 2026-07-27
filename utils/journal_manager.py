from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import List

from models.decisions import JournalEntry


class JournalManager:
    """Manage per-ticker journal files stored under journals/TICKER.json.

    Each file contains a JSON array of JournalEntry objects sorted by timestamp ascending.
    """

    def __init__(self, base_dir: Path | None = None, max_entries_per_ticker: int = 90) -> None:
        self.base_dir = (
            base_dir
            if base_dir is not None
            else Path(__file__).resolve().parent.parent / "journals"
        )
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.max_entries_per_ticker = max_entries_per_ticker
        self.logger = logging.getLogger("robinhood-agent.utils.journal_manager")

    def _path_for(self, ticker: str) -> Path:
        safe = ticker.upper()
        return self.base_dir / f"{safe}.json"

    def get_recent(self, ticker: str, n: int = 10) -> List[JournalEntry]:
        """Read journals/{ticker}.json and return the last n entries (most recent last).

        On any read error or if file does not exist, return [].
        """
        path = self._path_for(ticker)
        if not path.exists():
            return []
        try:
            raw = path.read_text(encoding="utf-8")
            data = json.loads(raw) if raw else []
            entries = [JournalEntry.model_validate(item) for item in data]
            return entries[-n:]
        except Exception as exc:
            self.logger.warning("journal_read_failed ticker=%s error=%s", ticker, exc)
            return []

    def append(self, ticker: str, entry: JournalEntry) -> None:
        """Append a JournalEntry to the ticker journal. Trim to max entries.

        Writes are atomic: write to a .tmp file then replace.
        On any write error we log a warning and do not raise.
        """
        path = self._path_for(ticker)
        try:
            existing = []
            if path.exists():
                try:
                    existing_raw = path.read_text(encoding="utf-8")
                    existing = json.loads(existing_raw) if existing_raw else []
                except Exception:
                    existing = []

            payload = existing + [entry.model_dump(mode="json")]
            # trim to most recent max_entries_per_ticker
            trimmed = payload[-self.max_entries_per_ticker :]
            tmp = path.with_suffix(".tmp")
            tmp.write_text(json.dumps(trimmed, indent=2), encoding="utf-8")
            tmp.replace(path)
        except Exception as exc:
            self.logger.warning("journal_write_failed ticker=%s error=%s", ticker, exc)

    def get_all_tickers(self) -> List[str]:
        """Return list of ticker names present in the journals directory."""
        try:
            files = [p for p in self.base_dir.glob("*.json") if p.is_file()]
            return [p.stem for p in files]
        except Exception as exc:
            self.logger.warning("journal_list_failed: %s", exc)
            return []

    def summarise_for_prompt(self, ticker: str, n: int = 15) -> str:
        """Return a PRICE-AND-METRICS-ONLY summary of recent journal entries for prompt injection.

        Past DECISIONS (IGNORE/BUY/HOLD/SELL) and their rationales are
        intentionally NOT shown — the model is not allowed to anchor on its
        own prior verdicts.  Only historical price and technical metric
        fields are included so the model can understand how the stock has
        actually moved over time without being biased by what previous
        cycles decided.

        With n=15 (the default), up to ~5 trading days of multi-cycle
        snapshots are shown (OPEN/MID/CLOSE), giving the model a real
        trajectory to read rather than a 2-3 session fragment.

        Example output:
        --- Historical PRICE/METRIC trajectory for TICKER (oldest→newest, last 15 sessions with price data; DECISIONS deliberately omitted) ---
        [2026-07-14 13:36 OPEN] price=203.53 | rsi=49.9 | macd_hist=1.21 | bb_zscore=0.12 | ret5d=+4.08% | ret30d=-5.00% | ret90d=+10.99% | dd30d=14.19% | vol_spike=-17.27% | vol30d=44.3 | vs20dma=+0.82%
        --- end metrics ---
        """
        entries = self.get_recent(ticker, n)
        if not entries:
            return f"No prior price/metric data for {ticker}."

        # Filter to entries that have actual price data (current_price > 0).
        # Zero-price entries are data-outage sessions; skip them so the
        # trajectory is not polluted with phantom $0 points.
        valid = [e for e in entries if isinstance(e.key_metrics, dict) and float(e.key_metrics.get("current_price", 0.0)) > 0.0]

        shown = min(len(valid), n)
        if not valid:
            return f"No price/metric data available for {ticker} (all recent sessions had price=0)."

        lines = [f"--- Historical PRICE/METRIC trajectory for {ticker} (oldest→newest, last {shown} sessions with price data; DECISIONS deliberately omitted) ---"]

        # Safe numeric formatters — coerce None/missing to 0 so the
        # method never crashes on a bare :.2f format spec.
        def _num(v, spec: str = ".2f") -> str:
            try:
                return format(float(v if v is not None else 0.0), spec)
            except Exception:
                return "—"

        def _pct(v) -> str:
            try:
                return f"{float(v if v is not None else 0.0):+.2f}%"
            except Exception:
                return "—"

        for entry in valid[-n:]:
            try:
                try:
                    dt = datetime.fromisoformat(entry.timestamp)
                    timestr = dt.strftime("%Y-%m-%d %H:%M")
                except Exception:
                    timestr = entry.timestamp
                km = entry.key_metrics if isinstance(entry.key_metrics, dict) else {}
                price_str = _num(km.get("current_price"), ".2f") if km.get("current_price") else "—"
                line = (
                    f"[{timestr} {entry.cycle_type}] "
                    f"price={price_str} | "
                    f"rsi={_num(km.get('rsi_14'), '.1f')} | "
                    f"macd_hist={_num(km.get('macd_histogram'), '.4f')} | "
                    f"bb_zscore={_num(km.get('bb_zscore'), '.4f')} | "
                    f"ret5d={_pct(km.get('return_5d'))} | "
                    f"ret30d={_pct(km.get('return_30d'))} | "
                    f"ret90d={_pct(km.get('return_90d'))} | "
                    f"dd30d={_num(km.get('drawdown_30d'), '.2f')}% | "
                    f"vol_spike={_num(km.get('volume_spike_pct'), '.2f')}% | "
                    f"vol30d={_num(km.get('realized_vol_30d'), '.2f')} | "
                    f"vs20dma={_pct(km.get('distance_from_20dma'))}"
                )
                lines.append(line)
            except Exception:
                # Best-effort fallback: just timestamp + cycle, skip if broken
                lines.append(f"[{entry.timestamp} {entry.cycle_type}] (metrics unavailable)")
        lines.append("--- end metrics ---")
        return "\n".join(lines)
