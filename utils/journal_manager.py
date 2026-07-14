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

    def summarise_for_prompt(self, ticker: str, n: int = 6) -> str:
        """Return a human-readable summary of the last n journal entries for prompt injection.

        Example output:
        --- Journal: TICKER (last N entries) ---
        [2024-06-15 09:35 OPEN] Decision: BUY | catalyst_strength=2.0 | Rationale: ...
        [2024-06-15 12:30 MID]  Decision: HOLD | technical_confirmation=1.5 | Rationale: ...
        --- end journal ---
        """
        entries = self.get_recent(ticker, n)
        if not entries:
            return f"No prior journal entries for {ticker}."

        lines = [f"--- Journal: {ticker} (last {len(entries)} entries) ---"]
        for entry in entries:
            try:
                try:
                    dt = datetime.fromisoformat(entry.timestamp)
                    timestr = dt.strftime("%Y-%m-%d %H:%M")
                except Exception:
                    timestr = entry.timestamp
                catalyst = entry.key_metrics.get("edge_catalyst") if isinstance(entry.key_metrics, dict) else None
                technical = entry.key_metrics.get("edge_technical") if isinstance(entry.key_metrics, dict) else None
                parts = [f"[{timestr} {entry.cycle_type}] Decision: {entry.decision}"]
                if catalyst is not None:
                    parts.append(f"catalyst_strength={catalyst}")
                if technical is not None:
                    parts.append(f"technical_confirmation={technical}")
                parts.append(f"Rationale: {entry.decision_rationale}")
                lines.append(" | ".join(parts))
            except Exception:
                # best-effort formatting
                lines.append(f"[{entry.timestamp} {entry.cycle_type}] Decision: {entry.decision} | Rationale: {entry.decision_rationale}")
        lines.append("--- end journal ---")
        return "\n".join(lines)
