from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path


class UTCFormatter(logging.Formatter):
    """Formatter that emits UTC timestamps."""

    def formatTime(self, record: logging.LogRecord, datefmt: str | None = None) -> str:
        dt = datetime.fromtimestamp(record.created, tz=timezone.utc)
        if datefmt:
            return dt.strftime(datefmt)
        return dt.isoformat(timespec="seconds")


def configure_logging() -> logging.Logger:
    """Configure console, file, and JSON trade logging handlers."""
    base = Path(__file__).resolve().parent.parent
    logs_dir = base / "logs"
    logs_dir.mkdir(exist_ok=True)

    logger = logging.getLogger("robinhood-agent")
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    if logger.handlers:
        for handler in list(logger.handlers):
            logger.removeHandler(handler)

    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.INFO)
    console.setFormatter(UTCFormatter("%(asctime)s [%(levelname)s] %(name)s - %(message)s"))
    logger.addHandler(console)

    file_handler = logging.FileHandler(logs_dir / "agent.log", encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(UTCFormatter("%(asctime)s [%(levelname)s] %(name)s - %(message)s"))
    logger.addHandler(file_handler)

    trade_logger = logging.getLogger("robinhood-agent.trades")
    trade_logger.setLevel(logging.INFO)
    trade_logger.propagate = False
    if trade_logger.handlers:
        for handler in list(trade_logger.handlers):
            trade_logger.removeHandler(handler)

    trade_handler = logging.FileHandler(logs_dir / "trades.json", encoding="utf-8")
    trade_handler.setLevel(logging.INFO)

    class JsonFormatter(logging.Formatter):
        def format(self, record: logging.LogRecord) -> str:
            payload = {
                "timestamp": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
                "level": record.levelname,
                "message": record.getMessage(),
            }
            return json.dumps(payload)

    trade_handler.setFormatter(JsonFormatter())
    trade_logger.addHandler(trade_handler)

    return logger
