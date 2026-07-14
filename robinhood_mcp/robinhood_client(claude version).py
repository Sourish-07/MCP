from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import subprocess
import time
import traceback
import uuid
from typing import Any

from mcp.response_recorder import capture_mcp_call, _extract_tool_name
from models.decisions import ExecutionResult
from models.market_data import EquityQuote, OHLCVBar
from models.portfolio import Portfolio, Position


class MCPError(RuntimeError):
    """Raised when a Robinhood MCP request fails."""


class RobinhoodMCPClient:
    """Interface to the Robinhood Trading MCP via the Claude Code CLI.

    The MCP must be registered first:
        claude mcp add robinhood-trading --transport http https://agent.robinhood.com/mcp/trading
    """

    CLAUDE_EXECUTABLE = r"C:\Users\Shris\AppData\Roaming\npm\claude.cmd"

    def __init__(self) -> None:
        self.logger = logging.getLogger("robinhood-agent.mcp")
        self.max_retries = 1
        self._mcp_debug_logger = self._setup_mcp_debug_logger()
        self._current_process: subprocess.Popen | None = None
        self._cached_account_number = ""

    # ------------------------------------------------------------------ #
    #  Diagnostic logging helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _setup_mcp_debug_logger() -> logging.Logger:
        """Create a dedicated logger that writes ONLY to logs/claude_mcp_debug.log."""
        logger = logging.getLogger("robinhood-agent.mcp.debug")
        logger.setLevel(logging.DEBUG)
        logger.propagate = False  # do NOT leak into the root / agent loggers

        # Ensure the logs directory exists (relative to project root detection)
        # We locate the project root by walking up from this file's directory.
        _current_file_dir = os.path.dirname(os.path.abspath(__file__))
        _project_root = os.path.dirname(_current_file_dir)  # mcp/ -> project root
        _log_dir = os.path.join(_project_root, "logs")
        os.makedirs(_log_dir, exist_ok=True)

        _log_path = os.path.join(_log_dir, "claude_mcp_debug.log")
        handler = logging.FileHandler(
            _log_path, mode="a", encoding="utf-8", delay=False
        )
        handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter(
            "%(asctime)s %(levelname)s [Request %(mcp_request_id)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        return logger

    def _generate_request_id(self) -> str:
        """Return a short unique request identifier (first 8 chars of UUID hex)."""
        return uuid.uuid4().hex[:8]

    @staticmethod
    def _format_output(label: str, text: str | None, max_chars: int = 500) -> str:
        """Format stdout/stderr for logging: first N, last N, or all if short."""
        if text is None:
            return f"  {label}: (None)\n"
        if not text:
            return f"  {label}: (empty)\n"
        length = len(text)
        if length <= max_chars:
            return f"  {label} ({length} chars):\n{text}\n"
        first = text[:max_chars]
        last = text[-max_chars:]
        return (
            f"  {label} ({length} chars) -- FIRST {max_chars}:\n{first}\n"
            f"  ... (truncated {length - 2 * max_chars} chars) ...\n"
            f"  {label} ({length} chars) -- LAST {max_chars}:\n{last}\n"
        )

    # ------------------------------------------------------------------ #
    #  Public MCP wrappers (unchanged business logic)
    # ------------------------------------------------------------------ #

    async def get_portfolio(self) -> Portfolio:
        """Fetch the current portfolio as a Portfolio model."""
        prompt = (
            "Use the robinhood-trading MCP tool get_portfolio "
            "and return the result as raw JSON only, no explanation."
        )
        raw = await self._call_with_retry(prompt)
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            last_brace = raw.rfind("}")
            if last_brace == -1:
                raise
            trimmed = raw[:last_brace + 1]
            open_braces = trimmed.count("{") - trimmed.count("}")
            open_brackets = trimmed.count("[") - trimmed.count("]")
            trimmed = trimmed + ("}" * open_braces) + ("]" * open_brackets)
            parsed = json.loads(trimmed)
        if isinstance(parsed, dict) and "data" in parsed:
            portfolio_data = parsed["data"]
        else:
            portfolio_data = parsed

        # Robinhood returns all numeric fields as strings — coerce to float
        for field in ("total_value", "equity_value", "options_value",
                      "futures_value", "event_contracts_value", "crypto_value",
                      "cash", "pending_deposits", "mutual_funds_value",
                      "fixed_income_value"):
            if field in portfolio_data and isinstance(portfolio_data[field], str):
                try:
                    portfolio_data[field] = float(portfolio_data[field])
                except (ValueError, TypeError):
                    portfolio_data[field] = 0.0

        # Robinhood returns buying_power as a nested dict — extract the value
        bp = portfolio_data.get("buying_power")
        if isinstance(bp, dict):
            portfolio_data["buying_power"] = float(
                bp.get("buying_power") or bp.get("unleveraged_buying_power") or 0
            )
        elif isinstance(bp, str):
            try:
                portfolio_data["buying_power"] = float(bp)
            except (ValueError, TypeError):
                portfolio_data["buying_power"] = 0.0

        # Final safety net: coerce any remaining string numerics that were
        # not in the original explicit field list, in case Robinhood adds
        # new fields or changes response shape slightly
        for key, value in list(portfolio_data.items()):
            if isinstance(value, str):
                try:
                    portfolio_data[key] = float(value)
                except (ValueError, TypeError):
                    pass

        return Portfolio.model_validate(portfolio_data)

    async def get_equity_quotes(self, tickers: list[str]) -> list[EquityQuote]:
        """Fetch latest quotes for up to 20 tickers."""
        if not tickers:
            return []
        chunk = tickers[:20]
        prompt = (
            f"Use the robinhood-trading MCP tool get_equity_quotes with tickers "
            f"{json.dumps(chunk)} and return raw JSON only, no explanation."
        )
        raw = await self._call_with_retry(prompt)
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            last_brace = raw.rfind("}")
            if last_brace == -1:
                raise
            trimmed = raw[:last_brace + 1]
            open_braces = trimmed.count("{") - trimmed.count("}")
            open_brackets = trimmed.count("[") - trimmed.count("]")
            trimmed = trimmed + ("}" * open_braces) + ("]" * open_brackets)
            parsed = json.loads(trimmed)
        if isinstance(parsed, dict) and "data" in parsed:
            results = parsed["data"].get("results", [])
            quote_list = [r["quote"] for r in results if "quote" in r]
        elif isinstance(parsed, list):
            quote_list = parsed
        else:
            quote_list = []
        mapped = []
        for q in quote_list:
            mapped.append({
                "ticker": q.get("symbol", ""),
                "price": float(q.get("last_trade_price") or q.get("price") or 0),
                "bid": float(q.get("bid_price") or q.get("bid") or 0),
                "ask": float(q.get("ask_price") or q.get("ask") or 0),
                "volume": float(q.get("volume") or 0),
                "prev_close": float(
                    q.get("adjusted_previous_close")
                    or q.get("previous_close")
                    or q.get("prev_close")
                    or 0
                ),
                "change_pct": 0.0,
            })
        return [EquityQuote.model_validate(m) for m in mapped]

    async def get_equity_historicals(self, ticker: str, span: str = "3month", interval: str = "day") -> list[OHLCVBar]:
        """Fetch OHLCV historical bars for a ticker.

        Args:
            ticker: Equity ticker symbol.
            span: One of 'day', 'week', 'month', '3month', 'year'.
            interval: Resolution such as 'day' or 'hour'.
        """
        prompt = (
            f"Use the robinhood-trading MCP tool get_equity_historicals for ticker '{ticker}' "
            f"with span '{span}' and interval 'day' and return raw JSON only, no explanation."
        )
        raw = await self._call_with_retry(prompt)
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            last_brace = raw.rfind("}")
            if last_brace == -1:
                raise
            trimmed = raw[:last_brace + 1]
            open_braces = trimmed.count("{") - trimmed.count("}")
            open_brackets = trimmed.count("[") - trimmed.count("]")
            trimmed = trimmed + ("}" * open_braces) + ("]" * open_brackets)
            parsed = json.loads(trimmed)
        if isinstance(parsed, dict) and "data" in parsed:
            results = parsed["data"].get("results", [])
            bar_list = results[0].get("bars", []) if results else []
        elif isinstance(parsed, list):
            bar_list = parsed
        else:
            bar_list = []
        mapped_bars = []
        for item in bar_list:
            mapped_bars.append({
                "ticker": ticker,
                "timestamp": item.get("begins_at") or item.get("timestamp", ""),
                "open": float(item.get("open_price") or item.get("open") or 0),
                "high": float(item.get("high_price") or item.get("high") or 0),
                "low": float(item.get("low_price") or item.get("low") or 0),
                "close": float(item.get("close_price") or item.get("close") or 0),
                "volume": float(item.get("volume") or 0),
            })
        return [OHLCVBar.model_validate(m) for m in mapped_bars]

    async def _get_account_number(self) -> str:
        """Fetch and cache the Robinhood account number for use in tool calls
        that require it explicitly."""
        if hasattr(self, '_cached_account_number') and self._cached_account_number:
            return self._cached_account_number
        prompt = (
            "Use the robinhood-trading MCP tool get_accounts and return the "
            "account number as raw JSON only, no explanation."
        )
        try:
            raw = await self._call_with_retry(prompt)
            parsed = json.loads(raw)
            if isinstance(parsed, dict) and "data" in parsed:
                results = parsed["data"].get("results", parsed["data"])
                if isinstance(results, list) and results:
                    account_number = results[0].get("account_number", "")
                elif isinstance(results, dict):
                    account_number = results.get("account_number", "")
                else:
                    account_number = ""
            else:
                account_number = ""
            self._cached_account_number = account_number
            return account_number
        except Exception as exc:
            self.logger.warning("account_number_fetch_failed: %s", exc)
            return ""

    async def get_equity_tradability(self, ticker: str) -> dict[str, Any]:
        """Fetch tradability metadata for a ticker."""
        account_number = await self._get_account_number()
        account_clause = f"for account '{account_number}' " if account_number else ""
        prompt = (
            f"Use the robinhood-trading MCP tool get_equity_tradability {account_clause}"
            f"for ticker '{ticker}' and return raw JSON only, no explanation."
        )
        raw = await self._call_with_retry(prompt)
        return json.loads(raw)

    async def get_watchlist_tickers(self, watchlist_name: str = "Default") -> list[str]:
        """Fetch ticker symbols from a named Robinhood watchlist via MCP."""
        prompt = (
            f"Use the robinhood-trading MCP tool get_watchlist_items for watchlist "
            f"named '{watchlist_name}' and return the list of symbols as a raw JSON "
            f"array of strings only, no explanation. Example: [\"AAPL\", \"NVDA\"]"
        )
        try:
            raw = await self._call_with_retry(prompt)
            parsed = json.loads(raw)
            if isinstance(parsed, list):
                return [str(t).upper().strip() for t in parsed if t]
            if isinstance(parsed, dict):
                results = parsed.get("data", {}).get("results",
                          parsed.get("results", []))
                if isinstance(results, list):
                    tickers = []
                    for item in results:
                        if isinstance(item, str):
                            tickers.append(item.upper().strip())
                        elif isinstance(item, dict):
                            sym = item.get("symbol") or item.get("ticker") or ""
                            if sym:
                                tickers.append(sym.upper().strip())
                    return tickers
            return []
        except Exception as exc:
            self.logger.warning("watchlist_fetch_failed name=%s error=%s",
                                watchlist_name, exc)
            return []

    async def get_equity_positions(self) -> list[Position]:
        """Fetch current open positions as Position models."""
        prompt = (
            "Use the robinhood-trading MCP tool get_equity_positions "
            "and return raw JSON array only, no explanation."
        )
        raw = await self._call_with_retry(prompt)
        return [Position.model_validate(item) for item in json.loads(raw)]

    async def review_equity_order(
        self, ticker: str, side: str, quantity: float, order_type: str = "market"
    ) -> dict[str, Any]:
        """Preview an order; returns pre-trade warnings without placing it."""
        prompt = (
            f"Use the robinhood-trading MCP tool review_equity_order with ticker '{ticker}', "
            f"side '{side}', quantity {quantity}, order_type '{order_type}' "
            "and return raw JSON only, no explanation."
        )
        raw = await self._call_with_retry(prompt)
        return json.loads(raw)

    async def place_equity_order(
        self,
        ticker: str,
        side: str,
        quantity: float,
        order_type: str = "market",
        dry_run: bool = True,
    ) -> ExecutionResult:
        """Submit or simulate an equity order.

        Args:
            dry_run: If True, calls review_equity_order only and never places a live order.
        """
        if dry_run:
            preview = await self.review_equity_order(ticker, side, quantity, order_type)
            self.logger.info("dry_run_preview ticker=%s preview=%s", ticker, preview)
            return ExecutionResult(
                ticker=ticker,
                status="SIMULATED",
                fill_price=preview.get("estimated_price", 0.0),
                quantity=quantity,
                dry_run=True,
                timestamp=preview.get("timestamp", ""),
            )

        prompt = (
            f"Use the robinhood-trading MCP tool place_equity_order with ticker '{ticker}', "
            f"side '{side}', quantity {quantity}, order_type '{order_type}' "
            "and return raw JSON only, no explanation."
        )
        raw = await self._call_with_retry(prompt)
        return ExecutionResult.model_validate(json.loads(raw))

    # ------------------------------------------------------------------ #
    #  Internal helpers
    # ------------------------------------------------------------------ #

    async def _call_with_retry(self, prompt: str) -> str:
        """Retry Claude CLI invocations with exponential back-off.

        Every attempt (success or failure) is automatically recorded by
        the response recorder BEFORE any business logic touches the data.
        The recorder captures the raw stdout/stderr exactly as the
        subprocess returned them.
        """
        request_id = self._generate_request_id()
        debug = self._mcp_debug_logger
        last_error: Exception | None = None
        total_start = time.perf_counter()
        tool_name = _extract_tool_name(prompt)

        for attempt in range(1, self.max_retries + 1):
            attempt_start = time.perf_counter()
            result_dict: dict[str, Any] | None = None
            exc_info: dict[str, Any] | None = None
            try:
                self.logger.info("mcp_call attempt=%d", attempt)

                extra = {"mcp_request_id": request_id}
                debug.info(
                    "Attempt %d/%d starting",
                    attempt, self.max_retries, extra=extra,
                )

                try:
                    result_dict = await asyncio.to_thread(
                        self._call_claude_mcp, prompt, request_id
                    )
                except (KeyboardInterrupt, asyncio.CancelledError):
                    proc = self._current_process
                    if proc is not None and proc.pid is not None:
                        self.logger.warning(
                            "manual_interrupt | killing process tree pid=%d", proc.pid
                        )
                        try:
                            subprocess.run(
                                ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                                capture_output=True,
                                timeout=10,
                            )
                        except Exception:
                            pass
                    raise

                # ── Non‑zero returncode → treat as failure ───────────
                if result_dict.get("returncode", 0) != 0:
                    stderr_hint = (result_dict.get("raw_stderr") or "")[:200]
                    raise MCPError(
                        stderr_hint
                        or (result_dict.get("raw_stdout") or "")[:200]
                        or f"Claude CLI exited with code {result_dict.get('returncode')}"
                    )

                text = result_dict["stripped_text"]
                self.logger.debug("mcp_response=%s", text[:200])

                attempt_elapsed = time.perf_counter() - attempt_start
                total_elapsed = time.perf_counter() - total_start
                debug.info(
                    "Attempt %d succeeded | attempt_runtime=%.3fs total_runtime=%.3fs",
                    attempt, attempt_elapsed, total_elapsed, extra=extra,
                )

                # ── RECORD SUCCESSFUL ATTEMPT ──────────────────────
                # Fire-and-forget: recording failure must NEVER block the return
                asyncio.ensure_future(
                    capture_mcp_call(
                        tool_name=tool_name,
                        prompt=prompt,
                        duration_ms=attempt_elapsed * 1000,
                        attempt=attempt,
                        claude_executable=self.CLAUDE_EXECUTABLE,
                        pid=result_dict.get("pid"),
                        returncode=result_dict.get("returncode"),
                        stdout_raw=result_dict.get("raw_stdout"),
                        stderr_raw=result_dict.get("raw_stderr"),
                        exception_info=None,
                    )
                )

                return text

            except Exception as exc:
                last_error = exc
                attempt_elapsed = time.perf_counter() - attempt_start
                total_elapsed = time.perf_counter() - total_start

                self.logger.warning("mcp_attempt_%d_failed: %s", attempt, exc)

                extra = {"mcp_request_id": request_id}
                debug.warning(
                    "Attempt %d failed | runtime=%.3fs total_runtime=%.3fs",
                    attempt, attempt_elapsed, total_elapsed, extra=extra,
                )
                debug.warning(
                    "Exception: %s | Message: %s",
                    type(exc).__name__, exc, extra=extra,
                )
                debug.exception(
                    "Full traceback for Attempt %d (Request %s)",
                    attempt, request_id, extra=extra,
                )

                # ── RECORD FAILED ATTEMPT ──────────────────────────
                # Build exception info with exact stdout, stderr, traceback
                exc_info = {
                    "exception_class": type(exc).__name__,
                    "exception_message": str(exc),
                    "exception_traceback": traceback.format_exc(),
                    "is_mcp_error": isinstance(exc, MCPError),
                    "is_json_decode_error": isinstance(exc, json.JSONDecodeError),
                }

                # If we got a result_dict from _call_claude_mcp before
                # it raised, capture its raw data; otherwise use None
                raw_stdout = result_dict.get("raw_stdout") if result_dict else None
                raw_stderr = result_dict.get("raw_stderr") if result_dict else None
                pid = result_dict.get("pid") if result_dict else None
                returncode = result_dict.get("returncode") if result_dict else None

                asyncio.ensure_future(
                    capture_mcp_call(
                        tool_name=tool_name,
                        prompt=prompt,
                        duration_ms=attempt_elapsed * 1000,
                        attempt=attempt,
                        claude_executable=self.CLAUDE_EXECUTABLE,
                        pid=pid,
                        returncode=returncode,
                        stdout_raw=raw_stdout,
                        stderr_raw=raw_stderr,
                        exception_info=exc_info,
                    )
                )

                if attempt < self.max_retries:
                    backoff = 2 ** (attempt - 1)
                    debug.info(
                        "Sleeping %d seconds before retry ...",
                        backoff, extra=extra,
                    )
                    await asyncio.sleep(backoff)

        total_elapsed = time.perf_counter() - total_start
        extra = {"mcp_request_id": request_id}
        debug.critical(
            "All %d attempts exhausted | total_runtime=%.3fs",
            self.max_retries, total_elapsed, extra=extra,
        )
        raise MCPError("Robinhood MCP request failed after retries") from last_error

    def _call_claude_mcp(self, prompt: str, request_id: str = "") -> dict[str, Any]:
        """Run the Claude CLI synchronously and return a result dict.

        The dict contains:
            - stripped_text: cleaned JSON after markdown fence removal
            - raw_stdout: original stdout exactly as received
            - raw_stderr: original stderr exactly as received
            - pid: process PID (or None)
            - returncode: process return code (or None)
            - total_elapsed: total wall-clock duration in seconds

        This refactor preserves the original raw data for the response
        recorder. The caller (`_call_with_retry`) extracts what it needs
        and passes the raw values to `capture_mcp_call()`.
        """
        if not request_id:
            request_id = self._generate_request_id()
        debug = self._mcp_debug_logger
        extra: dict[str, Any] = {"mcp_request_id": request_id}

        cmd = [
            self.CLAUDE_EXECUTABLE,
            "--model", "claude-haiku-4-5",
            "--print", prompt,
        ]
        working_dir = os.getcwd()
        prompt_char_count = len(prompt)
        prompt_word_count = len(prompt.split())

        # ── Pre-launch logging ──────────────────────────────────────
        debug.info("=" * 58, extra=extra)
        debug.info("REQUEST START", extra=extra)
        debug.info("=" * 58, extra=extra)
        debug.info("Request ID:        %s", request_id, extra=extra)
        debug.info("Characters:        %d", prompt_char_count, extra=extra)
        debug.info("Words:             %d", prompt_word_count, extra=extra)
        debug.info("Command list:      %s", repr(cmd), extra=extra)
        debug.info("Executable path:   %s", self.CLAUDE_EXECUTABLE, extra=extra)
        debug.info("Working directory: %s", working_dir, extra=extra)
        debug.info("Prompt:\n%s", prompt, extra=extra)

        launch_start = time.perf_counter()

        try:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,
            )
            self._current_process = process

            # ── Subprocess launched ─────────────────────────────────
            launch_elapsed = time.perf_counter() - launch_start
            debug.info(
                "Process started | PID=%d | executable=%s | launch_time=%.4fs",
                process.pid, self.CLAUDE_EXECUTABLE, launch_elapsed, extra=extra,
            )

            comm_start = time.perf_counter()

            stdout, stderr = process.communicate()
            returncode = process.returncode
            pid = process.pid
            self._current_process = None
            comm_elapsed = time.perf_counter() - comm_start
            total_elapsed = time.perf_counter() - launch_start

            # ── Process completed ───────────────────────────────────
            debug.info(
                "communicate() completed | comm_time=%.4fs | total_time=%.4fs",
                comm_elapsed, total_elapsed, extra=extra,
            )
            debug.info(
                "Return code:       %d", returncode, extra=extra,
            )
            debug.info(
                "stdout size:       %d chars", len(stdout or ""), extra=extra,
            )
            debug.info(
                "stderr size:       %d chars", len(stderr or ""), extra=extra,
            )

            debug.info("stdout:", extra=extra)
            debug.info(self._format_output("stdout", stdout), extra=extra)
            debug.info("stderr:", extra=extra)
            debug.info(self._format_output("stderr", stderr), extra=extra)

            # Always return the full result dict — NEVER lose stdout/stderr.
            # The caller inspects `returncode` to decide success vs failure.
            # This is critical: non‑zero exits can still contain valuable
            # stderr / stdout that the recorder must capture.

            if returncode != 0:
                debug.error(
                    "Non-zero return code | returncode=%d", returncode, extra=extra,
                )

            # ── Always strip / clean text for the caller ──────────────
            text = stdout.strip()
            text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
            text = re.sub(r"\s*```$", "", text)
            text = text.strip()

            debug.info("=" * 58, extra=extra)
            debug.info("REQUEST COMPLETED", extra=extra)
            debug.info("=" * 58, extra=extra)
            debug.info("Elapsed:          %.2f seconds", total_elapsed, extra=extra)
            debug.info("Return code:      %d", returncode, extra=extra)

            return {
                "stripped_text": text,
                "raw_stdout": stdout,
                "raw_stderr": stderr,
                "pid": pid,
                "returncode": returncode,
                "total_elapsed": total_elapsed,
            }

        except FileNotFoundError as exc:
            total_elapsed = time.perf_counter() - launch_start
            debug.exception(
                "Claude CLI not found | elapsed=%.2fs", total_elapsed, extra=extra,
            )
            raise MCPError(
                "Claude CLI not found on PATH — run: npm install -g @anthropic-ai/claude-code"
            ) from exc

        except OSError as exc:
            total_elapsed = time.perf_counter() - launch_start
            debug.exception(
                "Claude CLI OS error | elapsed=%.2fs", total_elapsed, extra=extra,
            )
            raise MCPError(f"Claude CLI OS error: {exc}") from exc
