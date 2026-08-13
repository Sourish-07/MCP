from __future__ import annotations

"""
mcp/robinhood_client.py — Direct Python MCP client for Robinhood Trading.

ARCHITECTURE CHANGE FROM PREVIOUS VERSION:
    The previous version shelled out to `claude --print "..."` for every
    single call, which meant every request paid for Claude Code's full
    internal system prompt + all 46 MCP tool schemas being loaded fresh
    (confirmed via Anthropic dashboard: averaging ~44,798 input tokens per
    call, some spiking to 71,600 — purely CLI overhead, unrelated to the
    actual one-line request being made).

    This version connects DIRECTLY to Robinhood's MCP server
    (https://agent.robinhood.com/mcp/trading) over Streamable HTTP using
    the official `mcp` Python SDK, authenticated via a real OAuth 2.1 +
    PKCE flow against Robinhood's own published endpoints. Zero LLM calls,
    zero subprocess, zero token cost anywhere in this file. This was
    validated end-to-end against the real Robinhood MCP server and a real
    $2,000 Agentic account before this rewrite was written — every phase
    (registration, browser auth, token refresh, tool discovery, and a live
    get_portfolio call returning real account data) passed.

WHAT STAYS IDENTICAL:
    Every public method signature, every return type (same Pydantic
    models), and every piece of business logic for cleaning up Robinhood's
    response shapes (string-to-float coercion, nested buying_power
    extraction, quote/historicals field remapping) is preserved exactly
    as validated in the previous version. Callers in core/data_ingest.py,
    core/execution.py, and elsewhere do not need to change.

WHAT'S NEW AND IMPORTANT:
    - Direct calls require `account_number` as an explicit argument for
      every account-scoped tool (get_portfolio, get_equity_positions,
      get_equity_orders, get_equity_tradability, review_equity_order,
      place_equity_order). This was NOT required when Claude Code sat in
      the middle — Claude was apparently resolving account context on its
      own before ever reaching the tool call. This is now handled
      automatically and cached via _get_account_number(), which correctly
      selects the account where agentic_allowed=true (NOT the default
      individual account) — this exact selection logic was validated live.
    - OAuth tokens (access + refresh) are cached to .rh_oauth_cache.json
      in the project root. The one-time browser login only needs to
      happen once; every subsequent run silently refreshes.
    - Every tool call result flows through the response recorder
      (mcp/response_recorder.py) via the new capture_direct_mcp_call()
      adapter, preserving full observability (schema drift detection,
      numeric anomaly detection, credential redaction) without any of the
      subprocess/stdout-specific machinery that no longer applies.

SETUP REQUIRED BEFORE THIS WORKS:
    1. pip install mcp   (confirmed working at mcp==1.28.1)
    2. First run will open a browser for one-time Robinhood login/approval.
       Every run after that reuses the cached refresh token silently.
    3. If .rh_oauth_cache.json is ever deleted or the refresh token is
       revoked from Robinhood's connected-apps settings, the next run
       will automatically fall back to a fresh browser login.
"""

import asyncio
import base64
import hashlib
import json
import logging
import secrets
import threading
import time
import traceback
import webbrowser
from datetime import datetime, timedelta, timezone
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlencode, urlparse

import httpx

from robinhood_mcp.response_recorder import capture_direct_mcp_call
from models.decisions import ExecutionResult
from models.market_data import EquityQuote, OHLCVBar
from models.portfolio import Portfolio, Position


class MCPError(RuntimeError):
    """Raised when a Robinhood MCP request fails."""


# --------------------------------------------------------------------- #
#  OAuth / connection constants — validated against the real server
# --------------------------------------------------------------------- #
ROBINHOOD_MCP_URL = "https://agent.robinhood.com/mcp/trading"
REGISTRATION_ENDPOINT = "https://agent.robinhood.com/oauth/trading/register"
AUTHORIZATION_ENDPOINT = "https://robinhood.com/oauth"
TOKEN_ENDPOINT = "https://api.robinhood.com/oauth2/token/"
REDIRECT_PORT = 53271
REDIRECT_URI = f"http://localhost:{REDIRECT_PORT}/callback"

# Project-root-relative cache path (mcp/ -> project root -> cache file)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
TOKEN_CACHE_PATH = _PROJECT_ROOT / ".rh_oauth_cache.json"

# Refresh proactively when less than this many seconds remain on the
# access token, rather than waiting for a live 401 mid-cycle.
TOKEN_REFRESH_MARGIN_SECONDS = 600  # 10 minutes

_SPAN_TO_TIMEDELTA = {
    "day": timedelta(days=1),
    "week": timedelta(days=7),
    "month": timedelta(days=30),
    "3month": timedelta(days=90),
    "6month": timedelta(days=180),
    "year": timedelta(days=365),
}


# --------------------------------------------------------------------- #
#  PKCE + local OAuth redirect catcher (identical validated pattern)
# --------------------------------------------------------------------- #
def _generate_pkce_pair() -> tuple[str, str]:
    """Return (code_verifier, code_challenge) per RFC 7636 using S256."""
    code_verifier = base64.urlsafe_b64encode(secrets.token_bytes(64)).decode("ascii").rstrip("=")
    digest = hashlib.sha256(code_verifier.encode("ascii")).digest()
    code_challenge = base64.urlsafe_b64encode(digest).decode("ascii").rstrip("=")
    return code_verifier, code_challenge


class _CallbackState:
    code: str | None = None
    state: str | None = None
    error: str | None = None
    received = threading.Event()


class _CallbackHandler(BaseHTTPRequestHandler):
    def do_GET(self):  # noqa: N802
        parsed = urlparse(self.path)
        qs = parse_qs(parsed.query)
        if "code" in qs:
            _CallbackState.code = qs["code"][0]
            _CallbackState.state = qs.get("state", [None])[0]
            body = b"<html><body><h2>Authorization received. You can close this tab.</h2></body></html>"
            self.send_response(200)
        else:
            _CallbackState.error = qs.get("error", ["unknown_error"])[0]
            body = b"<html><body><h2>Authorization failed. Check the terminal.</h2></body></html>"
            self.send_response(400)
        self.send_header("Content-Type", "text/html")
        self.end_headers()
        self.wfile.write(body)
        _CallbackState.received.set()

    def log_message(self, format, *args):  # noqa: A002
        pass


def _run_callback_server_blocking() -> None:
    server = HTTPServer(("localhost", REDIRECT_PORT), _CallbackHandler)
    server.timeout = 180
    server.handle_request()


class RobinhoodMCPClient:
    """Direct authenticated Python client for the Robinhood Trading MCP server.

    Replaces the previous Claude Code CLI subprocess bridge entirely.
    Connects over Streamable HTTP with a real OAuth 2.1 + PKCE token,
    cached and refreshed automatically across runs.
    """

    def __init__(self) -> None:
        self.logger = logging.getLogger("robinhood-agent.mcp")
        self.max_retries = 3
        self._cached_account_number: str = ""
        self._token_cache: dict[str, Any] = {}
        self._token_lock = asyncio.Lock()

    # ------------------------------------------------------------------ #
    #  Token lifecycle
    # ------------------------------------------------------------------ #

    def _load_token_cache(self) -> dict[str, Any]:
        if not TOKEN_CACHE_PATH.exists():
            return {}
        try:
            return json.loads(TOKEN_CACHE_PATH.read_text(encoding="utf-8"))
        except Exception as exc:
            self.logger.warning("token_cache_read_failed: %s", exc)
            return {}

    def _save_token_cache(self, token_data: dict[str, Any]) -> None:
        try:
            TOKEN_CACHE_PATH.write_text(json.dumps(token_data, indent=2), encoding="utf-8")
        except Exception as exc:
            self.logger.warning("token_cache_write_failed: %s", exc)

    def _token_is_fresh(self, token_data: dict[str, Any]) -> bool:
        """Return True if the cached access token still has enough life left."""
        if not token_data.get("access_token"):
            return False
        issued_at = token_data.get("_issued_at", 0)
        expires_in = token_data.get("expires_in", 0)
        if not expires_in:
            return False
        expiry = issued_at + expires_in
        return (expiry - time.time()) > TOKEN_REFRESH_MARGIN_SECONDS

    async def _register_client(self) -> dict[str, Any] | None:
        """Dynamic client registration (RFC 7591). Only needed once; the
        resulting client_id is cached alongside the refresh token."""
        payload = {
            "client_name": "robinhood-agent-trading-bot",
            "redirect_uris": [REDIRECT_URI],
            "grant_types": ["authorization_code", "refresh_token"],
            "response_types": ["code"],
            "token_endpoint_auth_method": "none",
        }
        try:
            async with httpx.AsyncClient(timeout=20) as client:
                resp = await client.post(REGISTRATION_ENDPOINT, json=payload)
                if resp.status_code in (200, 201):
                    return resp.json()
                self.logger.error(
                    "oauth_registration_failed status=%d body=%s",
                    resp.status_code, resp.text[:500],
                )
                return None
        except Exception as exc:
            self.logger.error("oauth_registration_error: %s", exc)
            return None

    async def _run_browser_auth_flow(self, client_id: str) -> dict[str, Any] | None:
        """Full authorization_code + PKCE flow. Opens a browser window and
        blocks (in a background thread) until the user approves or times out.
        Only invoked when no valid cached refresh token exists."""
        code_verifier, code_challenge = _generate_pkce_pair()
        state = secrets.token_urlsafe(16)

        auth_params = {
            "response_type": "code",
            "client_id": client_id,
            "redirect_uri": REDIRECT_URI,
            "code_challenge": code_challenge,
            "code_challenge_method": "S256",
            "state": state,
            "scope": "internal",
        }
        auth_url = f"{AUTHORIZATION_ENDPOINT}?{urlencode(auth_params)}"

        self.logger.warning(
            "oauth_browser_auth_required — opening browser for one-time "
            "Robinhood login/approval. If running headless, visit manually: %s",
            auth_url,
        )

        _CallbackState.code = None
        _CallbackState.state = None
        _CallbackState.error = None
        _CallbackState.received.clear()

        server_thread = threading.Thread(target=_run_callback_server_blocking, daemon=True)
        server_thread.start()
        webbrowser.open(auth_url)

        got_response = await asyncio.to_thread(_CallbackState.received.wait, 185)
        if not got_response:
            raise MCPError("OAuth browser flow timed out after 180s waiting for approval.")
        if _CallbackState.error:
            raise MCPError(f"Robinhood OAuth returned an error: {_CallbackState.error}")
        if _CallbackState.state != state:
            raise MCPError("OAuth state mismatch — possible stale or invalid callback.")

        token_payload = {
            "grant_type": "authorization_code",
            "code": _CallbackState.code,
            "redirect_uri": REDIRECT_URI,
            "client_id": client_id,
            "code_verifier": code_verifier,
        }
        async with httpx.AsyncClient(timeout=20) as client:
            resp = await client.post(
                TOKEN_ENDPOINT, data=token_payload,
                headers={"Content-Type": "application/x-www-form-urlencoded"},
            )
            if resp.status_code != 200:
                raise MCPError(f"OAuth token exchange failed: HTTP {resp.status_code}: {resp.text[:300]}")
            token_data = resp.json()
            token_data["_client_id"] = client_id
            token_data["_issued_at"] = time.time()
            return token_data

    async def _refresh_access_token(self, refresh_token: str, client_id: str) -> dict[str, Any] | None:
        try:
            async with httpx.AsyncClient(timeout=20) as client:
                resp = await client.post(
                    TOKEN_ENDPOINT,
                    data={
                        "grant_type": "refresh_token",
                        "refresh_token": refresh_token,
                        "client_id": client_id,
                    },
                    headers={"Content-Type": "application/x-www-form-urlencoded"},
                )
                if resp.status_code != 200:
                    self.logger.warning(
                        "oauth_refresh_rejected status=%d — will fall back to full browser auth",
                        resp.status_code,
                    )
                    return None
                token_data = resp.json()
                token_data["_client_id"] = client_id
                token_data["_issued_at"] = time.time()
                # Robinhood may or may not rotate the refresh_token; if it
                # doesn't send a new one, keep reusing the existing one.
                if "refresh_token" not in token_data:
                    token_data["refresh_token"] = refresh_token
                return token_data
        except Exception as exc:
            self.logger.warning("oauth_refresh_error: %s", exc)
            return None

    async def _ensure_valid_token(self) -> str:
        """Return a valid access token, refreshing or re-authenticating as needed.
        This is called automatically before every MCP session opens — callers
        never need to think about token lifecycle."""
        async with self._token_lock:
            if self._token_is_fresh(self._token_cache):
                return self._token_cache["access_token"]

            cached = self._load_token_cache()
            if self._token_is_fresh(cached):
                self._token_cache = cached
                return cached["access_token"]

            # Try silent refresh first if we have a refresh token + client_id
            if cached.get("refresh_token") and cached.get("_client_id"):
                refreshed = await self._refresh_access_token(
                    cached["refresh_token"], cached["_client_id"]
                )
                if refreshed:
                    self._token_cache = refreshed
                    self._save_token_cache(refreshed)
                    self.logger.info("oauth_token_refreshed_silently")
                    return refreshed["access_token"]

            # Full re-auth required: register (or reuse cached client_id) + browser flow
            client_id = cached.get("_client_id")
            if not client_id:
                reg = await self._register_client()
                if not reg or "client_id" not in reg:
                    raise MCPError("Could not register OAuth client with Robinhood.")
                client_id = reg["client_id"]

            fresh = await self._run_browser_auth_flow(client_id)
            if not fresh or not fresh.get("access_token"):
                raise MCPError("OAuth browser authentication did not yield an access token.")

            self._token_cache = fresh
            self._save_token_cache(fresh)
            self.logger.info("oauth_token_obtained_via_browser_flow")
            return fresh["access_token"]

    # ------------------------------------------------------------------ #
    #  Core MCP session + tool call machinery
    # ------------------------------------------------------------------ #

    async def _call_tool_with_retry(self, tool_name: str, arguments: dict[str, Any]) -> Any:
        """Open a fresh authenticated MCP session, call one tool, close cleanly.

        A new session per call is used deliberately rather than one long-lived
        session across the whole trading day — this avoids stale-connection
        failures across the multi-hour gaps between OPEN/MID/CLOSE cycles.
        Since this transport involves zero LLM tokens, the small extra
        connection overhead per call costs nothing meaningful.
        """
        from mcp import ClientSession
        from mcp.client.streamable_http import streamablehttp_client

        last_error: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            attempt_start = time.perf_counter()
            try:
                access_token = await self._ensure_valid_token()
                headers = {"Authorization": f"Bearer {access_token}"}

                async with streamablehttp_client(ROBINHOOD_MCP_URL, headers=headers, timeout=30) as (read, write, _sid):
                    async with ClientSession(read, write) as session:
                        await session.initialize()
                        result = await session.call_tool(tool_name, arguments)

                content = [
                    c.model_dump() if hasattr(c, "model_dump") else str(c)
                    for c in result.content
                ]
                duration_ms = (time.perf_counter() - attempt_start) * 1000

                asyncio.ensure_future(
                    capture_direct_mcp_call(
                        tool_name=tool_name,
                        arguments=arguments,
                        duration_ms=duration_ms,
                        attempt=attempt,
                        success=True,
                        response_content=content,
                        exception_info=None,
                    )
                )

                return content

            except Exception as exc:
                last_error = exc
                duration_ms = (time.perf_counter() - attempt_start) * 1000
                self.logger.warning(
                    "mcp_tool_call_attempt_%d_failed tool=%s error=%s",
                    attempt, tool_name, exc,
                )

                exc_info = {
                    "exception_class": type(exc).__name__,
                    "exception_message": str(exc),
                    "exception_traceback": traceback.format_exc(),
                }
                asyncio.ensure_future(
                    capture_direct_mcp_call(
                        tool_name=tool_name,
                        arguments=arguments,
                        duration_ms=duration_ms,
                        attempt=attempt,
                        success=False,
                        response_content=None,
                        exception_info=exc_info,
                    )
                )

                # Schema/validation errors (McpError with "invalid params" or
                # "missing properties") will fail identically on every retry —
                # no point burning attempts, surface the precise error now.
                exc_str = str(exc).lower()
                if "invalid params" in exc_str or "missing properties" in exc_str:
                    raise MCPError(
                        f"Tool '{tool_name}' rejected arguments {arguments}: {exc}"
                    ) from exc

                if attempt < self.max_retries:
                    await asyncio.sleep(2 ** (attempt - 1))

        raise MCPError(f"Tool '{tool_name}' failed after {self.max_retries} attempts") from last_error

    @staticmethod
    def _extract_json_payload(content: list[dict[str, Any]]) -> Any:
        """Direct MCP tool results wrap JSON as {"type": "text", "text": "..."}.
        Extract and parse the actual payload."""
        if not content:
            return None
        for item in content:
            text = item.get("text") if isinstance(item, dict) else None
            if text:
                try:
                    return json.loads(text)
                except json.JSONDecodeError:
                    return text
        return None

    # ------------------------------------------------------------------ #
    #  Account resolution — validated live: must filter agentic_allowed=true
    # ------------------------------------------------------------------ #

    async def _get_account_number(self) -> str:
        """Fetch and cache the Robinhood Agentic account number.

        CRITICAL: Robinhood accounts can have MULTIPLE accounts under one
        login (e.g. a primary individual margin account AND a separate
        Agentic cash account). This was validated live — using the wrong
        one (e.g. is_default=true) would return the wrong account's data
        entirely. The correct selection is explicitly agentic_allowed=true,
        never is_default or "first in list".
        """
        if self._cached_account_number:
            return self._cached_account_number

        try:
            content = await self._call_tool_with_retry("get_accounts", {})
            parsed = self._extract_json_payload(content)
            accounts = []
            if isinstance(parsed, dict):
                accounts = parsed.get("data", {}).get("accounts", [])

            agentic_account = next(
                (a for a in accounts if a.get("agentic_allowed") is True), None
            )
            if agentic_account:
                account_number = agentic_account.get("account_number", "")
                self.logger.info(
                    "agentic_account_selected nickname=%s type=%s",
                    agentic_account.get("nickname", "(none)"),
                    agentic_account.get("type", "(unknown)"),
                )
            else:
                account_number = ""
                self.logger.error(
                    "no_agentic_account_found — refusing to guess which account "
                    "to use. Check that your Robinhood Agentic account is "
                    "properly set up. Raw accounts seen: %s", accounts,
                )

            self._cached_account_number = account_number
            return account_number

        except Exception as exc:
            self.logger.warning("account_number_fetch_failed: %s", exc)
            return ""

    # ------------------------------------------------------------------ #
    #  Response cleanup helpers (identical business logic to previous
    #  version — Robinhood's raw field shapes haven't changed, only the
    #  transport used to reach them)
    # ------------------------------------------------------------------ #

    @staticmethod
    def _coerce_portfolio_fields(portfolio_data: dict[str, Any]) -> dict[str, Any]:
        for field in ("total_value", "equity_value", "options_value",
                      "futures_value", "event_contracts_value", "crypto_value",
                      "cash", "pending_deposits", "mutual_funds_value",
                      "fixed_income_value"):
            if field in portfolio_data and isinstance(portfolio_data[field], str):
                try:
                    portfolio_data[field] = float(portfolio_data[field])
                except (ValueError, TypeError):
                    portfolio_data[field] = 0.0

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

        for key, value in list(portfolio_data.items()):
            if isinstance(value, str):
                try:
                    portfolio_data[key] = float(value)
                except (ValueError, TypeError):
                    pass
        return portfolio_data

    # ------------------------------------------------------------------ #
    #  Public API — same signatures and return types as before
    # ------------------------------------------------------------------ #

    async def get_portfolio(self) -> Portfolio:
        """Fetch the current portfolio as a Portfolio model.

        IMPORTANT: Robinhood's `get_portfolio` tool only returns account-level
        aggregates (total_value, cash, buying_power, etc.) — it has NEVER
        included a per-symbol `positions` array (confirmed against real
        captured responses in logs/direct_mcp_trace.jsonl). The actual
        per-symbol holdings live behind the separate `get_equity_positions`
        tool. This method now fetches both and merges them, so
        `Portfolio.positions` is always the real, live account state
        instead of silently being an empty list forever.

        A failure fetching positions never breaks the aggregate portfolio
        fetch — it just leaves `.positions` empty for this call, exactly
        as before, so existing safety nets (never reconcile against an
        empty positions list) keep working.
        """
        account_number = await self._get_account_number()
        args = {"account_number": account_number} if account_number else {}
        content = await self._call_tool_with_retry("get_portfolio", args)
        parsed = self._extract_json_payload(content)

        if isinstance(parsed, dict) and "data" in parsed:
            portfolio_data = parsed["data"]
        else:
            portfolio_data = parsed if isinstance(parsed, dict) else {}

        portfolio_data = self._coerce_portfolio_fields(portfolio_data)
        portfolio = Portfolio.model_validate(portfolio_data)

        try:
            portfolio.positions = await self.get_equity_positions()
        except Exception as exc:
            self.logger.warning("get_portfolio_positions_merge_failed: %s", exc)

        return portfolio

    async def get_equity_quotes(self, tickers: list[str]) -> list[EquityQuote]:
        """Fetch latest quotes for up to 20 tickers. Not account-scoped."""
        if not tickers:
            return []
        chunk = tickers[:20]
        content = await self._call_tool_with_retry("get_equity_quotes", {"symbols": chunk})
        parsed = self._extract_json_payload(content)

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

    async def get_equity_historicals(
        self, ticker: str, span: str = "3month", interval: str = "day"
    ) -> list[OHLCVBar]:
        """Fetch OHLCV historical bars for a ticker. Not account-scoped.

        NOTE: the real MCP schema has no "span" concept — it requires an
        explicit start_time/end_time RFC3339 range. This method converts
        the familiar span keyword into that range internally so the
        public signature stays identical for existing callers.
        """
        delta = _SPAN_TO_TIMEDELTA.get(span, timedelta(days=90))
        end_time = datetime.now(timezone.utc)
        start_time = end_time - delta

        args = {
            "symbols": [ticker],
            "start_time": start_time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "end_time": end_time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "interval": interval,
        }
        content = await self._call_tool_with_retry("get_equity_historicals", args)
        parsed = self._extract_json_payload(content)

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

    async def get_equity_historicals_batch(
        self, tickers: list[str], span: str = "3month", interval: str = "day"
    ) -> dict[str, list[OHLCVBar]]:
        """Fetch OHLCV bars for up to 10 tickers in a single call.

        Returns a dict mapping ticker -> list[OHLCVBar]. Callers with more
        than 10 tickers should chunk and call this multiple times.
        """
        if not tickers:
            return {}
        if len(tickers) > 10:
            raise ValueError("get_equity_historicals_batch accepts at most 10 tickers per call")

        delta = _SPAN_TO_TIMEDELTA.get(span, timedelta(days=90))
        end_time = datetime.now(timezone.utc)
        start_time = end_time - delta

        args = {
            "symbols": tickers,
            "start_time": start_time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "end_time": end_time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "interval": interval,
        }
        content = await self._call_tool_with_retry("get_equity_historicals", args)
        parsed = self._extract_json_payload(content)

        results_by_ticker: dict[str, list[OHLCVBar]] = {t: [] for t in tickers}
        if isinstance(parsed, dict) and "data" in parsed:
            results = parsed["data"].get("results", [])
            for result in results:
                symbol = result.get("symbol", "")
                bar_list = result.get("bars", [])
                mapped_bars = []
                for item in bar_list:
                    mapped_bars.append({
                        "ticker": symbol,
                        "timestamp": item.get("begins_at") or item.get("timestamp", ""),
                        "open": float(item.get("open_price") or item.get("open") or 0),
                        "high": float(item.get("high_price") or item.get("high") or 0),
                        "low": float(item.get("low_price") or item.get("low") or 0),
                        "close": float(item.get("close_price") or item.get("close") or 0),
                        "volume": float(item.get("volume") or 0),
                    })
                results_by_ticker[symbol] = [OHLCVBar.model_validate(m) for m in mapped_bars]
        return results_by_ticker

    async def get_equity_tradability(self, ticker: str) -> dict[str, Any]:
        """Fetch tradability metadata for a ticker. Account-scoped.

        Returns the per-symbol result dict directly (e.g. containing
        'tradeable', 'state', 'account_type_tradabilities', etc.), not
        the outer {"data": {"results": [...]}} envelope Robinhood
        actually returns.
        """
        account_number = await self._get_account_number()
        args = {"symbols": [ticker]}
        if account_number:
            args["account_number"] = account_number
        content = await self._call_tool_with_retry("get_equity_tradability", args)
        parsed = self._extract_json_payload(content)

        if isinstance(parsed, dict):
            results = parsed.get("data", {}).get("results", [])
            if isinstance(results, list) and results:
                for item in results:
                    if isinstance(item, dict) and item.get("symbol", "").upper() == ticker.upper():
                        return item
                return results[0] if isinstance(results[0], dict) else {}
        return {}

    async def _resolve_watchlist_id(self, watchlist_name: str) -> str:
        """Resolve a watchlist's display name to its list_id UUID."""
        cache_attr = f"_watchlist_id_cache"
        if not hasattr(self, cache_attr):
            setattr(self, cache_attr, {})
        cache = getattr(self, cache_attr)
        if watchlist_name in cache:
            return cache[watchlist_name]

        content = await self._call_tool_with_retry("get_watchlists", {})
        parsed = self._extract_json_payload(content)

        watchlists = []
        if isinstance(parsed, dict):
            watchlists = (
                parsed.get("data", {}).get("results")
                or parsed.get("data", {}).get("watchlists")
                or parsed.get("results")
                or []
            )
        elif isinstance(parsed, list):
            watchlists = parsed

        list_id = ""
        for wl in watchlists:
            if not isinstance(wl, dict):
                continue
            name = wl.get("name") or wl.get("title") or wl.get("display_name") or ""
            if name.strip().lower() == watchlist_name.strip().lower():
                list_id = wl.get("list_id") or wl.get("id") or wl.get("uuid") or ""
                break

        if not list_id:
            self.logger.warning(
                "watchlist_name_not_resolved name=%s raw_watchlists=%s",
                watchlist_name, watchlists,
            )
        cache[watchlist_name] = list_id
        return list_id

    async def get_watchlist_tickers(self, watchlist_name: str = "Default") -> list[str]:
        """Fetch ticker symbols from a named Robinhood watchlist."""
        try:
            list_id = await self._resolve_watchlist_id(watchlist_name)
            if not list_id:
                return []

            content = await self._call_tool_with_retry("get_watchlist_items", {"list_id": list_id})
            parsed = self._extract_json_payload(content)

            if isinstance(parsed, list):
                return [str(t).upper().strip() for t in parsed if t]
            if isinstance(parsed, dict):
                results = parsed.get("data", {}).get("results", parsed.get("data", {}).get("items", parsed.get("results", [])))
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
            self.logger.warning("watchlist_fetch_failed name=%s error=%s", watchlist_name, exc)
            return []

    async def get_equity_positions(self) -> list[Position]:
        """Fetch current open positions as Position models. Account-scoped.

        THIS IS THE SINGLE PLACE real per-symbol holdings enter the system.
        `get_portfolio` (above) never carries them — this is a separate tool.

        Field names are mapped defensively (several candidate keys per
        field, same pattern already proven in get_equity_quotes) because
        this tool was previously never called end-to-end, so its exact
        response shape has not been observed live yet. Every call is
        still captured to logs/direct_mcp_trace.jsonl by
        _call_tool_with_retry — check that file after the first live run
        and extend the candidate-key tuples below if a field comes back
        as 0/empty that shouldn't be.

        A single malformed row never drops the whole list: each item is
        parsed independently so one bad symbol can't hide every other
        real position from the caller.
        """
        account_number = await self._get_account_number()
        args = {"account_number": account_number} if account_number else {}
        content = await self._call_tool_with_retry("get_equity_positions", args)
        parsed = self._extract_json_payload(content)

        if isinstance(parsed, dict) and "data" in parsed:
            items = parsed["data"].get("results", parsed["data"].get("positions", []))
        elif isinstance(parsed, list):
            items = parsed
        else:
            items = []

        def _first(d: dict[str, Any], *keys: str, default: Any = 0) -> Any:
            for k in keys:
                if k in d and d[k] not in (None, ""):
                    return d[k]
            return default

        def _f(v: Any) -> float:
            try:
                return float(v)
            except (TypeError, ValueError):
                return 0.0

        positions: list[Position] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            try:
                ticker = str(_first(item, "symbol", "ticker", "instrument_symbol", default="")).upper().strip()
                if not ticker:
                    self.logger.warning("get_equity_positions_skip_no_ticker item_keys=%s", list(item.keys()))
                    continue
                quantity = _f(_first(item, "quantity", "shares", "quantity_available"))
                avg_cost = _f(_first(item, "average_buy_price", "avg_cost", "average_cost", "cost_basis_per_share"))
                current_price = _f(_first(item, "current_price", "last_trade_price", "price", "mark_price"))
                market_value = _f(_first(item, "market_value", "equity", "current_value"))
                if market_value == 0.0 and current_price > 0 and quantity > 0:
                    market_value = current_price * quantity
                unrealized_pct_raw = _first(item, "unrealized_pnl_pct", "percent_change", "total_return_percent", default=None)
                if unrealized_pct_raw is not None:
                    unrealized_pnl_pct = _f(unrealized_pct_raw)
                elif avg_cost > 0 and current_price > 0:
                    unrealized_pnl_pct = ((current_price - avg_cost) / avg_cost) * 100.0
                else:
                    unrealized_pnl_pct = 0.0
                sector = str(_first(item, "sector", default=""))

                positions.append(Position.model_validate({
                    "ticker": ticker,
                    "quantity": quantity,
                    "avg_cost": avg_cost,
                    "current_price": current_price,
                    "market_value": market_value,
                    "unrealized_pnl_pct": unrealized_pnl_pct,
                    "sector": sector,
                }))
            except Exception as exc:
                self.logger.warning("get_equity_positions_row_skipped item_keys=%s error=%s", list(item.keys()), exc)
                continue

        self.logger.info("get_equity_positions_parsed count=%d", len(positions))
        return positions

    async def get_earnings_calendar(self) -> dict[str, Any]:
        """Fetch the upcoming earnings calendar directly from Robinhood's
        own MCP tool (not account-scoped)."""
        content = await self._call_tool_with_retry("get_earnings_calendar", {})
        parsed = self._extract_json_payload(content)
        return parsed if isinstance(parsed, dict) else {"data": {}}

    async def get_equity_fundamentals(self, tickers: list[str]) -> dict[str, dict]:
        """Fetch fundamentals for up to 10 tickers from Robinhood's own
        MCP tool. Not account-scoped. Returns a dict keyed by ticker symbol,
        each value being the raw fundamentals dict for that symbol."""
        if not tickers:
            return {}
        chunk = tickers[:10]
        content = await self._call_tool_with_retry("get_equity_fundamentals", {"symbols": chunk})
        parsed = self._extract_json_payload(content)
        results: dict[str, dict] = {}
        if isinstance(parsed, dict):
            data_block = parsed.get("data", {})
            found_list = data_block.get("results", [])
            not_found_list = data_block.get("not_found", [])
            for sym in not_found_list:
                self.logger.warning("fundamentals_not_found symbol=%s", sym)
            for item in found_list:
                sym = item.get("symbol", "")
                if sym:
                    results[sym] = item
        return results

    async def get_technical_indicator(
        self,
        symbol: str,
        indicator_type: str,
        interval: str,
        start_time: str,
        end_time: str | None = None,
        output: str = "latest",
        **extra_params: Any,
    ) -> dict:
        """Fetch a technical indicator from Robinhood's MCP tool.
        Not account-scoped. Returns the parsed JSON response as-is."""
        args: dict[str, Any] = {
            "symbol": symbol,
            "type": indicator_type,
            "interval": interval,
            "start_time": start_time,
            "output": output,
        }
        if end_time is not None:
            args["end_time"] = end_time
        for key, value in extra_params.items():
            if value is not None:
                args[key] = value
        try:
            content = await self._call_tool_with_retry("get_equity_technical_indicators", args)
            parsed = self._extract_json_payload(content)
            return parsed if isinstance(parsed, dict) else {}
        except Exception as exc:
            self.logger.warning(
                "technical_indicator_fetch_failed symbol=%s type=%s error=%s",
                symbol, indicator_type, exc,
            )
            return {}

    async def get_earnings_calendar_window(
        self, start_date: str, days: int, filter: str | None = None,
    ) -> dict:
        """Fetch a windowed earnings calendar from Robinhood's MCP tool.
        Not account-scoped. Returns parsed response as-is."""
        args: dict[str, Any] = {"start_date": start_date, "days": days}
        if filter is not None:
            args["filter"] = filter
        content = await self._call_tool_with_retry("get_earnings_calendar", args)
        parsed = self._extract_json_payload(content)
        return parsed if isinstance(parsed, dict) else {"data": {}}

    async def get_realized_pnl(self, span: str = "3month") -> dict:
        """Fetch realized P&L from Robinhood's MCP tool. Account-scoped."""
        account_number = await self._get_account_number()
        args: dict[str, Any] = {"account_number": account_number, "span": span}
        content = await self._call_tool_with_retry("get_realized_pnl", args)
        parsed = self._extract_json_payload(content)
        return parsed if isinstance(parsed, dict) else {}

    async def get_pnl_trade_history(
        self, span: str = "week", symbol: str | None = None,
    ) -> dict:
        """Fetch P&L trade history from Robinhood's MCP tool. Account-scoped."""
        account_number = await self._get_account_number()
        args: dict[str, Any] = {"account_number": account_number, "span": span}
        if symbol is not None:
            args["symbol"] = symbol
        content = await self._call_tool_with_retry("get_pnl_trade_history", args)
        parsed = self._extract_json_payload(content)
        return parsed if isinstance(parsed, dict) else {}

    async def get_equity_tax_lots(self, symbol: str) -> dict:
        """Fetch equity tax lots from Robinhood's MCP tool. Account-scoped."""
        account_number = await self._get_account_number()
        args: dict[str, Any] = {"account_number": account_number, "symbol": symbol}
        content = await self._call_tool_with_retry("get_equity_tax_lots", args)
        parsed = self._extract_json_payload(content)
        return parsed if isinstance(parsed, dict) else {}

    async def review_equity_order(
        self, ticker: str, side: str, quantity: float, order_type: str = "market"
    ) -> dict[str, Any]:
        """Preview an order; returns pre-trade warnings without placing it.
        Account-scoped."""
        account_number = await self._get_account_number()
        args = {
            "symbol": ticker,
            "side": side,
            "type": order_type,
            "quantity": f"{quantity:.6f}".rstrip("0").rstrip(".") or "0",
        }
        if account_number:
            args["account_number"] = account_number
        content = await self._call_tool_with_retry("review_equity_order", args)
        parsed = self._extract_json_payload(content)
        return parsed.get("data", {}) if isinstance(parsed, dict) else {}

    async def place_equity_order(
        self,
        ticker: str,
        side: str,
        quantity: float,
        order_type: str = "market",
        dry_run: bool = True,
    ) -> ExecutionResult:
        """Submit or simulate an equity order. Account-scoped.

        Args:
            dry_run: If True, calls review_equity_order only and never
                places a live order.
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

        account_number = await self._get_account_number()
        args = {
            "symbol": ticker,
            "side": side,
            "type": order_type,
            "quantity": f"{quantity:.6f}".rstrip("0").rstrip(".") or "0",
        }
        if account_number:
            args["account_number"] = account_number

        content = await self._call_tool_with_retry("place_equity_order", args)
        parsed = self._extract_json_payload(content)

        # Defensive extraction — MCP returns nested shapes that do not
        # match ExecutionResult fields directly.
        data = parsed.get("data", parsed) if isinstance(parsed, dict) else {}
        order = data.get("order", data) if isinstance(data, dict) else {}

        order_id = (
            order.get("id")
            or order.get("order_id")
            or data.get("id")
            or data.get("order_id")
            or ""
        )
        fill_price = float(
            order.get("average_price")
            or order.get("price")
            or order.get("filled_price")
            or data.get("average_price")
            or data.get("price")
            or 0.0
        )
        status_raw = str(
            order.get("state")
            or order.get("status")
            or data.get("state")
            or data.get("status")
            or "SUBMITTED"
        ).upper()

        if status_raw in ("FILLED", "COMPLETED", "EXECUTED"):
            status = "EXECUTED"
        elif status_raw in ("QUEUED", "CONFIRMED", "UNCONFIRMED", "SUBMITTED", ""):
            status = "SUBMITTED"
        else:
            status = status_raw or "SUBMITTED"

        return ExecutionResult(
            ticker=ticker,                    # always inject — never trust the payload
            order_id=str(order_id),
            status=status,
            fill_price=fill_price,
            quantity=quantity,
            timestamp=datetime.now(timezone.utc).isoformat(),
            dry_run=False,
            side=side,
        )