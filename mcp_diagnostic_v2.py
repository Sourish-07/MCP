"""
mcp_diagnostic_v2.py — Completes the real OAuth flow and proves an
authenticated, direct Python connection to Robinhood's MCP works end to end.

WHAT CHANGED FROM v1:
  - FIXED the naming collision: your own project has a folder named `mcp/`
    (robinhood-agent/mcp/robinhood_client.py) which was shadowing the real
    installed `mcp` PyPI package when run from inside that folder. This
    script explicitly strips the current directory from sys.path BEFORE
    importing mcp, so the real SDK loads correctly regardless of where you
    run it from.
  - IMPLEMENTS the full OAuth 2.1 + PKCE + Dynamic Client Registration flow
    discovered in the previous run's Phase 3, using the exact endpoints
    Robinhood's server advertised:
      registration_endpoint: https://agent.robinhood.com/oauth/trading/register
      authorization_endpoint: https://robinhood.com/oauth
      token_endpoint: https://api.robinhood.com/oauth2/token/
  - CACHES the resulting token to a local file (.rh_oauth_cache.json) so you
    only need to complete the browser login once; subsequent runs reuse the
    refresh token automatically.
  - Then attempts the same Phase 4/5 as before: list all tools, make ONE
    read-only get_portfolio call, using ONLY this token — zero Claude Code,
    zero subprocess, zero LLM tokens spent anywhere in this path.

SAFETY: identical guarantees as v1. FORBIDDEN_TOOLS (all order-placing tools)
are never called under any circumstance. This script is read-only.

HOW TO RUN:
  1. cd C:\\Users\\Shris\\OneDrive\\Desktop\\Project\\S-BOT\\robinhood-agent
  2. venv\\Scripts\\activate
  3. Save this file as mcp_diagnostic_v2.py in that folder
  4. python mcp_diagnostic_v2.py
  5. A browser window will open asking you to log into Robinhood and approve
     access — this is the SAME kind of one-time approval you did for Claude
     Code originally, just now for a plain Python script instead.
  6. After approving in the browser, come back to the terminal — the script
     will detect the redirect automatically and continue.
  7. Copy the FULL terminal output and paste it back into chat.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import os
import secrets
import sys
import threading
import traceback
import webbrowser
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import urlencode, urlparse, parse_qs

# ---------------------------------------------------------------------------
# CRITICAL FIX: remove the current script's directory (and cwd) from sys.path
# BEFORE importing mcp, so the real installed PyPI package is found instead
# of this project's own local mcp/ folder.
# ---------------------------------------------------------------------------
_this_dir = str(Path(__file__).resolve().parent)
sys.path = [p for p in sys.path if os.path.abspath(p or ".") != _this_dir]
if "" in sys.path:
    sys.path.remove("")

ROBINHOOD_MCP_URL = "https://agent.robinhood.com/mcp/trading"
REGISTRATION_ENDPOINT = "https://agent.robinhood.com/oauth/trading/register"
AUTHORIZATION_ENDPOINT = "https://robinhood.com/oauth"
TOKEN_ENDPOINT = "https://api.robinhood.com/oauth2/token/"
REDIRECT_PORT = 53271  # arbitrary local port for the OAuth redirect catcher
REDIRECT_URI = f"http://localhost:{REDIRECT_PORT}/callback"

TOKEN_CACHE_PATH = Path(__file__).resolve().parent / ".rh_oauth_cache.json"

FORBIDDEN_TOOLS = {
    "place_equity_order", "place_option_order",
    "cancel_equity_order", "cancel_option_order",
    "review_equity_order", "review_option_order",
}

results: dict[str, dict] = {}


def log_phase(phase: str, status: str, detail: str = "") -> None:
    results[phase] = {"status": status, "detail": detail}
    marker = {"PASS": "[PASS]", "FAIL": "[FAIL]", "BLOCKED": "[BLOCKED]", "INFO": "[INFO]"}.get(status, "[??]")
    print(f"\n{marker} {phase}")
    if detail:
        print(f"       {detail}")


def redact(value: str, keep: int = 6) -> str:
    if not value:
        return "(empty)"
    if len(value) <= keep * 2:
        return "*" * len(value)
    return f"{value[:keep]}...{'*' * 8}...{value[-keep:]}"


# ---------------------------------------------------------------------------
# PKCE helpers
# ---------------------------------------------------------------------------
def generate_pkce_pair() -> tuple[str, str]:
    """Return (code_verifier, code_challenge) per RFC 7636 using S256."""
    code_verifier = base64.urlsafe_b64encode(secrets.token_bytes(64)).decode("ascii").rstrip("=")
    digest = hashlib.sha256(code_verifier.encode("ascii")).digest()
    code_challenge = base64.urlsafe_b64encode(digest).decode("ascii").rstrip("=")
    return code_verifier, code_challenge


# ---------------------------------------------------------------------------
# Local HTTP server to catch the OAuth redirect
# ---------------------------------------------------------------------------
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
            body = b"<html><body><h2>Authorization received. You can close this tab and return to the terminal.</h2></body></html>"
            self.send_response(200)
        else:
            _CallbackState.error = qs.get("error", ["unknown_error"])[0]
            body = b"<html><body><h2>Authorization failed. Check the terminal for details.</h2></body></html>"
            self.send_response(400)
        self.send_header("Content-Type", "text/html")
        self.end_headers()
        self.wfile.write(body)
        _CallbackState.received.set()

    def log_message(self, format, *args):  # noqa: A002
        pass  # suppress default request logging


def _run_callback_server() -> None:
    server = HTTPServer(("localhost", REDIRECT_PORT), _CallbackHandler)
    server.timeout = 180
    server.handle_request()  # blocks for exactly one request, then returns


# ---------------------------------------------------------------------------
# PHASE A — Dynamic client registration
# ---------------------------------------------------------------------------
async def register_client() -> dict | None:
    import httpx

    print("\n" + "=" * 70)
    print("PHASE A — Dynamic client registration")
    print("=" * 70)

    payload = {
        "client_name": "robinhood-agent-direct-python-test",
        "redirect_uris": [REDIRECT_URI],
        "grant_types": ["authorization_code", "refresh_token"],
        "response_types": ["code"],
        "token_endpoint_auth_method": "none",
    }

    try:
        async with httpx.AsyncClient(timeout=20) as client:
            resp = await client.post(REGISTRATION_ENDPOINT, json=payload)
            if resp.status_code in (200, 201):
                data = resp.json()
                client_id = data.get("client_id")
                log_phase(
                    "Phase A: dynamic client registration", "PASS",
                    f"Registered successfully. client_id={redact(client_id, keep=8) if client_id else '(none returned)'}"
                )
                return data
            else:
                log_phase(
                    "Phase A: dynamic client registration", "FAIL",
                    f"HTTP {resp.status_code}: {resp.text[:500]}"
                )
                return None
    except Exception as exc:
        log_phase("Phase A: dynamic client registration", "FAIL", f"{type(exc).__name__}: {exc}")
        return None


# ---------------------------------------------------------------------------
# PHASE B — Full authorization_code + PKCE flow (opens browser)
# ---------------------------------------------------------------------------
async def run_oauth_flow(client_id: str) -> dict | None:
    import httpx

    print("\n" + "=" * 70)
    print("PHASE B — Browser authorization flow")
    print("=" * 70)

    code_verifier, code_challenge = generate_pkce_pair()
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

    print(f"\nStarting local callback listener on {REDIRECT_URI} ...")
    server_thread = threading.Thread(target=_run_callback_server, daemon=True)
    server_thread.start()

    print("Opening your browser now. Log into Robinhood and approve access if prompted.")
    print(f"(If the browser doesn't open automatically, visit this URL manually:)\n{auth_url}\n")
    webbrowser.open(auth_url)

    print("Waiting up to 180 seconds for you to complete the login in the browser...")
    got_response = _CallbackState.received.wait(timeout=185)

    if not got_response:
        log_phase("Phase B: browser authorization", "FAIL", "Timed out waiting for browser redirect after 180s.")
        return None

    if _CallbackState.error:
        log_phase("Phase B: browser authorization", "FAIL", f"Robinhood returned an error: {_CallbackState.error}")
        return None

    if _CallbackState.state != state:
        log_phase(
            "Phase B: browser authorization", "FAIL",
            f"State mismatch (possible security issue or stale callback): "
            f"expected {state}, got {_CallbackState.state}"
        )
        return None

    log_phase("Phase B: browser authorization", "PASS", "Authorization code received successfully.")

    # Exchange code for tokens
    print("\nExchanging authorization code for access token...")
    token_payload = {
        "grant_type": "authorization_code",
        "code": _CallbackState.code,
        "redirect_uri": REDIRECT_URI,
        "client_id": client_id,
        "code_verifier": code_verifier,
    }
    try:
        async with httpx.AsyncClient(timeout=20) as client:
            resp = await client.post(
                TOKEN_ENDPOINT,
                data=token_payload,
                headers={"Content-Type": "application/x-www-form-urlencoded"},
            )
            if resp.status_code == 200:
                token_data = resp.json()
                log_phase(
                    "Phase B: token exchange", "PASS",
                    f"Access token received. access_token={redact(token_data.get('access_token', ''))} "
                    f"refresh_token={'present' if token_data.get('refresh_token') else 'NOT present'} "
                    f"expires_in={token_data.get('expires_in', 'unknown')}s"
                )
                token_data["_client_id"] = client_id
                return token_data
            else:
                log_phase(
                    "Phase B: token exchange", "FAIL",
                    f"HTTP {resp.status_code}: {resp.text[:500]}"
                )
                return None
    except Exception as exc:
        log_phase("Phase B: token exchange", "FAIL", f"{type(exc).__name__}: {exc}")
        return None


async def refresh_token_flow(refresh_token: str, client_id: str) -> dict | None:
    import httpx

    print("\nAttempting to reuse cached refresh token instead of re-opening the browser...")
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
            if resp.status_code == 200:
                token_data = resp.json()
                token_data["_client_id"] = client_id
                log_phase("Phase B (cached): refresh token exchange", "PASS", "Refreshed successfully, no browser needed.")
                return token_data
            else:
                log_phase(
                    "Phase B (cached): refresh token exchange", "FAIL",
                    f"Cached refresh token rejected (HTTP {resp.status_code}), will fall back to full browser flow."
                )
                return None
    except Exception as exc:
        log_phase("Phase B (cached): refresh token exchange", "FAIL", f"{exc}, falling back to full browser flow.")
        return None


# ---------------------------------------------------------------------------
# PHASE C — Connect to MCP with the Bearer token, list tools, one safe call
# ---------------------------------------------------------------------------
async def phase_c_authenticated_mcp_call(access_token: str) -> None:
    print("\n" + "=" * 70)
    print("PHASE C — Authenticated direct MCP connection")
    print("=" * 70)

    try:
        from mcp.client.streamable_http import streamablehttp_client
        from mcp import ClientSession
    except ImportError as exc:
        log_phase(
            "Phase C: import streamable_http client", "FAIL",
            f"Still could not import the real mcp SDK client: {exc}\n"
            f"       sys.path currently: {sys.path[:5]}..."
        )
        return

    headers = {"Authorization": f"Bearer {access_token}"}

    try:
        async with streamablehttp_client(ROBINHOOD_MCP_URL, headers=headers, timeout=30) as (read, write, _sid):
            async with ClientSession(read, write) as session:
                init_result = await session.initialize()
                log_phase(
                    "Phase C: authenticated connection + initialize", "PASS",
                    f"Connected WITH a real access token, zero Claude Code involved. "
                    f"Server info: {getattr(init_result, 'serverInfo', init_result)}"
                )

                tools_result = await session.list_tools()
                discovered_names = sorted(t.name for t in tools_result.tools)
                print(f"\nTools discovered: {len(discovered_names)}")
                for name in discovered_names:
                    print(f"  - {name}")

                log_phase(
                    "Phase C: tool listing", "PASS",
                    f"{len(discovered_names)} tools discovered via direct authenticated connection."
                )

                if "get_portfolio" in discovered_names:
                    try:
                        if "get_accounts" in discovered_names:
                            print("\nFetching account number first (required by get_portfolio)...")
                            accounts_result = await session.call_tool("get_accounts", {})
                            accounts_dump = [
                                c.model_dump() if hasattr(c, "model_dump") else str(c)
                                for c in accounts_result.content
                            ]
                            print(f"get_accounts raw response: {json.dumps(accounts_dump, indent=2)[:1500]}")

                            # Try to extract account_number from whatever shape comes back
                            account_number = None
                            for item in accounts_dump:
                                text = item.get("text", "") if isinstance(item, dict) else ""
                                if text:
                                    try:
                                        parsed = json.loads(text)
                                        if isinstance(parsed, dict):
                                            accounts_list = parsed.get("data", {}).get("accounts", [])
                                            if isinstance(accounts_list, list) and accounts_list:
                                                # Explicitly prefer the account marked agentic_allowed=true.
                                                # Do NOT fall back to is_default or the first account —
                                                # the default account is the user's primary individual
                                                # account, not the isolated Agentic trading account, and
                                                # using the wrong one would return the wrong portfolio data.
                                                agentic_account = next(
                                                    (a for a in accounts_list if a.get("agentic_allowed") is True),
                                                    None
                                                )
                                                if agentic_account:
                                                    account_number = agentic_account.get("account_number")
                                                    print(f"Selected agentic account: nickname="
                                                          f"{agentic_account.get('nickname', '(none)')}, "
                                                          f"type={agentic_account.get('type')}")
                                                else:
                                                    print("WARNING: no account with agentic_allowed=true "
                                                          "was found in the response. Refusing to guess "
                                                          "which account to use.")
                                    except (json.JSONDecodeError, AttributeError) as parse_exc:
                                        print(f"Could not parse get_accounts response: {parse_exc}")

                            print(f"Extracted account_number: {redact(account_number) if account_number else 'COULD NOT EXTRACT — see raw response above'}")
                        else:
                            account_number = None

                        print("\nMaking ONE read-only call: get_portfolio ...")
                        call_args = {"account_number": account_number} if account_number else {}
                        result = await session.call_tool("get_portfolio", call_args)
                        content_dump = [
                            c.model_dump() if hasattr(c, "model_dump") else str(c)
                            for c in result.content
                        ]
                        log_phase(
                            "Phase C: live get_portfolio call", "PASS",
                            f"REAL portfolio data received directly from Robinhood via plain "
                            f"Python — zero Claude Code, zero LLM tokens spent on this call:\n"
                            f"{json.dumps(content_dump, indent=2)[:3000]}"
                        )
                    except Exception as _portfolio_exc:
                        log_phase(
                            "Phase C: live get_portfolio call", "FAIL",
                            f"Failed to extract account_number or call get_portfolio.\n"
                            f"Error: {type(_portfolio_exc).__name__}: {_portfolio_exc}\n"
                            f"Raw get_accounts response was printed above (if available).\n"
                            f"{traceback.format_exc()}"
                        )
                else:
                    log_phase("Phase C: live get_portfolio call", "BLOCKED", "get_portfolio not in discovered tool list.")

                print(f"\nSafety check — forbidden tools present on server but NEVER called: "
                      f"{FORBIDDEN_TOOLS & set(discovered_names)}")

    except Exception as exc:
        log_phase(
            "Phase C: authenticated connection", "FAIL",
            f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"
        )


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
async def main() -> None:
    print("#" * 70)
    print("# ROBINHOOD DIRECT-PYTHON OAUTH + MCP DIAGNOSTIC (v2)")
    print(f"# Run started: {datetime.now(timezone.utc).isoformat()}")
    print("#" * 70)

    # Try cached token first
    token_data = None
    if TOKEN_CACHE_PATH.exists():
        try:
            cached = json.loads(TOKEN_CACHE_PATH.read_text(encoding="utf-8"))
            if cached.get("refresh_token") and cached.get("_client_id"):
                token_data = await refresh_token_flow(cached["refresh_token"], cached["_client_id"])
        except Exception as exc:
            print(f"Could not use token cache: {exc}")

    if token_data is None:
        reg_data = await register_client()
        if reg_data is None or "client_id" not in reg_data:
            log_phase("Overall", "FAIL", "Could not register a client, cannot proceed to browser auth.")
        else:
            token_data = await run_oauth_flow(reg_data["client_id"])

    if token_data and token_data.get("access_token"):
        try:
            TOKEN_CACHE_PATH.write_text(json.dumps(token_data, indent=2), encoding="utf-8")
            print(f"\nToken cached to {TOKEN_CACHE_PATH} for reuse on next run (refresh_token saved).")
        except Exception as exc:
            print(f"Could not write token cache: {exc}")

        await phase_c_authenticated_mcp_call(token_data["access_token"])
    else:
        log_phase("Overall", "FAIL", "No access token obtained, cannot test authenticated MCP connection.")

    print("\n" + "#" * 70)
    print("# FINAL SUMMARY")
    print("#" * 70)
    for phase, info in results.items():
        print(f"  [{info['status']:8s}] {phase}")

    print("\n" + "-" * 70)
    if results.get("Phase C: live get_portfolio call", {}).get("status") == "PASS":
        print("VERDICT: FULLY CONFIRMED. Direct Python OAuth + MCP access to Robinhood")
        print("works end to end, with zero Claude Code and zero LLM tokens involved.")
        print("Option 3 (rebuild robinhood_client.py around this) is fully validated.")
    else:
        print("VERDICT: Did not reach a fully authenticated live call. See FAIL/BLOCKED")
        print("phases above for exactly where it stopped. Paste full output for diagnosis.")
    print("-" * 70)
    print(f"\nRun finished: {datetime.now(timezone.utc).isoformat()}")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\nInterrupted by user.")
    except Exception as exc:
        print(f"\n\nUNEXPECTED TOP-LEVEL ERROR: {type(exc).__name__}: {exc}")
        print(traceback.format_exc())