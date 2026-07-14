"""
mcp_deep_schema_dump.py — read-only diagnostic, see module docstring below
for full details on what this inspects and what it refuses to call.
"""

import sys
from pathlib import Path

# This MUST be the first executable code in the file, before any local
# project imports. It guarantees the project root (where this script and
# the robinhood_mcp/ package both live) is at the very front of the
# import search path, regardless of anything else that could interfere.
_PROJECT_ROOT = str(Path(__file__).resolve().parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import asyncio
import json
import traceback
from datetime import datetime, timezone
from typing import Any

from robinhood_mcp.robinhood_client import RobinhoodMCPClient

# ---------------------------------------------------------------------------
# FORBIDDEN_TOOLS — hardcoded set of tool names that must NEVER be called
# by this script, under any circumstance. Asserted before every tool call.
# ---------------------------------------------------------------------------
FORBIDDEN_TOOLS: set[str] = {
    "place_equity_order",
    "place_option_order",
    "cancel_equity_order",
    "cancel_option_order",
    "review_equity_order",
    "review_option_order",
    "add_to_watchlist",
    "remove_from_watchlist",
    "create_watchlist",
    "update_watchlist",
    "add_option_to_watchlist",
    "remove_option_from_watchlist",
    "follow_watchlist",
    "unfollow_watchlist",
    "create_scan",
    "update_scan_config",
    "update_scan_filters",
}

# ---------------------------------------------------------------------------
# TARGET_TOOLS — exactly this list, in this order, are the tools this script
# will dump schemas for and attempt one safe live call against.
# ---------------------------------------------------------------------------
TARGET_TOOLS: list[str] = [
    "get_equity_fundamentals",
    "get_equity_technical_indicators",
    "get_equity_tax_lots",
    "get_pnl_trade_history",
    "get_realized_pnl",
    "get_index_quotes",
    "get_indexes",
    "get_equity_orders",
    "get_equity_positions",
    "get_earnings_calendar",
    "get_accounts",
    "get_popular_watchlists",
    "get_option_watchlist",
]

# ---------------------------------------------------------------------------
# Tracking for final summary
# ---------------------------------------------------------------------------
_schema_ok: list[str] = []
_schema_missing: list[str] = []
_live_ok: list[str] = []
_live_failed: list[str] = []
_skipped_forbidden: list[str] = []
_skipped_no_args: list[str] = []
_skipped_not_found: list[str] = []


def _build_safe_args(
    schema: dict[str, Any],
    account_number: str,
) -> tuple[dict[str, Any] | None, str | None]:
    """Return (safe_args, skip_reason).

    If safe_args can be built from the schema + known safe defaults, returns
    (args_dict, None). If required fields cannot be reasonably inferred,
    returns (None, "explanation string").

    Rules (applied in order):
      1. If schema has "account_number" as a required property, include it.
      2. If schema has "symbols" as a required property (array), pass ["AAPL"].
      3. If schema has "symbol" as a required property (string), pass "AAPL".
      4. If schema has NO required properties, return {}.
      5. If schema has required properties that are NOT covered by the above,
         return (None, reason) — do NOT guess.
    """
    properties: dict[str, Any] = schema.get("properties", {})
    required: list[str] = schema.get("required", [])

    if not required:
        # No required fields — safe to call with empty args
        return {}, None

    args: dict[str, Any] = {}

    for field in required:
        if field == "account_number":
            if account_number:
                args["account_number"] = account_number
            else:
                return None, (
                    f"Required field 'account_number' but no account number "
                    f"could be resolved"
                )
        elif field == "symbols":
            args["symbols"] = ["AAPL"]
        elif field == "symbol":
            args["symbol"] = "AAPL"
        else:
            return None, (
                f"Required field '{field}' cannot be auto-filled with a safe "
                f"default (not account_number / symbols / symbol)"
            )

    return args, None


async def main() -> None:
    print("#" * 70)
    print("# MCP DEEP SCHEMA DUMP — READ-ONLY DIAGNOSTIC")
    print(f"# Run started: {datetime.now(timezone.utc).isoformat()}")
    print("#" * 70)
    print(f"# Forbidden tools (NEVER called): {len(FORBIDDEN_TOOLS)}")
    print(f"# Target tools to inspect:         {len(TARGET_TOOLS)}")
    print("#" * 70)

    # -----------------------------------------------------------------------
    # Step 1: Import + instantiate RobinhoodMCPClient
    # -----------------------------------------------------------------------
    try:
        from robinhood_mcp.robinhood_client import RobinhoodMCPClient
    except ImportError as exc:
        print(f"\n[FATAL] Could not import RobinhoodMCPClient: {exc}")
        print("Make sure you're running from inside the robinhood-agent directory")
        print("with the virtual environment activated.")
        return

    client = RobinhoodMCPClient()

    # -----------------------------------------------------------------------
    # Step 2: Get a valid access token (reuses cached refresh token silently
    #         if still valid; opens browser only as last resort)
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("STEP 1 — Ensuring valid OAuth token (cached refresh if possible)")
    print("=" * 70)
    try:
        access_token = await client._ensure_valid_token()
        print(f"[OK] Access token obtained (length={len(access_token)})")
    except Exception as exc:
        print(f"[FATAL] Could not obtain access token: {type(exc).__name__}: {exc}")
        print(traceback.format_exc())
        return

    # -----------------------------------------------------------------------
    # Step 3: Resolve the real agentic account number
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("STEP 2 — Resolving agentic account number")
    print("=" * 70)
    try:
        account_number = await client._get_account_number()
        if account_number:
            print(f"[OK] Agentic account number resolved: "
                  f"{account_number[:4]}...{account_number[-4:] if len(account_number) > 8 else account_number}")
        else:
            print("[WARN] Account number could NOT be resolved — tools requiring "
                  "'account_number' will be skipped for live calls")
    except Exception as exc:
        print(f"[WARN] Account resolution failed: {type(exc).__name__}: {exc}")
        print("       Tools requiring 'account_number' will be skipped for live calls")
        account_number = ""

    # -----------------------------------------------------------------------
    # Step 4: Open ONE single MCP session and keep it open for all tools
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("STEP 3 — Opening single MCP session for the entire diagnostic run")
    print("=" * 70)

    try:
        from mcp.client.streamable_http import streamablehttp_client
        from mcp import ClientSession
    except ImportError as exc:
        print(f"[FATAL] Could not import real mcp SDK: {exc}")
        print(f"        sys.path[:5] = {sys.path[:5]}")
        return

    headers = {"Authorization": f"Bearer {access_token}"}

    try:
        async with streamablehttp_client(
            "https://agent.robinhood.com/mcp/trading",
            headers=headers,
            timeout=30,
        ) as (read, write, _sid):
            async with ClientSession(read, write) as session:
                await session.initialize()
                print("[OK] MCP session initialized successfully")

                # -----------------------------------------------------------
                # Step 5: list_tools() once, filter to TARGET_TOOLS
                # -----------------------------------------------------------
                print("\n" + "=" * 70)
                print("STEP 4 — Discovering tools via session.list_tools()")
                print("=" * 70)
                tools_result = await session.list_tools()
                all_tool_names = sorted(t.name for t in tools_result.tools)
                print(f"Total tools on server: {len(all_tool_names)}")
                for name in all_tool_names:
                    marker = ""
                    if name in FORBIDDEN_TOOLS:
                        marker = " [FORBIDDEN — will NEVER be called]"
                    elif name in TARGET_TOOLS:
                        marker = " [TARGET]"
                    print(f"  - {name}{marker}")

                # Build a map of tool_name -> Tool object for target tools
                tool_map: dict[str, Any] = {}
                for t in tools_result.tools:
                    if t.name in TARGET_TOOLS:
                        tool_map[t.name] = t

                # Check which target tools are missing from the server
                for target_name in TARGET_TOOLS:
                    if target_name not in tool_map:
                        _skipped_not_found.append(target_name)
                        print(f"\n[NOTE] Target tool '{target_name}' not found in "
                              f"server tool list — skipping")

                # -----------------------------------------------------------
                # Step 6: For each target tool that IS present, dump schema
                #         + attempt one safe live call
                # -----------------------------------------------------------
                for tool_name in TARGET_TOOLS:
                    if tool_name not in tool_map:
                        continue  # already logged above

                    tool_obj = tool_map[tool_name]

                    # ----- SAFETY ASSERTION -----
                    if tool_name in FORBIDDEN_TOOLS:
                        _skipped_forbidden.append(tool_name)
                        print(f"\n{'=' * 70}")
                        print(f"TOOL: {tool_name}")
                        print(f"{'=' * 70}")
                        print("[SKIPPED FOR SAFETY] This tool is in FORBIDDEN_TOOLS "
                              "and will NOT be called under any circumstance.")
                        continue

                    print(f"\n{'=' * 70}")
                    print(f"TOOL: {tool_name}")
                    print(f"{'=' * 70}")

                    # --- 6a: Dump the full inputSchema ---
                    raw_schema = getattr(tool_obj, "inputSchema", None)
                    if raw_schema is None:
                        print("[WARN] No inputSchema found on this tool object")
                        _schema_missing.append(tool_name)
                    else:
                        try:
                            schema_dict = (
                                raw_schema.model_dump()
                                if hasattr(raw_schema, "model_dump")
                                else raw_schema
                            )
                            print("INPUT SCHEMA:")
                            print(json.dumps(schema_dict, indent=2))
                            _schema_ok.append(tool_name)
                        except Exception as schema_exc:
                            print(f"[WARN] Could not serialize schema: "
                                  f"{type(schema_exc).__name__}: {schema_exc}")
                            _schema_missing.append(tool_name)
                            schema_dict = None

                    # --- 6b: Build safe args for a live call ---
                    if raw_schema is None:
                        print("[SKIP LIVE CALL] No schema available to build args from")
                        _skipped_no_args.append(tool_name)
                        continue

                    try:
                        schema_for_args = (
                            raw_schema.model_dump()
                            if hasattr(raw_schema, "model_dump")
                            else raw_schema
                        )
                    except Exception:
                        schema_for_args = {}

                    safe_args, skip_reason = _build_safe_args(
                        schema_for_args, account_number
                    )

                    if safe_args is None:
                        print(f"[SKIP LIVE CALL] {skip_reason}")
                        _skipped_no_args.append(tool_name)
                        continue

                    # --- 6c: Attempt the live call ---
                    print(f"\nAttempting live call with args: "
                          f"{json.dumps(safe_args)}")
                    try:
                        result = await session.call_tool(tool_name, safe_args)
                        content_dump = [
                            c.model_dump() if hasattr(c, "model_dump") else str(c)
                            for c in result.content
                        ]
                        raw_text = json.dumps(content_dump, indent=2)
                        if len(raw_text) > 3000:
                            print(f"LIVE RESPONSE (first 3000 chars of "
                                  f"{len(raw_text)} total):")
                            print(raw_text[:3000])
                            print(f"... [truncated, {len(raw_text) - 3000} "
                                  f"more chars]")
                        else:
                            print("LIVE RESPONSE:")
                            print(raw_text)
                        _live_ok.append(tool_name)
                    except Exception as live_exc:
                        print(f"LIVE CALL FAILED: {type(live_exc).__name__}: "
                              f"{live_exc}")
                        _live_failed.append(tool_name)

    except Exception as session_exc:
        print(f"\n[FATAL] MCP session-level error: "
              f"{type(session_exc).__name__}: {session_exc}")
        print(traceback.format_exc())
        # Still print whatever summary we have so far

    # -----------------------------------------------------------------------
    # Step 7: Final summary
    # -----------------------------------------------------------------------
    print("\n" + "#" * 70)
    print("# FINAL SUMMARY")
    print("#" * 70)

    print(f"\nSchema successfully retrieved ({len(_schema_ok)}):")
    for name in _schema_ok:
        print(f"  [SCHEMA OK]     {name}")

    if _schema_missing:
        print(f"\nSchema missing / failed to dump ({len(_schema_missing)}):")
        for name in _schema_missing:
            print(f"  [SCHEMA MISS]   {name}")

    print(f"\nLive call succeeded ({len(_live_ok)}):")
    for name in _live_ok:
        print(f"  [LIVE OK]       {name}")

    if _live_failed:
        print(f"\nLive call failed ({len(_live_failed)}):")
        for name in _live_failed:
            print(f"  [LIVE FAIL]     {name}")

    print(f"\nSkipped — forbidden (never called) ({len(_skipped_forbidden)}):")
    for name in _skipped_forbidden:
        print(f"  [SKIP FORBIDDEN] {name}")

    if _skipped_no_args:
        print(f"\nSkipped — couldn't build safe args ({len(_skipped_no_args)}):")
        for name in _skipped_no_args:
            print(f"  [SKIP NO ARGS]  {name}")

    if _skipped_not_found:
        print(f"\nSkipped — not found in server tool list "
              f"({len(_skipped_not_found)}):")
        for name in _skipped_not_found:
            print(f"  [SKIP ABSENT]   {name}")

    print("\n" + "-" * 70)
    print(f"Run finished: {datetime.now(timezone.utc).isoformat()}")
    print("-" * 70)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\nInterrupted by user.")
    except Exception as exc:
        print(f"\n\nUNEXPECTED TOP-LEVEL ERROR: {type(exc).__name__}: {exc}")
        print(traceback.format_exc())