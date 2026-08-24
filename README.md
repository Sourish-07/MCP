# robinhood-agent

An **Anthropic-only, MCP-native, autonomous equity trading agent** for a Robinhood **Agentic** brokerage account. The system runs scheduled decision cycles throughout the trading day, reads live portfolio/quotes/technical data directly from Robinhood's MCP server over Streamable HTTP, drives all judgment through Claude (Sonnet/Haiku), enforces hard risk rules deterministically in code, and executes orders through the same MCP server. Built for research, experimentation, and education — **not** for unsupervised live capital.

---

## What It Is

`robinhood-agent` is a scheduled, agentic trading loop. Three times per day (at market open, midday, and near close) it wakes up, pulls the current account state and live market data, feeds each watchlist ticker to an LLM portfolio manager (Claude), turns the model's structured JSON verdict into sizing + risk checks, and then either simulates or places the trade. A separate **1-minute price monitor** continuously checks already-open positions against hard stop-loss / take-profit / profit-review thresholds so risk is enforced even between the three scheduled reviews.

Key design decisions baked into the code:

- **Anthropic only.** The legacy OpenAI-based routing/extraction stack was removed entirely. Every model call goes through `utils/anthropic_client.py`.
- **MCP-native data.** Market data and broker actions come from Robinhood's official trading MCP server via the `mcp` Python SDK — a direct connection that replaced an earlier, expensive `claude --print` shelling approach. No LLM tokens are spent fetching data.
- **Deterministic risk guardrails.** Stop-loss, take-profit, max positions, position sizing, and safe execution windows are enforced in plain Python (`core/execution.py`, `core/position_manager.py`), never left to the model.
- **DRY_RUN enforced in code.** Even if DRY_RUN isn't set, execution logic treats missing config as dry-run; live order placement only happens when DRY_RUN is explicitly false.
- **LLM cost budgeting.** Every model usage is metered and persisted (`utils/cost_tracker.py`) with hard daily/monthly ceilings.

---

## High-Level Architecture

```
                    ┌─────────────────────────────────────────────┐
                    │                  main.py                    │
                    │     (TradingAgent — APScheduler loop)       │
                    │  OPEN / MID / CLOSE  +  1-min monitor       │
                    └─────────────────────────────────────────────┘
                          │                │                │
                          ▼                ▼                ▼
             ┌──────────────┐       ┌──────────────┐   ┌──────────────┐
             │DataIngestLayer│       │ NewsFetcher  │   │MetricsEngine  │
             └──────────────┘       └──────────────┘   └──────────────┘
                          │                │                │
                          └──────┬─────────┴───────┬────────┘
                                 ▼                 ▼
                    ┌───────────────────┐  ┌──────────────────┐
                    │   DecisionEngine   │◄─ │  AnthropicClient │
                    └───────────────────┘  └──────────────────┘
                                 │ TradeDecision + edge score
                                 ▼
                    ┌───────────────────┐
                    │   ExecutionEngine  │  edge→size, windows, dry-run
                    └───────────────────┘
                                 │
                    ┌────────────▼─────────────┐
                    │ PositionManager + MCP     │
                    │  (positions.json, exits)  │
                    └──────────────────────────┘
```

**Data flow summary:** ingest → enrich (metrics, news) → decide (LLM) → validate/size (code) → execute (MCP) → journal + persist. Exit signals from the monitor or the cycle flow back through `review_exit` (a monitor-style HOLD/SELL prompt) and the same execution path.

---

## Directory Structure

```
robinhood-agent/
├── main.py                     # Entry point. TradingAgent: config, scheduler, cycles, monitor.
├── README.md                   # This file.
├── requirements.txt            # Dependency pin list (note: UTF-16 encoded).
├── .env                        # Secrets (FINNHUB_API_KEY, ANTHROPIC_API_KEY, DRY_RUN…). Not committed.
├── mcp_diagnostic_v2.py        # Standalone read-only OAuth+MCP connectivity probe.
├── mcp_deep_schema_dump.py     # Enumerates MCP tool schemas once (development aid).
├── .claude/settings.local.json # Claude Code allow-list of Robinhood MCP tools (legacy CLI).
│
├── config/settings.json        # Tunables: models, trading, schedule, watchlist, news, costs.
│
├── core/                       # Business logic layer.
│   ├── data_ingest.py          # Phase-1 ingestion: portfolio + quotes + historical bars.
│   ├── metrics.py              # RSI/MACD/BB/ATR/fundamentals (RH MCP) + local returns/vol.
│   ├── news_fetch.py           # Finnhub news fetch, dedup, freshness, Haiku summary/fallback.
│   ├── decision_engine.py      # LLM prompts + JSON parsing → TradeDecision / ExitDecision.
│   ├── execution.py            # Tradeable + size + window + dry-run → place/simulate order.
│   ├── position_manager.py     # positions.json, reconcile, exit signals, realized P&L.
│   └── pnl_tracker.py          # Realized P&L + trade-history report builder.
│
├── models/                     # Pydantic data contracts (shared across layers).
│   ├── decisions.py            # DecisionType, CycleType, EdgeScore, TradeDecision, …
│   ├── market_data.py          # OHLCVBar, MarketMetrics, EquityQuote.
│   ├── news.py                 # NewsItem (Finnhub schema).
│   └── portfolio.py            # PositionRecord, Position, Portfolio.
│
├── robinhood_mcp/
│   ├── robinhood_client.py                 # Direct MCP client (OAuth) — ACTIVE.
│   ├── robinhood_client(claude version).py # Older CLI subprocess client — legacy/unused.
│   └── response_recorder.py    # Observability: traces/filters MCP responses.
│
├── utils/
│   ├── anthropic_client.py     # Async Anthropic wrapper: retry + cost metering.
│   ├── cost_tracker.py         # Singleton cost tracker + budget guardrails.
│   ├── earnings_calendar.py    # Cached earnings-window lookups (informational only).
│   ├── journal_manager.py      # Per-ticker rolling journals for model memory.
│   └── logger.py               # UTC console/file logs + JSON trade log config.
│
├── data/
│   ├── earnings_calendar.json  # Cached upcoming RH earnings.
│   └── seen_headlines.json     # Finnhub article-id dedup state for the day.
│
├── journals/                   # Per-ticker history (TICKER.json) + Inactive/.
└── logs/                       # Runtime output & state (auto-created).
    ├── agent.log               # Main DEBUG log (UTC).
    ├── trades.json             # JSON array of every trade attempt.
    ├── positions.json          # Open + closed records with realized P&L.
    ├── cost_tracker.json       # Persisted cost records/totals.
    └── direct_mcp_trace.jsonl  # Recorded direct MCP calls (observability).
```

---

## The Trading Day: Cycles + Monitor

The agent is driven by an APScheduler (`AsyncIOScheduler`) with cron jobs on **US/Eastern** market hours. All times below are Eastern.

| Job | Time | What happens |
|-----|------|--------------|
| **OPEN** | 09:35 | Highest-conviction new entries. Full weight given to the historical price/metric trajectory (journal) + earnings context. Also reviews clear exit signals. |
| **MID** | 12:30 | Intraday reassessment. Raises the bar for *new* positions vs OPEN (higher edge required / prefer HOLD/IGNORE). |
| **CLOSE** | 15:30 | Never opens new positions. Only HOLD / SELL, or set up tomorrow's thesis in the reasoning summary. |
| **Monitor** | every 60s (in window) | Price-threshold monitoring of already-open positions for stop-loss / take-profit / profit-review breaches. |

The three cycle types are modelled as `CycleType` (`OPEN`, `MID`, `CLOSE`) in `models/decisions.py`.

### What happens each cycle (`run_cycle`)

1. **Load config + resolve watchlist.** Watchlist first attempts the Robinhood watchlist named `"Default"`; falls back to `config/settings.json`'s `default_tickers`.
2. **Ingest** (`DataIngestLayer.ingest`): fetch the live `Portfolio`, equity quotes, and ~3 months of daily historical bars for the union of watchlist + open positions (deduped, capped at 20 symbols).
3. **Enrich:**
   - `MetricsEngine.compute` per ticker — RSI, MACD histogram, Bollinger z-score, ATR, SMA-overrun, fundamentals (market cap / P/E / dividend / sector / industry) directly from Robinhood MCP, plus locally-computed 5d/30d/90d returns, 30d drawdown, and 7d/30d realized volatility.
   - `NewsFetcher.fetch_news` — Finnhub company news per ticker + general-market news (attached to `SPY`/`MARKET`), ID-deduplicated for the day, freshness-filtered, capped per ticker.
   - `EarningsCalendar.get_upcoming` — informational earnings-window flag.
4. **Decide** (`DecisionEngine`): for each non-held watchlist ticker → `make_decision(...)` (LLM → `TradeDecision`). For held positions that tripped an exit signal → `review_exit(...)` (LLM → HOLD/SELL).
5. **Establish portfolio context:** effective buying power is "freed" as SELLs execute, positions counts and P&L maps are updated in-loop so later tickers in the same cycle see the effect of earlier fills.
6. **Execute** (`ExecutionEngine.execute`): validate against safe windows, size the position from edge, respect DRY_RUN, place/simulate the order.
7. **Journal:** append a `JournalEntry` to the ticker's journal for the cycle (`JournalManager.append`), recording bull/bear thesis, decision, rationale, key metrics, news, fill price, unrealized P&L, and earnings flag.

### The 1-minute monitor (`run_price_monitor`)

Every 60s during market hours the monitor:

1. Loads open position records (`PositionManager.get_open_records`).
2. Refreshes portfolio + quotes only (`DataIngestLayer.ingest_quotes_only`, no historical bars to save API/time).
3. Runs `PositionManager.check_exits` to produce `ExitSignal`s:
   - `STOP_LOSS` — hard: unrealized ≤ stop_loss_pct (−7%). **Forced SELL.**
   - `PROFIT_TARGET` — soft: unrealized ≥ take_profit_pct (20%). Editable per record.
   - `PROFIT_REVIEW` — advisory: unrealized ≥ 15% (`REVIEW_GAIN_PCT`).
   - `MAX_HOLDING_PERIOD` — advisory: days held > 15.
4. For each signal, calls a lightweight, **monitor-only** system prompt (`_MONITOR_SYSTEM_PROMPT_TEXT`, ~300 tokens) via `review_exit` that produces a HOLD/SELL verdict with an EXIT-CONVICTION edge score. This compact prompt (rather than the long decision prompt) keeps repeated stop-loss/profit reviews cheap.
5. STOP_LOSS is never overridden: its decision is forced to SELL by the code path.
6. Executable SELLs flow through the same `ExecutionEngine`; realized P&L is recorded before the position record is dropped.

### Threshold evaluation (`_evaluate_threshold_signal`)

The monitor computes price levels from each open record's `entry_price`:
- `stop_px = entry * (1 + stop_loss_pct)`, `tp_px = entry * (1 + take_profit_pct)`.
- Determines the appropriate signal type from the current quote, then hands it to the exit-review path. `_monitor_time_window` guards when the monitor is allowed to act (e.g., not at extreme open/close auction windows).

### Monitor scheduling (exact)

Two cron jobs drive the monitor, both Mon–Fri ET: `hour="9-15", minute="*"` (every minute 09:00–15:59) plus `16:00`. Both use `max_instances=1, coalesce=True`. Inside each tick, `_monitor_time_window(now_et)` classifies the time as `MARKET` / `PRE` / `POST` / `CLOSED` and the monitor only acts within the appropriate window. Per-ticker state (`_last_monitor_review`) throttles re-evaluating side-sensitive SOFT signals (PROFIT_TARGET / PROFIT_REVIEW) after a HOLD so the LLM isn't hit every tick; STOP_LOSS ignores that throttle entirely.

### Agent startup behavior (`start()`)

1. `configure_logging()` and log the effective `dry_run`.
2. Register the 3 cycle cron jobs + the monitor cron jobs, then `scheduler.start()`.
3. **Reconcile on boot** — fetch the live portfolio and run `position_manager.reconcile_with_mcp` (skipped if the portfolio pull is empty/missing `positions`, per the safety rule).
4. **Smart startup catch-up** — if the agent is started mid-day it immediately runs the appropriate cycle for the current time: before OPEN's slot → wait for OPEN; after OPEN's slot → run `OPEN` now; after MID's slot → run `MID` now; otherwise run `CLOSE` now. Weekends / after-hours simply wait for the next trading day.
5. Enter an idle `asyncio.sleep(60)` loop until shutdown (Ctrl-C).

---

## How a Decision Is Made

Every ticker decision is produced by `DecisionEngine`, which builds a rich prompt and parses a strict JSON response. Two entry points exist:

### `make_decision` — new-position assessment (watchlist tickers)

Used for tickers the account does **not** currently hold. It assembles:

- Any `SPY` market context line (when present) so the model can gauge regime.
- The `CYCLE` type (OPEN / MID / CLOSE), which changes the model's mandate (see schedule table).
- An **earnings window block** (`EarningsCalendar.earnings_window_for_prompt`) — informational catalyst context, never a ban.
- A **journal block** (`JournalManager.summarise_for_prompt`) — the last ~15 price/metric snapshots for the ticker. Crucially, past **decisions** (BUY/IGNORE/etc.) are deliberately **omitted** so the model cannot anchor on its own prior verdicts; only the price/metric trajectory is shown.
- `TICKER`, `CURRENT PRICE`, `NEWS HEADLINES` (JSON), `MARKET METRICS` (JSON), `FUNDAMENTALS`.
- `ACCOUNT` line: total value, effective buying power, positions count, `max_positions=12`.
- `OTHER OPEN POSITIONS` roster (`PositionManager.other_positions_line`) showing each held ticker + its live P&L %.
- The `EXISTING POSITION` detail block (only when a position record is supplied) — entry price, quantity, unrealized %, days held, stop/take-profit.
- `WEAKEST CURRENT POSITION` — which held name has the worst P&L (for ROTATE decisions).

The model must answer the **core forward-looking test**: *what specific condition — news-driven OR technical — would make this stock HIGHER in 1–2 weeks, and has that condition happened (priced in) or is it still developing?* A thesis that only explains past strength is explicitly rejected as a valid `bull_thesis`. Valid forward catalysts include both dated news events **and** technical/price-action setups (e.g., consolidation + a 3–8% low-volume dip with no bad news = selling exhaustion / mean-reversion).

The model returns a `TradeDecision`: `decision` (`BUY`/`SELL`/`HOLD`/`IGNORE`/`ROTATE`), `bull_thesis`, `bear_thesis`, `failure_conditions`, a structured `edge` score, `position_size_pct`, optional `rotate_from_ticker`, plus `risk_notes` / `reasoning_summary`.

### `review_exit` — HOLD/SELL for open positions (monitor + cycles)

Used when a held position trips a threshold (stop-loss / profit-target / profit-review / max-holding) or on a cycle exit review. It uses the compact monitor prompt (`_MONITOR_SYSTEM_PROMPT_TEXT`) and a short `_EXIT_REVIEW_RULES` rubric:

- The model MUST state entry vs current price and unrealized %.
- `SELL` needs explicit evidence (thesis broken, technical failure vs plan, or capital better deployed).
- For EXIT decisions, edge is scored as **exit conviction**, not buy quality: `catalyst_strength` = strength of the exit reason (0–2), `technical_confirmation` = price supports exit (0–2), `portfolio_fit` = value of freeing capital / cutting risk (0–1).
- `HOLD` is valid when unrealized is small and the thesis is intact (including holding through earnings).
- `STOP_LOSS` remains a forced SELL via the existing code path.
- Earnings proximity is informational only — never an automatic SELL.

Output is an `ExitDecision` (ticker, decision, reasoning, edge_score). The code enforces that a STOP_LOSS signal always resolves to SELL regardless of the model's answer.

---

## Edge Scoring

`StructuredEdgeScore` replaces the old arbitrary 0–5 float. Claude fills **three bounded components** and the **code computes the total** — the model cannot fake the number.

| Component | Range | Meaning |
|-----------|-------|---------|
| `catalyst_strength` | 0 – 2 | Real news catalyst specific to the company: 0 none, 1 minor/unclear, 2 strong. |
| `technical_confirmation` | 0 – 2 | Price/volume support: 0 against trend, 1 neutral, 2 strong confirmation. |
| `portfolio_fit` | 0 – 1 | Improves portfolio: 0 redundant/risky, 0.5 neutral, 1 clearly improves. |
| **`total`** | 0 – 5 | Computed by `compute_total()` = sum of the three, clamped. |

The decision engine's system prompt teaches calibrated mapping: e.g., a strong technical mean-reversion setup alone can score technical_confirmation 1.5–1.7 plus portfolio_fit 0.7–0.9 to clear the threshold, while a pure "structural demand is strong" claim with no dated event and no qualifying technical pattern is mathematically unable to reach the threshold — by design.

### Position sizing from edge (`ExecutionEngine._size_from_edge`)

| Edge total `total` | Position size (% of portfolio value) |
|--------------------|--------------------------------------|
| `>= 4.5` | 8% |
| `>= 4.0` | 6% |
| `>= 3.5` | 4% |
| `>= 3.0` | 3% |
| `< 3.0` | 0% (no position) |

The trade must clear `trading.edge_score_threshold` (3.0) to open a position.

---

## Risk Controls

Risk is enforced deterministically in code, independent of the model's judgment:

### Position management (`core/position_manager.py`, `logs/positions.json`)

- **Position records**: each opened position stores `entry_price`, `entry_date`, `entry_edge_score`, `stop_loss_pct` (default `-0.07`), `take_profit_pct` (default `0.20`), `quantity`, `order_id`.
- **Reconciliation** (`reconcile_with_mcp`): synchronises local `positions.json` with live MCP holdings. **Safety rule:** if the MCP portfolio pull is empty/None, local records are left untouched (a failed/empty pull must never wipe records). Records are only removed when the live response is non-empty and the ticker is genuinely absent (or qty ≤ 0). Quantity mismatches are adjusted to live values.
- **Exit signals** (`check_exits`): as described in the monitor section — STOP_LOSS (hard, forced), PROFIT_TARGET (soft), PROFIT_REVIEW (advisory), MAX_HOLDING_PERIOD (advisory). Signals may combine (e.g., `PROFIT_REVIEW+MAX_HOLDING_PERIOD`).
- **Realized P&L**: on a SELL/ROTATE fill, `record_realized_pnl` logs the closed trade (`realized_pct`, `closed_at`) before the open record is removed (`record_exit`), and open positions are marked with live unrealized P&L.

### Execution safeguards (`core/execution.py`)

- **DRY_RUN enforced in code**: `effective_dry_run = settings.env.DRY_RUN(True) OR param dry_run`. Live orders require an explicitly false DRY_RUN.
- **Safe execution windows** (`_is_safe_execution_window`): blocks trades in the volatile open/close auctions — blocks `09:30–09:34` and `15:50–15:59` ET (weekends always allowed → returns True). Blocked → `SKIPPED_TIME_WINDOW`.
- **Non-trades skipped**: IGNORE / HOLD → `SKIPPED`.
- **Tradeability check**: before placing, the client checks `get_equity_tradability` and `review_equity_order`; invalid inputs → `REJECTED` statuses.
- **Position limits**: `max_positions` (12) and per-position $ cap (`max_position_pct` 8%) are respected; opening beyond limits → `REJECTED_LIMITS`.
- **Budget guard**: cost overruns throw `CostLimitExceededError` (see Cost Tracking).

### Portfolio/config guardrails (`config/settings.json`)

| Setting | Default | Purpose |
|---------|---------|---------|
| `max_positions` | 12 | Hard cap on concurrent open names. |
| `max_position_pct` | 0.08 | Per-position cap on portfolio value. |
| `edge_score_threshold` | 3.0 | Minimum edge to open a position. |
| `stop_loss_pct` | -0.07 | Forced-loss threshold. |
| `profit_review_pct` | 0.15 | Advisory review trigger (also `REVIEW_GAIN_PCT`). |
| `max_holding_days` | 15 | Advisory max-holding trigger. |
| `portfolio_size` | 2000 | Reference account size (used for reporting/inputs). |
| `cost_limits.daily_usd_limit` | 3.00 | Daily model-cost budget. |
| `cost_limits.monthly_usd_limit` | 50.00 | Monthly model-cost budget. |

**Note on intervals:** the hard stop-loss (`-7%`) and default take-profit (`20%`) thresholds are defaults on each `PositionRecord`; the `Position`/`PositionRecord` models expose them per-position, and exit checks use the per-record values.

---

## The MCP Client & OAuth

`robinhood_mcp/robinhood_client.py` is the **direct** Python bridge to Robinhood's trading MCP server. It is an architectural replacement for an earlier version that shelled out to `claude --print "…"` for every call (which cost ~45k input tokens of pure CLI overhead per call).

**Transport.** Uses the official `mcp` SDK over **Streamable HTTP** against `https://agent.robinhood.com/mcp/trading`.

**Authentication — OAuth 2.1 + PKCE + Dynamic Client Registration.**
- **Dynamic Client Registration** (RFC 7591) against the registration endpoint obtains a `client_id` (cached). Grant types: `authorization_code` + `refresh_token`.
- **Browser auth flow** (`_run_browser_auth_flow`): builds an authorization URL with a PKCE code challenge (`S256`), opens the browser, and catches the redirect on a local callback server (`http://localhost:53271/callback` in a background thread).
- **Token cache** (`.rh_oauth_cache.json`): access + refresh tokens persisted in the project root. Subsequent runs reuse the cached refresh token and **silently refresh** (`_refresh_access_token`) via the token endpoint; only if there's no valid refresh token does it re-open the browser.
- `_ensure_valid_token` is called before every MCP session opens; callers never manage token lifecycle.

**Session lifecycle.** A **fresh authenticated session is opened per tool call** (`_call_tool_with_retry`) and closed after — deliberate, to avoid stale-connection failures across the multi-hour gaps between OPEN/MID/CLOSE cycles. Retries are handled for transient failures (`max_retries = 3`).

**Account selection.** Account-scoped tools require an `account_number`. `_get_account_number()` calls `get_accounts` and — critically — selects the account where `agentic_allowed=true` (the Agentic account), *not* the default individual account. It caches the chosen number; if none is found it refuses to guess (safer than trading the wrong account).

**Why `Portfolio.positions` is populated.** Robinhood's `get_portfolio` tool returns only account-level aggregates (total value, cash, buying power) and never a per-symbol positions array. The client therefore also calls `get_equity_positions` and **merges** the two so `Portfolio.positions` reflects real live holdings (a failure there leaves `.positions` empty, preserving the safety net that prevents wiping local records on an empty pull).

**Response cleanup helpers** coerce Robinhood's string-typed numerics into floats (`_coerce_portfolio_fields`), unwrap nested `buying_power`, and remap quote/historicals fields into the Pydantic models — identical business logic to the validated legacy version.

**Legacy file.** `robinhood_mcp/robinhood_client(claude version).py` is the old CLI-subprocess implementation and is **no longer used**.

---

## Broker (MCP) Tool Coverage

The client wraps each broker MCP tool as an async Python method that returns a Pydantic model / parsed dict:

- `get_portfolio()` → `Portfolio` (merged with positions) — account-scoped.
- `get_equity_quotes(tickers)` → `list[EquityQuote]` — not account-scoped.
- `get_equity_historicals(ticker, span, interval)` → `list[OHLCVBar]`, plus `get_equity_historicals_batch(...)`.
- `get_equity_tradability(ticker)` → tradability metadata — account-scoped.
- `get_equity_positions()` → `list[Position]` — account-scoped.
- `get_watchlist_tickers(watchlist_name)` → symbols from a named watchlist.
- `get_earnings_calendar()` / `get_earnings_calendar_window(start_date, days)` → upcoming earnings.
- `get_equity_fundamentals(tickers)` → market cap / P/E / dividend / sector/industry.
- `get_technical_indicator(symbol, indicator, …)` → RSI / MACD / Bollinger / ATR / SMA series used by `MetricsEngine`.
- `get_realized_pnl(span)` / `get_pnl_trade_history(span)` / `get_equity_tax_lots(symbol)` → P&L and history — account-scoped.
- `review_equity_order(...)`, `place_equity_order(...)`, and cancel/option equivalents → order lifecycle.

Every call result flows through `robinhood_mcp/response_recorder.py` (`capture_direct_mcp_call`), which preserves observability (schema drift detection, numeric anomaly detection, credential redaction) without the old subprocess machinery.

> The `.claude/settings.local.json` allow-list lists these `mcp__robinhood-trading__*` tools for the legacy Claude Code CLI permissions layer; the running system no longer depends on it.

---

## Persistence & Logging

All state lives under `logs/` and `journals/` (auto-created by the app).

- **`logs/positions.json`** — The authoritative open/closed position ledger (`PositionManager`). Contains open records, per-position live marks (unrealized P&L), and a `closed_trades` list recording `realized_pct` / `closed_at` on each sale. This is the single source of truth for "what else is held" and is used to build the `OTHER OPEN POSITIONS` roster for prompts.
- **`logs/trades.json`** — A JSON array of every trade attempt (decision, edge score, position size %, action type, status, fill price, quantity, dry-run flag, reasoning summary). Written by `ExecutionEngine._log_trade`.
- **`journals/TICKER.json`** — Per-ticker rolling history of `JournalEntry` objects (`JournalManager`, max 90 entries, atomic `.tmp`→replace writes). Each entry records timestamp, cycle type, bull/bear thesis, decision, rationale, key metrics snapshot, news used, fill price, unrealized P&L, and earnings flag. Used to build the model's price/metric trajectory (decisions deliberately omitted).
- **`logs/cost_tracker.json`** — Persisted per-call cost records plus rollup daily/monthly totals (`CostTracker`).
- **`logs/agent.log`** — Main structured DEBUG log (UTC timestamps via `logger.py`).
- **`logs/performance.json`, `logs/attribution_report_*.json`** — Performance snapshots and optional on-demand attribution reports.
- **`logs/direct_mcp_trace.jsonl`, `logs/mcp_response_trace.jsonl`, `logs/schema_history.json`** — MCP observability / schema-drift history from `response_recorder.py`.
- **`data/earnings_calendar.json`** — Cached upcoming earnings (from RH MCP) read by the informational earnings-window logic.
- **`data/seen_headlines.json`** — Finnhub article-id dedup state so news isn't re-shown to the model every cycle of the same day.

**Logging** (`utils/logger.py`): a `robinhood-agent` logger routes to console (INFO) + `agent.log` (DEBUG), and a separate `robinhood-agent.trades` logger streams JSON lines to `logs/trades.json` (note: `ExecutionEngine` also writes `trades.json` directly). All times are UTC.

---

## Cost Tracking

`utils/cost_tracker.py` meters every Anthropic call — a singleton, so totals are shared process-wide and persisted.

**Pricing (per token, hard-coded):**

| Model | Base input | Cache write (5m) | Cache read | Output |
|-------|-----------|------------------|-----------|--------|
| Haiku | $1.00/M | $1.25/M | $0.10/M | $5.00/M |
| Sonnet | $3.00/M | $3.75/M | $0.30/M | $15.00/M |

Unknown models default to Sonnet pricing. `record()` computes cost from regular (non-cached) input + cache-creation + cache-read + output, appends a `CostRecord`, updates daily/monthly totals, and persists.

**Budget guardrails** (`check_limits`): configured daily ($3.00) and monthly ($50.00) ceilings. It logs a warning when either is exceeded and **raises `CostLimitExceededError` when daily spend reaches 2× the daily limit** (fail-fast). The `user` message flow surfaces cost-limit errors so the trading loop can halt cleanly instead of silently racking up spend.

**Retry** (`utils/anthropic_client.py`): completion attempts retry up to 3× on rate-limit/529-style errors (with a 5s sleep). It strips markdown code fences from responses and omits the `temperature` parameter for newer model generations that reject it (e.g., `claude-sonnet-5`). Both `system` (string or cached content blocks) and `user` payloads are passed through to `messages.create`.

---

## Configuration

### `config/settings.json` (primary)

```jsonc
{
  "models": {
    "decision": "claude-sonnet-5",   // used for the decision engine & exit reviews
    "cheap": "claude-haiku-4-5"       // used for cheap news summarization/fallback
  },
  "trading": {
    "max_positions": 12,              // hard cap on open names
    "max_position_pct": 0.08,         // per-position cap on portfolio value
    "edge_score_threshold": 3.0,      // min edge to open a position
    "stop_loss_pct": -0.07,           // default forced-loss level
    "profit_review_pct": 0.15,        // advisory profit-review trigger
    "max_holding_days": 15,           // advisory max-holding trigger
    "portfolio_size": 2000,           // reference account size
    "dry_run": false                  // NOTE: the real gate is env.DRY_RUN
  },
  "schedule": {
    "open_hour": 9,   "open_minute": 35,     // OPEN cycle
    "mid_hour": 12,   "mid_minute": 30,      // MID cycle
    "close_hour": 15, "close_minute": 30,    // CLOSE cycle
    "timezone": "America/New_York"
  },
  "watchlist": {
    "default_tickers": ["JPM","NVDA","AVGO","CAT","MU","TSM","UNH","XOM","AMD","COST"],
    "robinhood_watchlist_name": "Default"
  },
  "cost_limits": { "daily_usd_limit": 3.00, "monthly_usd_limit": 50.00 },
  "news": {
    "provider": "finnhub",
    "lookback_days": 3,
    "max_per_ticker": 5,
    "fresh_window_hours": 24,
    "general_market": { "enabled": true, "category": "general", "count": 8 },
    "haiku_summary":  { "enabled": false, "fetch_cap": 20, "max_summary_sentences": 5 },
    "haiku_fallback": { "enabled": false, "trigger_when_no_news": true }
  }
}
```

### `.env` (secrets / environment)

- `FINNHUB_API_KEY` — for `NewsFetcher` (company + general market news).
- `ANTHROPIC_API_KEY` — for `anthropic.AsyncAnthropic` (the model provider).
- `DRY_RUN` — **the primary live-trading gate.** Resolution: env `DRY_RUN` if set, else `config/settings.json` → `trading.dry_run`, then `.lower() != "false"`. An explicit `DRY_RUN=false` is what enables live order placement; `ExecutionEngine` also forces dry-run via an OR with its own dry-run flag (default true), so a missing/ambiguous value never enables live trading accidentally.
- `ROBINHOOD_MCP_URL` — optional explicit MCP endpoint override (default is the well-known `https://agent.robinhood.com/mcp/trading` in the client).

> There is currently **no `.env.example`** in the repo (the legacy README referenced one); the values above must be provided in a `.env` you create.

### Initialization order (`TradingAgent.__init__`)

`AsyncIOScheduler(timezone="America/New_York")` → `_load_config()` (loads `config/settings.json` + `.env`) → `CostTracker()` (singleton) → `RobinhoodMCPClient()` → `DataIngestLayer` → `MetricsEngine` → `NewsFetcher(AnthropicClient(cheap model), FINNHUB_API_KEY, settings)` → `JournalManager` → `EarningsCalendar()` (+ `set_client(mcp_client)`) → `DecisionEngine(AnthropicClient(decision model), journal_manager, earnings_calendar, settings)` → `PositionManager` → `PnLTracker(mcp_client)` → `ExecutionEngine(mcp_client, position_manager, settings)`. Monitor state locks/caches are then initialised.

**DRY_RUN resolution (exact):** `_load_config` computes `dry_run = os.getenv("DRY_RUN", str(settings.trading.dry_run)).lower() != "false"`. So `.env`'s `DRY_RUN` wins if present; if unset it falls back to `config/settings.json` → `trading.dry_run`. `ExecutionEngine` further enforces `effective_dry_run = bool(settings.env.DRY_RUN, default=True) or bool(dry_run_param)`, so live orders require DRY_RUN to be explicitly `"false"` **and** no forced dry-run flag to be passed. This belt-and-suspenders means a missing value never silently enables live trading.

---

## Setup

1. **Python 3.11+** (the code uses modern typing / async).
2. Create a virtual environment and activate it.
3. Install dependencies: `pip install -r requirements.txt` (includes `anthropic`, `apscheduler>=3.11`, `httpx`, `mcp`, `pydantic`, `pytz`, `python-dotenv`, `uvicorn`, etc.).
4. Create `.env` with `ANTHROPIC_API_KEY`, `FINNHUB_API_KEY`, and `DRY_RUN=true` (start in dry-run).
5. **(First run only)** Authenticate with Robinhood — the app opens the browser for the one-time OAuth approval; a `.rh_oauth_cache.json` is then reused/silently refreshed on later runs. Requires an active Robinhood **Agentic** account configured to allow (`agentic_allowed=true`).
6. Run: `python main.py`

> The legacy setup step `claude mcp add robinhood-trading --transport http https://agent.robinhood.com/mcp/trading` is no longer required for the active code path (the client connects directly). It is retained for reference and for the legacy `.claude` permission allow-list.

**Execution modes:**
- **Dry-run (default):** `DRY_RUN=true` (or unset) → every trade is validated and logged as `SIMULATED` / `WOULD_EXECUTE` but no real orders are placed.
- **Live:** only after you review the logic, set `DRY_RUN=false` in `.env` AND keep `trading.dry_run` consistent; live orders flow through `review_equity_order` / `place_equity_order` only after passing all code-level risk checks and the safe-window gate.

---

## Important Warning

**This project is for research, experimentation, and education. Trading involves real financial risk, and this is genuinely autonomous money-moving software.** It places/simulates orders against a real brokerage account based on LLM judgment, and while risk rules are enforced in code, LLM output is non-deterministic and markets are unpredictable.

- Do **not** use live capital without independent code review, compliance checks, and an explicit risk policy.
- Start strictly in **dry-run** (`DRY_RUN=true`) and study `logs/trades.json`, `logs/positions.json`, and the journals before ever enabling live trading.
- The hard-coded pricing/limits, thresholds, and OAuth/account-selection logic should be re-validated against your account and Robinhood's current API/terms before live use.
- Nothing here is investment advice.
