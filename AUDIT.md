# CODEBASE AUDIT - Robinhood MCP Trading Bot

## 1. Fixed Ticker List Location
- **File**: `config/settings.json`
- **Key**: `watchlist.default_tickers` (array of 10 tickers: JPM, NVDA, AVGO, CAT, MU, TSM, UNH, XOM, AMD, COST)
- **Loaded in**: `TradingAgent.__init__()` via `self.config["watchlist"]["default_tickers"]`
- **Used in**: `TradingAgent.run_cycle()` → passed to `DataIngestLayer.ingest()`

## 2. Trading Universe Decision Logic
- `TradingAgent.run_cycle()` gets watchlist from config
- `DataIngestLayer.ingest(tickers, existing_tickers)` takes union of watchlist + open positions (deduped, max 20)
- `DecisionEngine.make_decision()` evaluates each ticker
- `ExecutionEngine.execute()` enforces `max_positions` (from config: 12, not 8 as task states)

## 3. Holdings / Position Tracking
- **PositionManager** (`core/position_manager.py`): Persists `PositionRecord` to `logs/positions.json`
- **Reconciliation**: `reconcile_with_mcp(portfolio)` syncs local records with live Robinhood positions
- **Portfolio model** (`models/portfolio.py`): `positions: list[Position]` from MCP `get_portfolio()`
- **Open positions retrieved via**: `PositionManager.get_open_records()` → returns list of `PositionRecord`

## 4. Max Positions Logic
- **Config**: `trading.max_positions = 12` (NOT 8 as stated in task)
- **Enforcement**: `ExecutionEngine._get_open_position_count()` counts records in `positions.json`
- **Position sizing**: `_size_from_edge()` maps edge score to 3%/4%/6%/8% of portfolio

## 5. Market Hours & 3 Daily Trades
- **Scheduler**: APScheduler `AsyncIOScheduler` in `TradingAgent.start()`
- **Jobs**:
  - OPEN: Mon-Fri 9:35 ET (`CycleType.OPEN`)
  - MID: Mon-Fri 12:30 ET (`CycleType.MID`)
  - CLOSE: Mon-Fri 15:30 ET (`CycleType.CLOSE`)
  - Price monitor: every minute 9:30-16:00 ET Mon-Fri
- **Timezone**: `America/New_York` (configurable)
- **Execution window guard**: Blocks 9:30-9:34 and 15:50-15:59 ET

## 6. Finnhub Integration (EXISTING)
- **File**: `core/news_fetch.py` → `NewsFetcher` class
- **Endpoints**: `/company-news` (per ticker) + `/news` (general market)
- **API Key**: `FINNHUB_API_KEY` from env var (loaded via `os.getenv()`)
- **Config**: `news.lookback_days=3`, `max_per_ticker=5`, `fresh_window_hours=24`
- **Deduplication**: Finnhub `id` field persisted to `data/seen_headlines.json`
- **Rate limiting**: Basic semaphore (10 concurrent), no exponential backoff
- **Haiku fallback/summary**: Optional Anthropic Haiku calls for tickers with no news

## 7. Logging, Error Handling, Retry Patterns
- **Logger**: `utils/logger.py` → UTCFormatter, console + file (`logs/agent.log`) + JSON trades (`logs/trades.json`)
- **Error handling**: Try/except with `logger.warning()` throughout, never crashes on single ticker failure
- **Retries**: 
  - Anthropic: 3 attempts, 5s delay, retry on rate limit (529) / "too many requests"
  - Robinhood MCP: `_call_tool_with_retry()` with exponential backoff (max 3 attempts)
  - Finnhub: No retry/backoff currently (only semaphore)

## 8. Existing Periodic Jobs
- Only the 3 daily trading cycles + 1-minute price monitor
- **NO weekly jobs exist**

## 9. Dependency Management
- **File**: `requirements.txt` with pinned versions
- **Key deps**: anthropic==0.113.0, httpx==0.28.1, pydantic==2.10.1, apscheduler==3.11.3, mcp==1.28.1, python-dotenv==1.2.2
- **No ML/transformers deps currently**

## 10. Environment Variables & Secrets
- **Loading**: `load_dotenv()` in `main.py` line 13
- **Secrets**:
  - `FINNHUB_API_KEY` → used in `NewsFetcher.__init__()`
  - Robinhood OAuth → cached in `.rh_oauth_cache.json` (auto-managed)
  - Anthropic API key → via `ANTHROPIC_API_KEY` env var (used by `anthropic.AsyncAnthropic()`)
  - `DRY_RUN` → in `config["env"]["DRY_RUN"]`
- **No .env.example file exists**

## 11. Key Integration Points for New Ticker Selection Layer
- **Replace watchlist source**: `TradingAgent.run_cycle()` reads from config → needs to read dynamic list
- **Protected tickers**: `PositionManager.get_open_records()` returns held tickers
- **Persistence**: Match existing style (JSON files in `logs/` or `data/`)
- **Scheduler**: Add weekly job to existing `AsyncIOScheduler`
- **Config**: Add new section to `settings.json` for sentiment selection params
- **Feature flag**: Add `ENABLE_SENTIMENT_TICKER_SELECTION` to config

## 12. Discrepancies with Task Description
- Task says "max 8 open positions" → **Actual: 12** (config: `trading.max_positions`)
- Task says "total universe size of 10 tickers" → **Actual: 10** (watchlist has 10)
- These are compatible (10 universe, 12 max positions = can hold all 10)
- Will implement with configurable universe size (default 10) and max positions (from config)