# robinhood-agent

Anthropic-only agentic trading system rebuilt around the new core pipeline:
- core/data_ingest.py
- core/metrics.py
- core/news_fetch.py
- core/decision_engine.py
- core/position_manager.py
- core/execution.py

## What changed

- Removed the legacy OpenAI-based routing/extraction stack.
- Switched all model calls to Anthropic only.
- Enforced DRY_RUN in execution logic, not only in configuration.
- Added position entry/exit logging in logs/positions.json.
- Added Anthropic pricing and retry handling in utils/anthropic_client.py and utils/cost_tracker.py.

## Setup

1. Create a Python 3.11+ virtual environment.
2. Install dependencies:
   pip install -r requirements.txt
3. Copy the sample environment file and fill in your keys:
   copy .env.example .env

## MCP setup

1. Install the Claude Code CLI.
2. Register the Robinhood MCP endpoint:
   claude mcp add robinhood-trading --transport http https://agent.robinhood.com/mcp/trading
3. Verify the CLI can call the MCP before running the agent.

## Execution mode

- Default: DRY_RUN=true
- To enable live trading only after validation, set DRY_RUN=false in .env.

Run:
   python main.py

## Important warning

This project is for research, experimentation, and education. Trading involves financial risk. Do not use live capital without independent review, compliance checks, and an explicit risk policy.
