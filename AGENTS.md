# AI Trading Bot - SEMI_AUTO AGENTS.md

**Generated:** 2026-01-24  
**Python:** 3.14+ | **Package Manager:** uv

## Project Overview

Weex exchange trading bot with AI-driven ensemble trading decisions. Combines technical analysis, LLM insights, ML predictions, and news sentiment for short-term leveraged trading.

## Structure

```
semi-auto/
├── ai/                   # Trading logic and AI analysis
│   ├── trading.py        # Main trading loop orchestrator
│   ├── strategy.py       # Trading signal generation
│   ├── market_analyzer.py # Market data & indicator calculation
│   ├── ml_client.py      # ML predictions client (localhost:8000)
│   ├── news_client.py    # News sentiment client (localhost:3001)
│   ├── ensemble_voter.py # Multi-source signal voting
│   └── model_tracker.py  # ML model performance tracking
├── api/                  # FastAPI endpoints
├── utils/                # Indicators, logging, config
├── tests/                # Unit tests (pytest)
├── main.py               # Entry point (uvicorn on port 8888)
├── config.yaml           # Configuration file
└── trading.log           # JSON structured logs
```

## Commands

### Development & Testing
```bash
# Install dependencies
uv sync

# Run development server
uv run python main.py

# Run all tests
uv run pytest tests/

# Run with coverage
uv run pytest tests/ --cov=ai --cov-report=term-missing

# Run single test file (use full path or relative)
uv run pytest tests/test_indicators.py -v
uv run pytest tests/test_ml_client.py -v
uv run pytest tests/test_ensemble_voter.py -v
uv run pytest tests/test_config.py -v

# Run single test class
uv run pytest tests/test_indicators.py::TestCalculateVWMA -v

# Run single test method
uv run pytest tests/test_indicators.py::TestCalculateVWMA::test_normal_data -v

# Linting (ruff)
uv run ruff check .
uv run ruff check --fix .
```

## Code Style Guidelines

### Imports
- Standard library → third-party → local
- Explicit imports only (no `from x import *`)
- Group with blank lines between groups:
  ```python
  import asyncio
  from dataclasses import dataclass
  from typing import Any
  
  import aiohttp
  
  from ai.config import SETTINGS
  from utils.logger import add_log
  ```

### Naming
- **Functions/variables:** `snake_case`
- **Classes:** `PascalCase`
- **Constants:** `UPPER_SNAKE_CASE`
- **Private methods:** `_single_underscore_prefix`
- **Type variables:** `PascalCase` (e.g., `T`, `Result`)

### Type Hints
- Explicit types for parameters and returns
- Avoid `Any` - prefer `Union`, `Optional`, specific types
- Built-in generics (`list[str]`, `dict[str, int]`) for Python 3.9+
- Never suppress types with `as any`, `@ts-ignore`, `@ts-expect-error`

### Error Handling
- Specific exceptions (`ValueError`, `TypeError`, `KeyError`)
- Wrap API calls with try-except and retry logic
- Log errors with structured context:
  ```python
  try:
      result = await client.get_kline(coin, granularity=granularity)
  except Exception as e:
      add_log(f"API error for {coin}: {e}", level="ERROR", coin=coin, error_type="api_failure")
      raise
  ```
- Empty catch blocks are forbidden: `except: {}` ❌

### Logging
- Use `utils.logger.add_log()` for structured logging:
  ```python
  add_log(message="Trade executed", level="INFO", data={"coin": "BTCUSDT", "price": 50000.0}, coin="BTCUSDT")
  ```
- Console: `[2026-01-17 15:00:00] INFO: message`
- File (trading.log): JSON with `timestamp`, `level`, `message`, `data`, `coin`, `error_type`

### Functions & Classes
- Keep small - single responsibility
- Use `@dataclass` for simple data containers
- Prefer composition over deep inheritance
- Add docstrings for complex functions

### Ensemble Trading System
When adding new signals to the ensemble:
1. Add signal class in `ai/ensemble_voter.py`
2. Add weight in `SOURCE_WEIGHTS` (ai/config.py)
3. Integrate in `MarketAnalyzer.get_ml_market_signals()`
4. Add tests in `tests/test_ensemble_voter.py`
5. Update `ai/__init__.py` exports

Current weights: Technical(30%), LLM(25%), ML Predictions(35%), News(10%)

### Anti-Patterns
- ❌ Don't use `from utils import *`
- ❌ Don't catch bare `except:`
- ❌ Don't log sensitive data (API keys, passwords)
- ❌ Don't use global state
- ❌ Don't ignore linting errors
- ❌ Don't commit commented-out code
- ❌ Don't use type suppression (`as any`, `@ts-ignore`)

## Key Files

| File | Purpose |
|------|---------|
| `ai/trading.py` | Main trading loop orchestrator |
| `ai/ensemble_voter.py` | Multi-source signal voting system |
| `ai/ml_client.py` | ML predictions from localhost:8000 |
| `ai/news_client.py` | News sentiment from localhost:3001 |
| `utils/indicators.py` | Technical analysis (RSI, MACD, ADX, etc.) |
| `utils/logger.py` | Structured JSON logging |
| `api/app.py` | FastAPI application |
| `tests/` | Unit tests (29+ tests) |

## External Services

| Service | Host | Purpose |
|---------|------|---------|
| ML Server | localhost:8000 | Price predictions (12 models) |
| News API | localhost:3001 | Crypto news sentiment (nirholas/free-crypto-news) |
| Weex API | api.weex.com | Exchange trading |

## Common Tasks

### Add New ML Model
1. Add model name to `ML_AVAILABLE_MODELS` in `ai/config.py`
2. ML server automatically includes it in predictions
3. Performance tracked in `model_performance.json`

### Add New Indicator
1. Implement in `utils/indicators.py`
2. Add to `calculate_indicators()` with config support
3. Add unit tests in `tests/test_indicators.py`
4. Update `config.yaml` if configurable

### Add New Timeframe
1. Add to `CANDLES_CONFIG` in `ai/config.py`
2. Update `TIMEFRAMES` list
3. Update `tf_order` in display/logging functions
