# AI Trading Bot - SEMI_AUTO AGENTS.md

**Generated:** 2026-01-17  
**Python:** 3.14+ | **Package Manager:** uv

## Project Overview

Weex exchange trading bot with AI indicator analysis. FastAPI backend with custom technical analysis indicators supporting 6 timeframes (1m, 15m, 1h, 4h, 1D, 1W).

## Structure

```
semi-auto/
├── ai/              # Trading logic and AI analysis
├── api/             # FastAPI endpoints
├── utils/           # Indicators, logging, config
├── tests/           # Unit tests (pytest)
├── main.py          # Entry point (uvicorn on port 8888)
├── config.yaml      # Configuration file
└── trading.log      # JSON structured logs
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
uv run pytest tests/ --cov=utils --cov-report=term-missing

# Run single test file
uv run pytest tests/test_indicators.py -v

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
- Group with blank lines between groups

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
- Use `@retry_on_api_error` decorator for common patterns

### Logging
- Use `utils.logger.add_log()` for structured logging:
  ```python
  add_log(message="Trade executed", level="INFO", data={"coin": "BTCUSDT", "price": 50000.0}, coin="BTCUSDT")
  ```
- Console: `[2026-01-17 15:00:00] INFO: message`
- File (trading.log): JSON with `timestamp`, `level`, `message`, `data`, `coin`, `error_type`

### Functions
- Keep small - single responsibility
- Use type hints for all parameters and returns
- Add docstrings for complex functions

### Classes
- Use `@dataclass` for simple data containers
- Prefer composition over deep inheritance

### Testing
- Test classes for grouped functionality
- Test methods with descriptive names: `test_normal_data`, `test_zero_volume`
- Mock external dependencies (API calls, file I/O)

### Configuration
- Use `config.yaml` for environment-specific settings
- Load via `utils.config_loader.load_config()`
- Never hardcode magic numbers

### API Integration (Weex)
- Use `weex_client` for API calls
- **Lowercase granularity**: `"1d"`, `"1w"` NOT `"1D"`, `"1W"`
- Handle rate limits with exponential backoff
- Use safe wrappers: `safe_get_kline()`, `safe_get_history_kline()`

### Anti-Patterns
- ❌ Don't use `from utils import *`
- ❌ Don't catch bare `except:`
- ❌ Don't log sensitive data (API keys, passwords)
- ❌ Don't use global state
- ❌ Don't ignore linting errors
- ❌ Don't commit commented-out code

## Key Files

| File | Purpose |
|------|---------|
| `ai/trading.py` | Trading logic, rate limiting, retry wrappers |
| `utils/indicators.py` | Technical analysis (RSI, MACD, ADX, etc.) |
| `utils/logger.py` | Structured JSON logging |
| `api/app.py` | FastAPI application |
| `tests/test_indicators.py` | 29 unit tests for indicators |

## Common Tasks

### Add New Indicator
1. Implement in `utils/indicators.py`
2. Add to `calculate_indicators()` with config support
3. Add unit tests in `tests/test_indicators.py`
4. Update `config.yaml` if configurable

### Add New Timeframe
1. Add to `CANDLES_CONFIG` in `ai/trading.py`
2. Update `tf_order` in `format_all_timeframes_log()` and `indicators_to_structured()`

## Notes

- Russian comments mixed with English in legacy code (acceptable)
- pandas/numpy used extensively for data processing
- asyncio for concurrent API operations
- Structured logging critical for debugging
