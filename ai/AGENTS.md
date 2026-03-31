# AI Trading Engine

**Parent:** [/](../AGENTS.md)

**Core Components:** Ensemble trading system combining technical, ML, LLM, and news signals.

## Structure

```
ai/
├── trading.py           # Main loop orchestrator
├── position_manager.py  # Position sizing & risk
├── market_analyzer.py   # OHLCV + indicators
├── ensemble_voter.py    # Signal aggregation
├── ml_client.py         # localhost:8000 predictions
├── llm_client.py        # LLM reasoning (OpenAI/Anthropic)
├── news_client.py       # localhost:3001 sentiment
├── model_tracker.py     # ML model performance
├── ml_experience.py     # Experience replay storage
├── monitoring.py        # Metrics & alerts
└── strategies/          # Trading strategies
    ├── ensemble.py      # Ensemble-based signals
    ├── breakout.py      # Breakout detection
    ├── momentum.py      # Momentum signals
    ├── technical.py     # Technical indicator signals
    └── trend.py         # Trend-following
```

## WHERE TO LOOK

| Task | File | Notes |
|------|------|-------|
| Trading loop | `trading.py` | Main `run()` function |
| Position sizing | `position_manager.py` | Risk calculations |
| Signal voting | `ensemble_voter.py` | Weighted ensemble |
| Indicator calc | `market_analyzer.py` | RSI, MACD, ADX, etc. |
| ML predictions | `ml_client.py` | localhost:8000 |

## CONVENTIONS

- **Signal classes**: Inherit from `TradingSignal` dataclass
- **Weights**: Defined in `SOURCE_WEIGHTS` (ai/config.py)
- **Logging**: Use `add_log(coin=..., data=...)`
- **Exceptions**: Wrap API calls with try/except + logging

## ANTI-PATTERNS

- ❌ Don't modify `ml_experience.py` directly (use `add_experience()`)
- ❌ Don't hardcode API endpoints (use SETTINGS from config)
- ❌ Don't skip logging on trade execution

## SIGNAL FLOW

```
market_analyzer → technical/breakout signals
                    ↓
llm_client → LLM analysis (prompt-based)
                    ↓
news_client → sentiment score (-1 to 1)
                    ↓
ml_client → ML predictions (12 models)
                    ↓
ensemble_voter → weighted vote → trading.py
                    ↓
position_manager → sizing → execution
```

## KEY CONFIG

```python
# ai/config.py
SOURCE_WEIGHTS = {
    "technical": 0.30,
    "llm": 0.25,
    "ml_predictions": 0.35,
    "news": 0.10
}

TIMEFRAMES = ["1m", "5m", "15m", "1h", "4h"]
COINS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT"]
```
