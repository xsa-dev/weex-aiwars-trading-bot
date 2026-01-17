"""Trading bot configuration constants."""

import os

# Number of candles to fetch for each timeframe
# These ensure consistent time coverage (~48 hours) across all timeframes
CANDLES_CONFIG = {
    "1m": 1000,  # ~16 hours (was 2880 - caused 40020 error)
    "15m": 192,  # 48 hours
    "1h": 192,  # 48 hours
    "4h": 100,  # 400 hours (~16.6 days)
    # 1D and 1W are fetched via get_history_kline (no limit)
}

# Timeframe order for display and analysis
TIMEFRAMES = ["1m", "15m", "1h", "4h", "1D", "1W"]

# Order execution settings
DEFAULT_ORDER_SIZE = 0.001
MIN_PROFIT_THRESHOLD = 1.0  # % - take profit if profit exceeds this
STOP_LOSS_PERCENT = 0.98  # 2% stop loss
TAKE_PROFIT_PERCENT = 1.03  # 3% take profit
RISK_REWARD_RATIO = 1.5

# Loop settings
LOOP_DELAY = 60  # seconds between iterations
MARKET_DATA_DELAY = 0.1  # seconds between coin fetches

# Strategy confidence thresholds
SIGNAL_CONFIDENCE_DEFAULT = 0.5
SIGNAL_CONFIDENCE_STRONG = 0.75
SIGNAL_CONFIDENCE_WEAK = 0.60

# Minimum signals required for position
MIN_BULLISH_SIGNALS = 3  # Need 3+ bullish timeframes for LONG

# Paper trading mode - trades saved to disk, not sent to Weex
# Can be enabled via environment variable: export PAPER_TRADING=true
PAPER_TRADING = os.getenv("PAPER_TRADING", "").lower() in ("true", "1", "yes")
PAPER_TRADES_FILE = "paper_trades.json"
