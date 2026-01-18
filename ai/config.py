"""Trading bot configuration constants."""

import os

# =============================================================================
# News API Configuration
# =============================================================================

NEWS_SERVER_HOST = os.getenv(
    "NEWS_SERVER_HOST", "https://free-crypto-news-api.cloudpub.ru"
)
NEWS_API_ENDPOINT = "/api/news"
NEWS_REQUEST_TIMEOUT = 10  # seconds
NEWS_CACHE_DURATION = 300  # 5 minutes in seconds

# Coin keywords for news sentiment filtering
COIN_KEYWORDS = {
    "cmt_btcusdt": ["bitcoin", "btc"],
    "cmt_ethusdt": ["ethereum", "eth"],
    "cmt_xrpusdt": ["xrp", "ripple"],
    "cmt_solusdt": ["solana", "sol"],
    "cmt_adabusdt": ["cardano", "ada"],
    "cmt_dotusdt": ["polkadot", "dot"],
    "cmt_dogeusdt": ["dogecoin", "doge"],
    "cmt_maticusdt": ["polygon", "matic"],
    "cmt_ltcusdt": ["litecoin", "ltc"],
    "cmt_linkusdt": ["chainlink", "link"],
    "cmt_uniusdt": ["uniswap", "uni"],
    "cmt_avaxusdt": ["avalanche", "avax"],
    "cmt_atomusdt": ["cosmos", "atom"],
    "cmt_near-usdt": ["near protocol", "near"],
    "cmt_injusdt": ["injective", "inj"],
}

# =============================================================================
# Ensemble Voting Configuration
# =============================================================================

# Source weights for ensemble voting (must sum to 1.0)
SOURCE_WEIGHTS = {
    "technical": 0.35,
    "llm": 0.25,
    "ml_predictions": 0.35,
}

# Confidence thresholds for trading decisions
CONFIDENCE_THRESHOLD_BUY = 0.70  # 70% confidence for buy signals
CONFIDENCE_THRESHOLD_SELL = 0.70  # 70% confidence for sell signals

# Emergency mode - stop trading if drawdown exceeds threshold
EMERGENCY_MODE_ENABLED = True
EMERGENCY_DRAWDOWN_THRESHOLD = 0.10  # 10% portfolio drawdown triggers HOLD


# Position sizing based on confidence
# Returns multiplier for position size
def get_position_multiplier(confidence: float) -> float:
    """Get position size multiplier based on signal confidence."""
    if confidence >= 0.80:
        return 1.5  # Very high confidence - increase size
    elif confidence >= 0.70:
        return 1.2  # High confidence - normal size
    elif confidence >= 0.65:
        return 0.8  # Moderate confidence - reduce size
    else:
        return 0.5  # Low confidence - minimal size


# =============================================================================
# ML Prediction Configuration
# =============================================================================

ML_SERVER_HOST = os.getenv("ML_SERVER_HOST", "http://localhost:8000")
ML_REQUEST_TIMEOUT = 30  # seconds
ML_PREDICTION_HOURS = 6  # Predict next 6 hours
ML_AVAILABLE_MODELS = [
    "XGBRegressor",
    "LightGBM",
    "CatBoost",
    "RandomForestRegressor",
    "ExtraTreesRegressor",
    "LinearRegression",
    "Ridge",
    "ElasticNet",
    "KNeighborsRegressor",
    "SVR",
    "LSTM",
    "GRU",
]
ML_MODEL_TRACKING_FILE = "model_performance.json"

# =============================================================================
# LLM Configuration (DeepSeek/VSE API)
# =============================================================================

VSE_LLM_API_KEY = os.getenv("VSE_LLM_API_KEY", "")
VSE_LLM_API_BASE = os.getenv("VSE_LLM_API_BASE", "https://api.vsellm.ru/v1")
LLM_MODEL = "deepseek/deepseek-v3.2-speciale"
LLM_REQUEST_TIMEOUT = 120  # seconds

# =============================================================================
# Candles Configuration
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
MAX_RISK_PER_TRADE_USDT = 10.0  # Maximum risk in USDT per trade
MAX_TOTAL_RISK_USDT = 80.0  # Maximum total risk in USDT
MAX_OPEN_POSITIONS = 8  # Maximum number of open positions
MIN_PROFIT_THRESHOLD = 1.0  # % - take profit if profit exceeds this
STOP_LOSS_PERCENT = 0.98  # 2% stop loss
TAKE_PROFIT_PERCENT = 1.03  # 3% take profit
RISK_REWARD_RATIO = 1.5

# Loop settings
LOOP_DELAY = 300  # seconds between iterations
MARKET_DATA_DELAY = 0.1  # seconds between coin fetches

# Strategy confidence thresholds
SIGNAL_CONFIDENCE_DEFAULT = 0.5
SIGNAL_CONFIDENCE_STRONG = 0.75
SIGNAL_CONFIDENCE_WEAK = 0.60

# Minimum signals required for position
MIN_BULLISH_SIGNALS = 4  # Need 3+ bullish timeframes for LONG

# Paper trading mode - trades saved to disk, not sent to Weex
# Can be enabled via environment variable: export PAPER_TRADING=true
PAPER_TRADING = os.getenv("PAPER_TRADING", "").lower() in ("true", "1", "yes")
PAPER_TRADES_FILE = "paper_trades.json"
