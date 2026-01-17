"""AI Trading Bot modules."""

# Core trading loop
from ai.trading import trade_loop, start_trading, stop_trading

# Market analysis
from ai.market_analyzer import (
    MarketAnalyzer,
    MarketData,
    Indicators,
    MLSignals,
    RateLimiter,
    format_indicators_log,
    indicators_to_structured,
)

# Strategy
from ai.strategy import Decision, Strategy, RiskLevel, assess_risk

# Position management
from ai.position_manager import PositionManager, ActionResult, close_all_positions

__all__ = [
    # Trading loop
    "trade_loop",
    "start_trading",
    "stop_trading",
    # Market analyzer
    "MarketAnalyzer",
    "MarketData",
    "Indicators",
    "MLSignals",
    "RateLimiter",
    "format_indicators_log",
    "indicators_to_structured",
    # Strategy
    "Decision",
    "Strategy",
    "RiskLevel",
    "assess_risk",
    # Position manager
    "PositionManager",
    "ActionResult",
    "close_all_positions",
]
