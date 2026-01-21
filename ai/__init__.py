"""AI Trading Bot modules."""

from ai.trading import trade_loop, start_trading, stop_trading

from ai.market_analyzer import (
    MarketAnalyzer,
    MarketData,
    Indicators,
    MLSignals,
    RateLimiter,
    format_indicators_log,
    indicators_to_structured,
)

from ai.position_manager import PositionManager, ActionResult, close_all_positions

# ML Client
from ai.ml_client import MLClient, MLPrediction, EnsembleMLPrediction

# News Client
from ai.news_client import NewsClient, NewsSentiment, NewsArticle

# Ensemble Voting
from ai.ensemble_voter import (
    EnsembleVoter,
    EnsembleSignal,
    TechnicalSignal,
    LLMSignal,
    MLPredictionSignal,
    NewsSentimentSignal,
)

# Model Tracking
from ai.model_tracker import ModelTracker, ModelPerformance

# Monitoring
from ai.monitoring import metrics, start_metrics_server, MetricsCollector

# Strategies
from ai.strategies import (
    BaseStrategy,
    StrategySignal,
    TechnicalStrategy,
    TrendStrategy,
    MomentumStrategy,
    BreakoutStrategy,
    EnsembleCombiner,
)

# ML Experience
from ai.ml_experience import MLExperienceTracker

__all__ = [
    "trade_loop",
    "start_trading",
    "stop_trading",
    "MarketAnalyzer",
    "MarketData",
    "Indicators",
    "MLSignals",
    "RateLimiter",
    "format_indicators_log",
    "indicators_to_structured",
    "PositionManager",
    "ActionResult",
    "close_all_positions",
    "MLClient",
    "MLPrediction",
    "EnsembleMLPrediction",
    "NewsClient",
    "NewsSentiment",
    "NewsArticle",
    "EnsembleVoter",
    "EnsembleSignal",
    "TechnicalSignal",
    "LLMSignal",
    "MLPredictionSignal",
    "NewsSentimentSignal",
    "ModelTracker",
    "ModelPerformance",
    # Monitoring
    "metrics",
    "start_metrics_server",
    "MetricsCollector",
    # Strategies
    "BaseStrategy",
    "StrategySignal",
    "TechnicalStrategy",
    "TrendStrategy",
    "MomentumStrategy",
    "BreakoutStrategy",
    "EnsembleCombiner",
    # ML Experience
    "MLExperienceTracker",
]
