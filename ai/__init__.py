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

from ai.strategy import Decision, Strategy, RiskLevel, assess_risk

from ai.position_manager import PositionManager, ActionResult, close_all_positions

# ML Client
from ai.ml_client import MLClient, MLPrediction, EnsembleMLPrediction

# News Client
from ai.news_client import NewsClient, NewsSentiment, NewsArticle

# LLM Client
from ai.llm_client import LLMClient, LLMAnalysisResult, create_llm_client

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
    "Decision",
    "Strategy",
    "RiskLevel",
    "assess_risk",
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
]
