"""Ensemble voting system combining signals from multiple sources.

Combines:
- Technical analysis (RSI, ADX, MACD, SuperTrend)
- LLM analysis (DeepSeek v3.2)
- ML predictions (XGBoost, LSTM, GRU, etc.)
- News sentiment (nirholas/free-crypto-news)
"""

from dataclasses import dataclass
from typing import Any

from ai.config import (
    SOURCE_WEIGHTS,
    CONFIDENCE_THRESHOLD_BUY,
    CONFIDENCE_THRESHOLD_SELL,
    get_position_multiplier,
    DEFAULT_ORDER_SIZE,
    STOP_LOSS_PERCENT,
    TAKE_PROFIT_PERCENT,
    EMERGENCY_DRAWDOWN_THRESHOLD,
    EMERGENCY_MODE_ENABLED,
)
from utils.logger import add_log


@dataclass
class EnsembleSignal:
    """Final ensemble trading signal."""

    coin: str
    signal: str
    confidence: float
    reasoning: str
    source_breakdown: dict[str, dict[str, Any]]
    risk_level: str
    position_size: float
    stop_loss: float | None
    take_profit: float | None
    is_emergency: bool = False


@dataclass
class TechnicalSignal:
    """Technical analysis signal."""

    signal: str
    confidence: float
    reasoning: str
    indicators_used: dict[str, Any] = None


@dataclass
class LLMSignal:
    """LLM analysis signal."""

    signal: str
    confidence: float
    reasoning: str
    model: str = "deepseek-v3.2"


@dataclass
class MLPredictionSignal:
    """ML prediction signal."""

    direction: str
    confidence: float
    avg_change: float
    model_count: int
    models_used: list[str]


@dataclass
class NewsSentimentSignal:
    """News sentiment signal."""

    sentiment: str
    confidence: float
    total_articles: int
    reasoning: str = ""


class EnsembleVoter:
    """Combine signals from technical, LLM, ML, and news sources."""

    def __init__(
        self,
        weights: dict[str, float] | None = None,
        buy_threshold: float = CONFIDENCE_THRESHOLD_BUY,
        sell_threshold: float = CONFIDENCE_THRESHOLD_SELL,
        hold_threshold: float = 0.50,
        emergency_threshold: float = EMERGENCY_DRAWDOWN_THRESHOLD,
    ):
        self.weights = weights or SOURCE_WEIGHTS.copy()
        self.buy_threshold = buy_threshold
        self.sell_threshold = sell_threshold
        self.hold_threshold = hold_threshold
        self.emergency_threshold = emergency_threshold

    def vote(
        self,
        coin: str,
        price: float,
        technical: TechnicalSignal,
        llm: LLMSignal | None = None,
        ml: MLPredictionSignal | None = None,
        news: NewsSentimentSignal | None = None,
        current_portfolio_drawdown: float = 0.0,
        current_position: dict[str, Any] | None = None,
    ) -> EnsembleSignal:
        """Combine all signals into final decision."""
        is_emergency = (
            EMERGENCY_MODE_ENABLED
            and current_portfolio_drawdown >= self.emergency_threshold
        )

        if is_emergency:
            add_log(
                f"EMERGENCY MODE for {coin}: drawdown {current_portfolio_drawdown:.1%}",
                level="WARNING",
                coin=coin,
            )
            return EnsembleSignal(
                coin=coin,
                signal="hold",
                confidence=0.0,
                reasoning=f"EMERGENCY: {current_portfolio_drawdown:.1%} ≥ {self.emergency_threshold:.1%}",
                source_breakdown={},
                risk_level="EXTREME",
                position_size=0.0,
                stop_loss=None,
                take_profit=None,
                is_emergency=True,
            )

        source_breakdown: dict[str, dict[str, Any]] = {}

        # Technical (always available)
        tech_direction = self._parse_direction(technical.signal)
        tech_confidence = technical.confidence
        tech_weight = self.weights.get("technical", 0.30)

        source_breakdown["technical"] = {
            "signal": technical.signal,
            "direction": tech_direction,
            "confidence": tech_confidence,
            "reasoning": technical.reasoning[:150],
            "weight": tech_weight,
            "indicators": technical.indicators_used or {},
        }

        # LLM (optional)
        if llm:
            llm_direction = self._parse_direction(llm.signal)
            llm_confidence = llm.confidence
            llm_weight = self.weights.get("llm", 0.25)

            source_breakdown["llm"] = {
                "signal": llm.signal,
                "direction": llm_direction,
                "confidence": llm_confidence,
                "reasoning": llm.reasoning[:150],
                "weight": llm_weight,
                "model": llm.model,
            }
        else:
            self._redistribute_weight("llm")

        # ML predictions (optional)
        if ml:
            ml_direction = self._parse_direction(ml.direction)
            ml_confidence = ml.confidence
            ml_weight = self.weights.get("ml_predictions", 0.35)

            source_breakdown["ml_predictions"] = {
                "signal": ml.direction,
                "direction": ml_direction,
                "confidence": ml_confidence,
                "reasoning": f"{ml.model_count} models, avg: {ml.avg_change:+.2f}%",
                "weight": ml_weight,
                "model_count": ml.model_count,
                "models_used": ml.models_used[:5],
            }
        else:
            self._redistribute_weight("ml_predictions")

        # News sentiment (optional)
        if news:
            news_direction = self._parse_direction(news.sentiment)
            news_confidence = news.confidence
            news_weight = self.weights.get("news", 0.10)

            source_breakdown["news"] = {
                "signal": news.sentiment,
                "direction": news_direction,
                "confidence": news_confidence,
                "reasoning": f"{news.total_articles} recent articles",
                "weight": news_weight,
            }
        else:
            self._redistribute_weight("news")

        weighted_confidence = self._calculate_weighted_confidence(source_breakdown)
        final_signal, vote_details = self._vote_direction(source_breakdown)

        risk_level = self._calculate_risk_level(
            weighted_confidence,
            vote_details["buy_votes"],
            vote_details["sell_votes"],
            len(source_breakdown),
        )

        position_multiplier = get_position_multiplier(weighted_confidence)
        position_size = DEFAULT_ORDER_SIZE * position_multiplier
        stop_loss, take_profit = self._calculate_sl_tp(price, final_signal)

        reasoning = self._generate_reasoning(
            coin=coin,
            final_signal=final_signal,
            confidence=weighted_confidence,
            risk_level=risk_level,
            source_breakdown=source_breakdown,
        )

        return EnsembleSignal(
            coin=coin,
            signal=final_signal,
            confidence=weighted_confidence,
            reasoning=reasoning,
            source_breakdown=source_breakdown,
            risk_level=risk_level,
            position_size=position_size,
            stop_loss=stop_loss,
            take_profit=take_profit,
        )

    def _parse_direction(self, direction: str) -> str:
        direction_lower = str(direction).lower()
        if direction_lower in ["long", "buy", "bullish", "up", "overbought"]:
            return "buy"
        elif direction_lower in ["short", "sell", "bearish", "down", "oversold"]:
            return "sell"
        else:
            return "hold"

    def _redistribute_weight(self, missing_source: str):
        if missing_source not in self.weights:
            return
        missing_weight = self.weights.pop(missing_source)
        total_other_weight = sum(self.weights.values())
        if total_other_weight == 0:
            return
        for source in self.weights:
            self.weights[source] += missing_weight * (
                self.weights[source] / total_other_weight
            )

    def _calculate_weighted_confidence(self, source_breakdown: dict) -> float:
        if not source_breakdown:
            return 0.0
        total_weighted_conf = 0.0
        total_weight = 0.0
        for source, data in source_breakdown.items():
            confidence = data.get("confidence", 0.5)
            weight = data.get("weight", 0.25)
            total_weighted_conf += confidence * weight
            total_weight += weight
        return total_weighted_conf / total_weight if total_weight > 0 else 0.0

    def _vote_direction(self, source_breakdown: dict) -> tuple[str, dict]:
        directions = {
            source: data.get("direction", "hold")
            for source, data in source_breakdown.items()
        }
        buy_votes = sum(1 for d in directions.values() if d == "buy")
        sell_votes = sum(1 for d in directions.values() if d == "sell")
        vote_details = {
            "buy_votes": buy_votes,
            "sell_votes": sell_votes,
            "total": len(directions),
        }

        if buy_votes > sell_votes:
            return "buy", vote_details
        elif sell_votes > buy_votes:
            return "sell", vote_details
        else:
            buy_conf = sum(
                source_breakdown[s].get("confidence", 0)
                * source_breakdown[s].get("weight", 0)
                for s in directions
                if directions[s] == "buy"
            )
            sell_conf = sum(
                source_breakdown[s].get("confidence", 0)
                * source_breakdown[s].get("weight", 0)
                for s in directions
                if directions[s] == "sell"
            )
            if buy_conf > sell_conf:
                return "buy", vote_details
            elif sell_conf > buy_conf:
                return "sell", vote_details
            else:
                return "hold", vote_details

    def _calculate_risk_level(
        self, confidence: float, buy_votes: int, sell_votes: int, total_sources: int
    ) -> str:
        if total_sources == 0:
            return "HIGH"
        consensus = max(buy_votes, sell_votes) / total_sources
        if confidence >= 0.80 and consensus >= 0.75:
            return "LOW"
        elif confidence >= 0.70 and consensus >= 0.60:
            return "MEDIUM"
        elif confidence >= 0.65:
            return "HIGH"
        else:
            return "EXTREME"

    def _calculate_sl_tp(self, price: float, signal: str) -> tuple:
        if price <= 0:
            return None, None
        if signal == "buy":
            return round(price * STOP_LOSS_PERCENT, 2), round(
                price * TAKE_PROFIT_PERCENT, 2
            )
        elif signal == "sell":
            return round(price * (2 - STOP_LOSS_PERCENT), 2), round(
                price * (2 - TAKE_PROFIT_PERCENT), 2
            )
        else:
            return None, None

    def _generate_reasoning(
        self,
        coin: str,
        final_signal: str,
        confidence: float,
        risk_level: str,
        source_breakdown: dict,
    ) -> str:
        parts = [f"{coin.upper()}:", f"{final_signal.upper()} signal"]
        if confidence >= 0.80:
            parts.append("Very High Confidence")
        elif confidence >= 0.70:
            parts.append("High Confidence")
        elif confidence >= 0.65:
            parts.append("Moderate Confidence")
        else:
            parts.append("Low Confidence")
        parts.append(f"({confidence:.0%})")
        parts.append(f"Risk: {risk_level}")
        source_summary = [
            f"{s}({data.get('signal', 'N/A')}@{data.get('confidence', 0):.0%})"
            for s, data in source_breakdown.items()
        ]
        parts.append("| " + " | ".join(source_summary))
        return " | ".join(parts)
