"""
Unit tests for ai/ensemble_voter.py
"""

import pytest

from ai.ensemble_voter import (
    EnsembleVoter,
    EnsembleSignal,
    TechnicalSignal,
    LLMSignal,
    MLPredictionSignal,
    NewsSentimentSignal,
)
from ai.config import SOURCE_WEIGHTS


class TestEnsembleVoter:
    """Tests for EnsembleVoter class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.voter = EnsembleVoter(
            weights=SOURCE_WEIGHTS,
            buy_threshold=0.70,
            sell_threshold=0.70,
            hold_threshold=0.30,
        )

    def test_parse_direction_buy(self):
        """Test parsing buy direction."""
        assert self.voter._parse_direction("long") == "buy"
        assert self.voter._parse_direction("buy") == "buy"
        assert self.voter._parse_direction("bullish") == "buy"
        assert self.voter._parse_direction("up") == "buy"
        assert self.voter._parse_direction("overbought") == "buy"

    def test_parse_direction_sell(self):
        """Test parsing sell direction."""
        assert self.voter._parse_direction("short") == "sell"
        assert self.voter._parse_direction("sell") == "sell"
        assert self.voter._parse_direction("bearish") == "sell"
        assert self.voter._parse_direction("down") == "sell"
        assert self.voter._parse_direction("oversold") == "sell"

    def test_parse_direction_hold(self):
        """Test parsing hold direction."""
        assert self.voter._parse_direction("hold") == "hold"
        assert self.voter._parse_direction("neutral") == "hold"
        assert self.voter._parse_direction("unknown") == "hold"

    def test_vote_technical_only(self):
        """Test voting with technical signal only."""
        technical = TechnicalSignal(
            signal="buy",
            confidence=0.80,
            reasoning="RSI oversold",
            indicators_used={"rsi": 25, "adx": 30},
        )

        result = self.voter.vote(
            coin="cmt_btcusdt",
            price=50000.0,
            technical=technical,
        )

        assert result.signal == "buy"
        assert result.confidence == 0.80
        assert result.risk_level in ["LOW", "MEDIUM", "HIGH"]
        assert result.stop_loss == 49000.0  # 2% below
        assert result.take_profit == 52500.0  # 5% above

    def test_vote_with_ml_prediction(self):
        """Test voting with ML prediction."""
        technical = TechnicalSignal(
            signal="buy",
            confidence=0.70,
            reasoning="RSI oversold",
        )

        ml = MLPredictionSignal(
            direction="up",
            confidence=0.85,
            avg_change=2.5,
            model_count=10,
            models_used=["GRU", "LSTM", "XGBRegressor"],
        )

        result = self.voter.vote(
            coin="cmt_btcusdt",
            price=50000.0,
            technical=technical,
            ml=ml,
        )

        assert result.signal == "buy"
        assert result.confidence > 0.70
        assert "ml_predictions" in result.source_breakdown

    def test_vote_with_news_sentiment(self):
        """Test voting with news sentiment."""
        technical = TechnicalSignal(
            signal="hold",
            confidence=0.50,
            reasoning="RSI neutral",
        )

        news = NewsSentimentSignal(
            sentiment="bullish",
            confidence=0.75,
            headline_count=10,
        )

        result = self.voter.vote(
            coin="cmt_btcusdt",
            price=50000.0,
            technical=technical,
            news=news,
        )

        assert result.signal == "buy"  # News bullish should tip the balance
        assert "news" in result.source_breakdown

    def test_vote_disagreement_hold(self):
        """Test voting when sources disagree."""
        technical = TechnicalSignal(
            signal="buy",
            confidence=0.60,
            reasoning="Weak bullish",
        )

        ml = MLPredictionSignal(
            direction="down",
            confidence=0.60,
            avg_change=-1.5,
            model_count=10,
            models_used=["GRU", "LSTM"],
        )

        news = NewsSentimentSignal(
            sentiment="bearish",
            confidence=0.60,
            headline_count=5,
        )

        result = self.voter.vote(
            coin="cmt_btcusdt",
            price=50000.0,
            technical=technical,
            ml=ml,
            news=news,
        )

        # Should be hold due to disagreement
        assert result.signal == "hold"

    def test_vote_emergency_mode(self):
        """Test emergency mode when drawdown exceeds threshold."""
        technical = TechnicalSignal(
            signal="buy",
            confidence=0.90,
            reasoning="Strong signal",
        )

        result = self.voter.vote(
            coin="cmt_btcusdt",
            price=50000.0,
            technical=technical,
            current_portfolio_drawdown=0.15,  # 15% drawdown
        )

        assert result.signal == "hold"
        assert result.is_emergency is True
        assert result.confidence == 0.0
        assert "EMERGENCY" in result.reasoning

    def test_calculate_sl_tp_buy(self):
        """Test SL/TP calculation for buy signal."""
        stop_loss, take_profit = self.voter._calculate_sl_tp(50000.0, "buy")

        assert stop_loss == 49000.0  # 2% below
        assert take_profit == 52500.0  # 5% above

    def test_calculate_sl_tp_sell(self):
        """Test SL/TP calculation for sell signal."""
        stop_loss, take_profit = self.voter._calculate_sl_tp(50000.0, "sell")

        assert stop_loss == 51000.0  # 2% above
        assert take_profit == 47500.0  # 5% below

    def test_calculate_sl_tp_hold(self):
        """Test SL/TP calculation for hold signal."""
        stop_loss, take_profit = self.voter._calculate_sl_tp(50000.0, "hold")

        assert stop_loss is None
        assert take_profit is None

    def test_calculate_risk_level_low(self):
        """Test risk level LOW."""
        assert self.voter._calculate_risk_level(0.85, 3, 0, 4) == "LOW"

    def test_calculate_risk_level_medium(self):
        """Test risk level MEDIUM."""
        assert self.voter._calculate_risk_level(0.75, 2, 1, 4) == "MEDIUM"

    def test_calculate_risk_level_high(self):
        """Test risk level HIGH."""
        assert self.voter._calculate_risk_level(0.68, 2, 1, 4) == "HIGH"

    def test_calculate_risk_level_extreme(self):
        """Test risk level EXTREME."""
        assert self.voter._calculate_risk_level(0.50, 1, 1, 4) == "EXTREME"

    def test_redistribute_weight(self):
        """Test weight redistribution when source is unavailable."""
        initial_weights = self.voter.weights.copy()

        self.voter._redistribute_weight("llm", 0.5)

        # llm should be removed
        assert "llm" not in self.voter.weights

        # Other weights should increase
        for source, weight in self.voter.weights.items():
            if source in initial_weights:
                assert weight > initial_weights[source]

    def test_vote_direction_buy_wins(self):
        """Test vote direction when buy has most votes."""
        source_breakdown = {
            "technical": {"direction": "buy", "confidence": 0.7, "weight": 0.3},
            "llm": {"direction": "buy", "confidence": 0.6, "weight": 0.25},
            "ml_predictions": {"direction": "hold", "confidence": 0.5, "weight": 0.30},
            "news": {"direction": "hold", "confidence": 0.5, "weight": 0.15},
        }

        signal, vote_details = self.voter._vote_direction(source_breakdown)

        assert signal == "buy"
        assert vote_details["buy_votes"] == 2
        assert vote_details["sell_votes"] == 0

    def test_vote_direction_sell_wins(self):
        """Test vote direction when sell has most votes."""
        source_breakdown = {
            "technical": {"direction": "hold", "confidence": 0.5, "weight": 0.3},
            "llm": {"direction": "sell", "confidence": 0.7, "weight": 0.25},
            "ml_predictions": {"direction": "sell", "confidence": 0.8, "weight": 0.30},
            "news": {"direction": "hold", "confidence": 0.5, "weight": 0.15},
        }

        signal, vote_details = self.voter._vote_direction(source_breakdown)

        assert signal == "sell"
        assert vote_details["sell_votes"] == 2
        assert vote_details["buy_votes"] == 0

    def test_generate_reasoning(self):
        """Test reasoning generation."""
        technical = TechnicalSignal(
            signal="buy",
            confidence=0.80,
            reasoning="RSI oversold",
        )

        result = self.voter.vote(
            coin="cmt_btcusdt",
            price=50000.0,
            technical=technical,
        )

        assert "BTCUSDT" in result.reasoning
        assert "BUY" in result.reasoning
        assert (
            "Very High Confidence" in result.reasoning
            or "High Confidence" in result.reasoning
        )


class TestTechnicalSignal:
    """Tests for TechnicalSignal dataclass."""

    def test_technical_signal_creation(self):
        """Test TechnicalSignal object creation."""
        signal = TechnicalSignal(
            signal="buy",
            confidence=0.75,
            reasoning="RSI oversold",
            indicators_used={"rsi": 25, "adx": 30, "macd": 0.5},
        )

        assert signal.signal == "buy"
        assert signal.confidence == 0.75
        assert signal.indicators_used["rsi"] == 25


class TestLLMSignal:
    """Tests for LLMSignal dataclass."""

    def test_llm_signal_creation(self):
        """Test LLMSignal object creation."""
        signal = LLMSignal(
            signal="buy",
            confidence=0.85,
            reasoning="Strong bullish pattern",
            model="deepseek-v3.2",
        )

        assert signal.signal == "buy"
        assert signal.confidence == 0.85
        assert signal.model == "deepseek-v3.2"


class TestMLPredictionSignal:
    """Tests for MLPredictionSignal dataclass."""

    def test_ml_prediction_signal_creation(self):
        """Test MLPredictionSignal object creation."""
        signal = MLPredictionSignal(
            direction="up",
            confidence=0.78,
            avg_change=2.5,
            model_count=10,
            models_used=["GRU", "LSTM", "XGBRegressor"],
        )

        assert signal.direction == "up"
        assert signal.confidence == 0.78
        assert signal.model_count == 10
        assert len(signal.models_used) == 3


class TestNewsSentimentSignal:
    """Tests for NewsSentimentSignal dataclass."""

    def test_news_sentiment_signal_creation(self):
        """Test NewsSentimentSignal object creation."""
        signal = NewsSentimentSignal(
            sentiment="bullish",
            confidence=0.70,
            headline_count=15,
        )

        assert signal.sentiment == "bullish"
        assert signal.confidence == 0.70
        assert signal.headline_count == 15


class TestEnsembleSignal:
    """Tests for EnsembleSignal dataclass."""

    def test_ensemble_signal_creation(self):
        """Test EnsembleSignal object creation."""
        signal = EnsembleSignal(
            coin="cmt_btcusdt",
            signal="buy",
            confidence=0.78,
            reasoning="Multiple sources agree",
            source_breakdown={},
            risk_level="MEDIUM",
            position_size=0.0012,
            stop_loss=49000.0,
            take_profit=52500.0,
            is_emergency=False,
        )

        assert signal.coin == "cmt_btcusdt"
        assert signal.signal == "buy"
        assert signal.risk_level == "MEDIUM"
        assert signal.stop_loss == 49000.0
        assert signal.take_profit == 52500.0

    def test_ensemble_signal_emergency(self):
        """Test EnsembleSignal with emergency mode."""
        signal = EnsembleSignal(
            coin="cmt_btcusdt",
            signal="hold",
            confidence=0.0,
            reasoning="Emergency mode",
            source_breakdown={},
            risk_level="EXTREME",
            position_size=0.0,
            stop_loss=None,
            take_profit=None,
            is_emergency=True,
        )

        assert signal.is_emergency is True
        assert signal.confidence == 0.0
        assert signal.position_size == 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
