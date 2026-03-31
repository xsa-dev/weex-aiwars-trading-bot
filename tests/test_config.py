"""
Unit tests for ai/config.py
"""

import pytest
from ai.config import (
    SOURCE_WEIGHTS,
    get_position_multiplier,
    CONFIDENCE_THRESHOLD_BUY,
    CONFIDENCE_THRESHOLD_SELL,
    EMERGENCY_DRAWDOWN_THRESHOLD,
    ML_AVAILABLE_MODELS,
    ML_PREDICTION_HOURS,
    MAX_RISK_PER_TRADE_USDT,
    MAX_TOTAL_RISK_USDT,
    MAX_OPEN_POSITIONS,
)


class TestSourceWeights:
    """Tests for source weights configuration."""

    def test_weights_sum_to_one(self):
        """Test that weights sum to 1.0."""
        total = sum(SOURCE_WEIGHTS.values())
        assert abs(total - 1.0) < 0.01

    def test_expected_sources(self):
        """Test that expected sources are present."""
        assert "technical" in SOURCE_WEIGHTS
        assert "ml_predictions" in SOURCE_WEIGHTS
        assert "news" in SOURCE_WEIGHTS
        assert "llm" not in SOURCE_WEIGHTS

    def test_ml_highest_weight(self):
        """Test that ML predictions has highest weight."""
        assert SOURCE_WEIGHTS["ml_predictions"] == 0.55

    def test_technical_second_weight(self):
        """Test that technical has second weight."""
        assert SOURCE_WEIGHTS["technical"] == 0.35


class TestPositionMultiplier:
    """Tests for position multiplier function."""

    def test_very_high_confidence(self):
        """Test multiplier for very high confidence (0.80-1.0)."""
        assert get_position_multiplier(0.85) == 1.5

    def test_high_confidence(self):
        """Test multiplier for high confidence (0.70-0.80)."""
        assert get_position_multiplier(0.75) == 1.2

    def test_medium_confidence(self):
        """Test multiplier for medium confidence (0.65-0.70)."""
        assert get_position_multiplier(0.67) == 0.8

    def test_low_confidence(self):
        """Test multiplier for low confidence (0.0-0.65)."""
        assert get_position_multiplier(0.60) == 0.5

    def test_boundary_values(self):
        """Test boundary values."""
        assert get_position_multiplier(0.80) == 1.5
        assert get_position_multiplier(0.70) == 1.2
        assert get_position_multiplier(0.65) == 0.8
        assert get_position_multiplier(0.0) == 0.5


class TestConfidenceThresholds:
    """Tests for confidence thresholds."""

    def test_buy_threshold(self):
        """Test buy threshold value."""
        assert CONFIDENCE_THRESHOLD_BUY == 0.70

    def test_sell_threshold(self):
        """Test sell threshold value."""
        assert CONFIDENCE_THRESHOLD_SELL == 0.70

    def test_thresholds_equal(self):
        """Test that buy and sell thresholds are equal."""
        assert CONFIDENCE_THRESHOLD_BUY == CONFIDENCE_THRESHOLD_SELL


class TestEmergencySettings:
    """Tests for emergency settings."""

    def test_emergency_threshold(self):
        """Test emergency drawdown threshold."""
        assert EMERGENCY_DRAWDOWN_THRESHOLD == 0.10


class TestMLConfiguration:
    """Tests for ML configuration."""

    def test_ml_models_list(self):
        """Test that ML models list is not empty."""
        assert len(ML_AVAILABLE_MODELS) > 0

    def test_ml_prediction_hours(self):
        """Test ML prediction hours configuration."""
        assert ML_PREDICTION_HOURS == 6

    def test_all_expected_models_present(self):
        """Test that all expected models are present."""
        expected = [
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
        for model in expected:
            assert model in ML_AVAILABLE_MODELS


class TestRiskConfiguration:
    """Tests for risk configuration."""

    def test_max_risk_per_trade(self):
        """Test max risk per trade."""
        assert MAX_RISK_PER_TRADE_USDT == 10.0

    def test_max_total_risk(self):
        """Test max total risk."""
        assert MAX_TOTAL_RISK_USDT == 80.0

    def test_max_positions(self):
        """Test max open positions."""
        assert MAX_OPEN_POSITIONS == 8

    def test_risk_values_relationship(self):
        """Test that risk values have correct relationship."""
        assert MAX_TOTAL_RISK_USDT > MAX_RISK_PER_TRADE_USDT
        assert MAX_OPEN_POSITIONS * MAX_RISK_PER_TRADE_USDT == MAX_TOTAL_RISK_USDT


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
