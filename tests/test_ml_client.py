"""
Unit tests for ai/ml_client.py
"""

import pytest
from unittest.mock import AsyncMock, MagicMock

# Mock the aiohttp module before importing MLClient
import sys

# Create mock module
mock_aiohttp = MagicMock()
mock_aiohttp.ClientSession = MagicMock
mock_aiohttp.ClientTimeout = MagicMock(return_value=MagicMock())
sys.modules["aiohttp"] = mock_aiohttp

from ai.ml_client import MLClient, MLPrediction, EnsembleMLPrediction


class TestMLClient:
    """Tests for MLClient class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.client = MLClient(
            host="http://localhost:8000",
            prediction_hours=6,
            timeout=30,
        )

    def teardown_method(self):
        """Clean up after tests."""
        # Reset session
        self.client.session = None

    def test_coin_to_symbol_btc(self):
        """Test coin symbol conversion for BTC."""
        symbol = self.client._coin_to_symbol("cmt_btcusdt")
        assert symbol == "BTC/USDT"

    def test_coin_to_symbol_eth(self):
        """Test coin symbol conversion for ETH."""
        symbol = self.client._coin_to_symbol("cmt_ethusdt")
        assert symbol == "ETH/USDT"

    def test_coin_to_symbol_sol(self):
        """Test coin symbol conversion for SOL."""
        symbol = self.client._coin_to_symbol("cmt_solusdt")
        assert symbol == "SOL/USDT"

    def test_coin_to_symbol_unknown(self):
        """Test coin symbol conversion for unknown coin."""
        symbol = self.client._coin_to_symbol("cmt_unknownusdt")
        assert symbol == "UNKNOWN/USDT"


class TestMLPrediction:
    """Tests for MLPrediction dataclass."""

    def test_ml_prediction_creation(self):
        """Test MLPrediction object creation."""
        prediction = MLPrediction(
            coin="cmt_btcusdt",
            model="GRU",
            prediction_hours=6,
            current_price=50000.0,
            predicted_price=51000.0,
            confidence=0.85,
            direction="up",
            change_percent=2.0,
            timestamp="2026-01-18 15:00:00",
            raw_response={},
        )

        assert prediction.coin == "cmt_btcusdt"
        assert prediction.model == "GRU"
        assert prediction.direction == "up"
        assert prediction.confidence == 0.85
        assert prediction.change_percent == 2.0

    def test_ml_prediction_direction_up(self):
        """Test MLPrediction with up direction."""
        prediction = MLPrediction(
            coin="cmt_btcusdt",
            model="GRU",
            prediction_hours=6,
            current_price=50000.0,
            predicted_price=51000.0,
            confidence=0.85,
            direction="up",
            change_percent=2.0,
            timestamp="2026-01-18 15:00:00",
            raw_response={},
        )

        assert prediction.direction == "up"

    def test_ml_prediction_direction_down(self):
        """Test MLPrediction with down direction."""
        prediction = MLPrediction(
            coin="cmt_btcusdt",
            model="GRU",
            prediction_hours=6,
            current_price=50000.0,
            predicted_price=49000.0,
            confidence=0.75,
            direction="down",
            change_percent=-2.0,
            timestamp="2026-01-18 15:00:00",
            raw_response={},
        )

        assert prediction.direction == "down"

    def test_ml_prediction_direction_neutral(self):
        """Test MLPrediction with neutral direction."""
        prediction = MLPrediction(
            coin="cmt_btcusdt",
            model="GRU",
            prediction_hours=6,
            current_price=50000.0,
            predicted_price=50000.0,
            confidence=0.50,
            direction="neutral",
            change_percent=0.0,
            timestamp="2026-01-18 15:00:00",
            raw_response={},
        )

        assert prediction.direction == "neutral"


class TestEnsembleMLPrediction:
    """Tests for EnsembleMLPrediction dataclass."""

    def test_ensemble_prediction_creation(self):
        """Test EnsembleMLPrediction object creation."""
        ensemble = EnsembleMLPrediction(
            coin="cmt_btcusdt",
            direction="up",
            confidence=0.78,
            avg_change=1.5,
            model_count=10,
            models_used=["GRU", "LSTM", "XGBRegressor"],
            individual_predictions=[
                {"model": "GRU", "direction": "up", "confidence": 0.85},
                {"model": "LSTM", "direction": "up", "confidence": 0.80},
            ],
        )

        assert ensemble.coin == "cmt_btcusdt"
        assert ensemble.direction == "up"
        assert ensemble.confidence == 0.78
        assert ensemble.model_count == 10
        assert len(ensemble.models_used) == 3
        assert len(ensemble.individual_predictions) == 2

    def test_ensemble_prediction_empty(self):
        """Test EnsembleMLPrediction with no predictions."""
        ensemble = EnsembleMLPrediction(
            coin="cmt_btcusdt",
            direction="neutral",
            confidence=0.0,
            avg_change=0.0,
            model_count=0,
            models_used=[],
            individual_predictions=[],
        )

        assert ensemble.direction == "neutral"
        assert ensemble.confidence == 0.0
        assert ensemble.model_count == 0


class TestMLClientModels:
    """Tests for ML client model availability."""

    def test_available_models_list(self):
        """Test that all expected models are available."""
        expected_models = [
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

        assert self.client.models == expected_models
        assert len(self.client.models) == 12


class TestMLClientAsync:
    """Tests for async methods of MLClient."""

    @pytest.mark.asyncio
    async def test_get_session_creates_session(self):
        """Test that get_session creates a new session."""
        session = await self.client._get_session()
        assert session is not None
        assert not session.closed

    @pytest.mark.asyncio
    async def test_get_session_reuses_session(self):
        """Test that get_session reuses existing session."""
        session1 = await self.client._get_session()
        session2 = await self.client._get_session()
        assert session1 == session2

    @pytest.mark.asyncio
    async def test_get_prediction_returns_none_on_error(self):
        """Test that get_prediction returns None on API error."""
        # Mock the session to return error
        mock_response = AsyncMock()
        mock_response.status = 500
        mock_response.json = AsyncMock(return_value={})

        mock_session = AsyncMock()
        mock_session.post = MagicMock(
            return_value=AsyncMock(
                __aenter__=AsyncMock(return_value=mock_response), __aexit__=AsyncMock()
            )
        )

        self.client.session = mock_session

        result = await self.client.get_prediction("cmt_btcusdt", "GRU", 6)
        assert result is None


class TestMLClientIntegration:
    """Integration tests for MLClient (require mock server)."""

    @pytest.fixture
    def mock_api_response(self):
        """Mock successful API response."""
        return {
            "company": "BTC/USDT",
            "next_hour_prediction": {
                "open": 50000.0,
                "high": 50500.0,
                "low": 49800.0,
                "close": 50200.0,
                "volume": 10000.0,
            },
            "prediction_summary": {
                "confidence": 0.85,
                "24h_change": "↑2.5%",
                "start_price": 50200.0,
                "end_price": 51500.0,
            },
            "confidence_scores": {
                "open": 0.88,
                "high": 0.86,
                "low": 0.87,
                "close": 0.85,
                "volume": 0.75,
            },
            "prediction_date": "2026-01-18 15:00:00",
        }

    def test_parse_change_positive(self):
        """Test parsing positive change."""
        change_str = "↑2.5%"
        result = float(change_str.replace("↑", "").replace("↓", "").replace("%", ""))
        assert result == 2.5

    def test_parse_change_negative(self):
        """Test parsing negative change."""
        change_str = "↓3.2%"
        result = float(change_str.replace("↑", "").replace("↓", "").replace("%", ""))
        assert result == -3.2

    def test_parse_change_neutral(self):
        """Test parsing neutral change."""
        change_str = "0%"
        result = float(change_str.replace("↑", "").replace("↓", "").replace("%", ""))
        assert result == 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
