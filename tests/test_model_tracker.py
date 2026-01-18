"""
Unit tests for ai/model_tracker.py
"""

import pytest
import tempfile
import os

from ai.model_tracker import ModelTracker, ModelPerformance


class TestModelPerformance:
    """Tests for ModelPerformance dataclass."""

    def test_creation(self):
        """Test ModelPerformance creation."""
        perf = ModelPerformance(
            model_name="GRU",
            total_predictions=100,
            correct_predictions=75,
            accuracy=0.75,
            f1_score=0.72,
            avg_confidence=0.80,
        )

        assert perf.model_name == "GRU"
        assert perf.total_predictions == 100
        assert perf.accuracy == 0.75
        assert perf.is_active is True

    def test_default_values(self):
        """Test default values."""
        perf = ModelPerformance(model_name="LSTM")

        assert perf.total_predictions == 0
        assert perf.correct_predictions == 0
        assert perf.accuracy == 0.0
        assert perf.f1_score == 0.0
        assert perf.is_active is True


class TestModelTracker:
    """Tests for ModelTracker class."""

    @pytest.fixture
    def temp_storage_file(self):
        """Create temporary storage file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write("{}")
            temp_path = f.name

        yield temp_path

        if os.path.exists(temp_path):
            os.remove(temp_path)

    def setup_method(self, temp_storage_file):
        """Set up test fixtures."""
        self.tracker = ModelTracker(
            storage_file=temp_storage_file,
            accuracy_threshold=0.50,
            min_trades_for_eval=5,
        )

    def test_record_prediction_correct(self):
        """Test recording a correct prediction."""
        self.tracker.record_prediction(
            model_name="GRU",
            coin="cmt_btcusdt",
            predicted_direction="up",
            actual_direction="up",
            confidence=0.85,
            predicted_change=2.5,
            actual_change=3.0,
        )

        assert "GRU" in self.tracker.performances
        perf = self.tracker.performances["GRU"]
        assert perf.total_predictions == 1
        assert perf.correct_predictions == 1
        assert perf.accuracy == 1.0

    def test_record_prediction_incorrect(self):
        """Test recording an incorrect prediction."""
        self.tracker.record_prediction(
            model_name="GRU",
            coin="cmt_btcusdt",
            predicted_direction="up",
            actual_direction="down",
            confidence=0.85,
            predicted_change=2.5,
            actual_change=-3.0,
        )

        perf = self.tracker.performances["GRU"]
        assert perf.total_predictions == 1
        assert perf.correct_predictions == 0
        assert perf.accuracy == 0.0

    def test_record_multiple_predictions(self):
        """Test recording multiple predictions."""
        predictions = [
            ("up", "up"),
            ("up", "up"),
            ("down", "up"),
            ("up", "up"),
            ("down", "down"),
        ]

        for predicted, actual in predictions:
            self.tracker.record_prediction(
                model_name="XGBRegressor",
                coin="cmt_btcusdt",
                predicted_direction=predicted,
                actual_direction=actual,
                confidence=0.75,
                predicted_change=2.0,
                actual_change=2.0,
            )

        perf = self.tracker.performances["XGBRegressor"]
        assert perf.total_predictions == 5
        assert perf.correct_predictions == 3
        assert perf.accuracy == 0.6  # 3/5

    def test_disable_underperforming_model(self):
        """Test that underperforming models get disabled."""
        for _ in range(10):
            self.tracker.record_prediction(
                model_name="BadModel",
                coin="cmt_btcusdt",
                predicted_direction="up",
                actual_direction="down",
                confidence=0.70,
                predicted_change=2.0,
                actual_change=-2.0,
            )

        perf = self.tracker.performances["BadModel"]
        assert perf.is_active is False
        assert "Accuracy" in perf.disabled_reason

    def test_get_active_models(self):
        """Test getting active models."""
        self.tracker.record_prediction(
            model_name="GoodModel",
            coin="cmt_btcusdt",
            predicted_direction="up",
            actual_direction="up",
            confidence=0.85,
            predicted_change=2.0,
            actual_change=2.0,
        )

        self.tracker.record_prediction(
            model_name="BadModel",
            coin="cmt_btcusdt",
            predicted_direction="up",
            actual_direction="down",
            confidence=0.85,
            predicted_change=2.0,
            actual_change=-2.0,
        )

        for _ in range(9):
            self.tracker.record_prediction(
                model_name="BadModel",
                coin="cmt_btcusdt",
                predicted_direction="up",
                actual_direction="down",
                confidence=0.85,
                predicted_change=2.0,
                actual_change=-2.0,
            )

        active = self.tracker.get_active_models()
        assert "GoodModel" in active
        assert "BadModel" not in active

    def test_get_best_models(self):
        """Test getting best performing models."""
        for _ in range(10):
            self.tracker.record_prediction(
                model_name="GoodModel",
                coin="cmt_btcusdt",
                predicted_direction="up",
                actual_direction="up",
                confidence=0.85,
                predicted_change=2.0,
                actual_change=2.0,
            )

        for _ in range(10):
            self.tracker.record_prediction(
                model_name="MediumModel",
                coin="cmt_btcusdt",
                predicted_direction="up",
                actual_direction="up",
                confidence=0.70,
                predicted_change=2.0,
                actual_change=1.0,
            )

        best = self.tracker.get_best_models(top_n=1)
        assert len(best) == 1
        assert best[0][0] == "GoodModel"

    def test_reset_model(self):
        """Test resetting a specific model."""
        for _ in range(5):
            self.tracker.record_prediction(
                model_name="TestModel",
                coin="cmt_btcusdt",
                predicted_direction="up",
                actual_direction="up",
                confidence=0.85,
                predicted_change=2.0,
                actual_change=2.0,
            )

        assert self.tracker.performances["TestModel"].total_predictions == 5

        self.tracker.reset_model("TestModel")

        perf = self.tracker.performances["TestModel"]
        assert perf.total_predictions == 0
        assert perf.accuracy == 0.0

    def test_get_evaluation_report(self):
        """Test getting evaluation report."""
        for _ in range(3):
            self.tracker.record_prediction(
                model_name="TestModel",
                coin="cmt_btcusdt",
                predicted_direction="up",
                actual_direction="up",
                confidence=0.80,
                predicted_change=2.0,
                actual_change=2.0,
            )

        report = self.tracker.get_evaluation_report()

        assert "timestamp" in report
        assert "summary" in report
        assert report["summary"]["total_models"] == 1
        assert report["summary"]["active_models"] == 1

    def test_persistence(self, temp_storage_file):
        """Test that data persists to file."""
        self.tracker.record_prediction(
            model_name="PersistentModel",
            coin="cmt_btcusdt",
            predicted_direction="up",
            actual_direction="up",
            confidence=0.85,
            predicted_change=2.0,
            actual_change=2.0,
        )

        new_tracker = ModelTracker(storage_file=temp_storage_file)

        assert "PersistentModel" in new_tracker.performances
        perf = new_tracker.performances["PersistentModel"]
        assert perf.total_predictions == 1


class TestModelTrackerEdgeCases:
    """Tests for edge cases in ModelTracker."""

    @pytest.fixture
    def temp_storage_file(self):
        """Create temporary storage file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write("{}")
            temp_path = f.name

        yield temp_path

        if os.path.exists(temp_path):
            os.remove(temp_path)

    def setup_method(self, temp_storage_file):
        """Set up test fixtures."""
        self.tracker = ModelTracker(
            storage_file=temp_storage_file,
            accuracy_threshold=0.50,
            min_trades_for_eval=10,
        )

    def test_model_not_disabled_before_min_trades(self):
        """Test model not disabled before reaching min trades."""
        for _ in range(5):
            self.tracker.record_prediction(
                model_name="NewModel",
                coin="cmt_btcusdt",
                predicted_direction="up",
                actual_direction="down",
                confidence=0.70,
                predicted_change=2.0,
                actual_change=-2.0,
            )

        perf = self.tracker.performances["NewModel"]
        assert perf.is_active is True

    def test_empty_trades_f1_calculation(self):
        """Test F1 calculation with empty trades."""
        perf = ModelPerformance(model_name="EmptyModel")
        assert perf.f1_score == 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
