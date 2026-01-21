import pytest
from unittest.mock import MagicMock, AsyncMock

from ai.ml_experience import (
    PredictionRecord,
    ModelPerformance,
    ExperienceDatabase,
    AdaptiveWeightCalculator,
    MLExperienceTracker,
)


class TestPredictionRecord:
    def test_creation(self):
        record = PredictionRecord(
            coin="BTCUSDT",
            model="GRU",
            predicted_direction="up",
            predicted_change_percent=2.5,
            confidence=0.75,
        )
        assert record.coin == "BTCUSDT"
        assert record.model == "GRU"
        assert record.predicted_direction == "up"
        assert record.was_correct is None


class TestModelPerformance:
    def test_accuracy_with_predictions(self):
        perf = ModelPerformance(
            model="GRU",
            coin="BTCUSDT",
            total_predictions=100,
            correct_predictions=75,
        )
        assert perf.accuracy == 0.75

    def test_accuracy_with_no_predictions(self):
        perf = ModelPerformance(model="GRU", coin="BTCUSDT")
        assert perf.accuracy == 0.0


class TestExperienceDatabase:
    @pytest.mark.asyncio
    async def test_record_prediction(self, db):
        record = PredictionRecord(
            coin="BTCUSDT",
            model="GRU",
            predicted_direction="up",
            predicted_change_percent=2.5,
            confidence=0.75,
        )
        prediction_id = await db.record_prediction(record)
        assert prediction_id is not None
        assert prediction_id > 0

    @pytest.mark.asyncio
    async def test_update_model_performance(self, db):
        await db.update_model_performance(
            model="GRU",
            coin="BTCUSDT",
            was_correct=True,
            confidence=0.75,
            error_percent=1.5,
        )
        perf = await db.get_model_performance("GRU", "BTCUSDT")
        assert perf.total_predictions == 1
        assert perf.accuracy == 1.0

    @pytest.mark.asyncio
    async def test_get_performance_summary(self, db):
        for model in ["GRU", "LSTM"]:
            await db.update_model_performance(
                model=model,
                coin="BTCUSDT",
                was_correct=True,
                confidence=0.75,
                error_percent=1.0,
            )
        summary = await db.get_performance_summary()
        assert "GRU" in summary
        assert "LSTM" in summary


class TestAdaptiveWeightCalculator:
    @pytest.fixture
    def mock_db(self):
        db = MagicMock(spec=ExperienceDatabase)
        db.get_performance_summary = AsyncMock(
            return_value={
                "GRU": {
                    "BTCUSDT": ModelPerformance(
                        model="GRU",
                        coin="BTCUSDT",
                        total_predictions=100,
                        correct_predictions=75,
                        avg_confidence=0.80,
                        avg_error_percent=1.5,
                    )
                }
            }
        )
        return db

    @pytest.mark.asyncio
    async def test_calculate_weights(self, mock_db):
        calculator = AdaptiveWeightCalculator(mock_db)
        weights = await calculator.calculate_weights("BTCUSDT")
        assert "GRU" in weights
        assert all(0 <= w <= 1 for w in weights.values())


class TestMLExperienceTracker:
    @pytest.fixture
    def tracker(self, temp_db_path):
        return MLExperienceTracker(
            db_path=temp_db_path,
            evaluation_window_hours=1,
            archive_after_days=1,
        )

    @pytest.mark.asyncio
    async def test_get_adaptive_weights(self, tracker):
        weights = await tracker.get_adaptive_weights("BTCUSDT")
        assert isinstance(weights, dict)
        assert len(weights) > 0

    @pytest.mark.asyncio
    async def test_get_performance_report(self, tracker):
        report = await tracker.get_performance_report()
        assert "generated_at" in report
        assert "models" in report

    @pytest.mark.asyncio
    async def test_run_maintenance(self, tracker):
        results = await tracker.run_maintenance()
        assert "archived_predictions" in results
        assert "evaluated_predictions" in results
