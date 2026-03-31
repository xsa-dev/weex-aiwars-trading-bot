import asyncio
import pytest
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

from ai.ml_experience import (
    ExperienceDatabase,
    PredictionRecord,
    ModelPerformance,
    MLExperienceTracker,
)


@pytest.fixture(scope="session")
def event_loop():
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def temp_db_path():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".db", delete=False) as f:
        db_path = f.name
    yield db_path
    try:
        Path(db_path).unlink()
    except FileNotFoundError:
        pass


@pytest.fixture
def db(temp_db_path):
    return ExperienceDatabase(temp_db_path)


@pytest.fixture
def tracker(temp_db_path):
    return MLExperienceTracker(
        db_path=temp_db_path,
        evaluation_window_hours=1,
        archive_after_days=1,
    )


@pytest.fixture
def sample_prediction_record():
    return PredictionRecord(
        id=None,
        coin="cmt_btcusdt",
        model="GRU",
        predicted_direction="up",
        predicted_change_percent=2.5,
        confidence=0.75,
        prediction_time=datetime.now() - timedelta(hours=6),
    )


@pytest.fixture
def sample_performance():
    return ModelPerformance(
        model="GRU",
        coin="cmt_btcusdt",
        total_predictions=100,
        correct_predictions=75,
        avg_confidence=0.80,
        avg_error_percent=1.5,
        last_updated=datetime.now(),
    )
