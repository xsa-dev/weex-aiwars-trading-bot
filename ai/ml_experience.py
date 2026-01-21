"""ML Experience Tracker - Tracks model predictions and performance for adaptive trading."""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import aiosqlite

from utils.logger import add_log


@dataclass
class PredictionRecord:
    """Record of a single ML model prediction."""

    coin: str
    model: str
    predicted_direction: str
    predicted_change_percent: float
    confidence: float
    id: Optional[int] = None
    prediction_time: datetime = field(default_factory=datetime.now)
    was_correct: Optional[bool] = None


@dataclass
class ModelPerformance:
    """Performance metrics for a model on a specific coin."""

    model: str
    coin: str
    total_predictions: int = 0
    correct_predictions: int = 0
    avg_confidence: float = 0.0
    avg_error_percent: float = 0.0
    last_updated: datetime = field(default_factory=datetime.now)

    @property
    def accuracy(self) -> float:
        """Calculate prediction accuracy."""
        if self.total_predictions == 0:
            return 0.0
        return self.correct_predictions / self.total_predictions


class ExperienceDatabase:
    """Async SQLite database for storing and querying ML experience data."""

    def __init__(self, db_path: str):
        """Initialize database connection.

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        self._connection: Optional[aiosqlite.Connection] = None

    async def connect(self) -> None:
        """Connect to the database and create tables if needed."""
        self._connection = await aiosqlite.connect(self.db_path)
        await self._create_tables()
        add_log("ExperienceDatabase connected", level="DEBUG", logger_name="ml_experience")

    async def _create_tables(self) -> None:
        """Create database tables if they don't exist."""
        if self._connection is None:
            raise RuntimeError("Database not connected")

        await self._connection.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                coin TEXT NOT NULL,
                model TEXT NOT NULL,
                predicted_direction TEXT NOT NULL,
                predicted_change_percent REAL NOT NULL,
                confidence REAL NOT NULL,
                prediction_time TIMESTAMP NOT NULL,
                was_correct BOOLEAN,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        await self._connection.execute("""
            CREATE TABLE IF NOT EXISTS model_performance (
                model TEXT NOT NULL,
                coin TEXT NOT NULL,
                total_predictions INTEGER DEFAULT 0,
                correct_predictions INTEGER DEFAULT 0,
                avg_confidence REAL DEFAULT 0.0,
                avg_error_percent REAL DEFAULT 0.0,
                last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (model, coin)
            )
        """)

        await self._connection.execute("""
            CREATE INDEX IF NOT EXISTS idx_predictions_coin_model
            ON predictions(coin, model)
        """)

        await self._connection.execute("""
            CREATE INDEX IF NOT EXISTS idx_predictions_time
            ON predictions(prediction_time)
        """)

        await self._connection.commit()

    async def close(self) -> None:
        """Close database connection."""
        if self._connection:
            await self._connection.close()
            self._connection = None

    async def record_prediction(self, record: PredictionRecord) -> int:
        """Record a new prediction.

        Args:
            record: PredictionRecord to store

        Returns:
            ID of the inserted record
        """
        if self._connection is None:
            await self.connect()

        cursor = await self._connection.execute(
            """
            INSERT INTO predictions (
                coin, model, predicted_direction, predicted_change_percent,
                confidence, prediction_time, was_correct
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                record.coin,
                record.model,
                record.predicted_direction,
                record.predicted_change_percent,
                record.confidence,
                record.prediction_time.isoformat(),
                record.was_correct,
            ),
        )
        await self._connection.commit()
        prediction_id = cursor.lastrowid
        add_log(
            f"Recorded prediction: {record.model} on {record.coin}",
            level="DEBUG",
            data={"prediction_id": prediction_id},
            logger_name="ml_experience",
        )
        return prediction_id

    async def update_model_performance(
        self,
        model: str,
        coin: str,
        was_correct: bool,
        confidence: float,
        error_percent: float,
    ) -> None:
        """Update model performance metrics.

        Args:
            model: Model name
            coin: Coin symbol
            was_correct: Whether the prediction was correct
            confidence: Model confidence for this prediction
            error_percent: Prediction error percentage
        """
        if self._connection is None:
            await self.connect()

        # Get current performance
        cursor = await self._connection.execute(
            """
            SELECT total_predictions, correct_predictions, avg_confidence, avg_error_percent
            FROM model_performance WHERE model = ? AND coin = ?
            """,
            (model, coin),
        )
        row = await cursor.fetchone()

        if row:
            total_predictions, correct_predictions, avg_confidence, avg_error_percent = row
            new_total = total_predictions + 1
            new_correct = correct_predictions + (1 if was_correct else 0)
            # Running average for confidence and error
            new_avg_confidence = (avg_confidence * total_predictions + confidence) / new_total
            new_avg_error = (avg_error_percent * total_predictions + error_percent) / new_total

            await self._connection.execute(
                """
                UPDATE model_performance SET
                    total_predictions = ?,
                    correct_predictions = ?,
                    avg_confidence = ?,
                    avg_error_percent = ?,
                    last_updated = ?
                WHERE model = ? AND coin = ?
                """,
                (
                    new_total,
                    new_correct,
                    new_avg_confidence,
                    new_avg_error,
                    datetime.now().isoformat(),
                    model,
                    coin,
                ),
            )
        else:
            await self._connection.execute(
                """
                INSERT INTO model_performance (
                    model, coin, total_predictions, correct_predictions,
                    avg_confidence, avg_error_percent, last_updated
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    model,
                    coin,
                    1,
                    1 if was_correct else 0,
                    confidence,
                    error_percent,
                    datetime.now().isoformat(),
                ),
            )

        await self._connection.commit()

    async def get_model_performance(self, model: str, coin: str) -> Optional[ModelPerformance]:
        """Get performance metrics for a model on a specific coin.

        Args:
            model: Model name
            coin: Coin symbol

        Returns:
            ModelPerformance object or None if not found
        """
        if self._connection is None:
            await self.connect()

        cursor = await self._connection.execute(
            """
            SELECT model, coin, total_predictions, correct_predictions,
                   avg_confidence, avg_error_percent, last_updated
            FROM model_performance WHERE model = ? AND coin = ?
            """,
            (model, coin),
        )
        row = await cursor.fetchone()

        if row:
            return ModelPerformance(
                model=row[0],
                coin=row[1],
                total_predictions=row[2],
                correct_predictions=row[3],
                avg_confidence=row[4],
                avg_error_percent=row[5],
                last_updated=datetime.fromisoformat(row[6]) if isinstance(row[6], str) else row[6],
            )
        return None

    async def get_performance_summary(self) -> Dict[str, Dict[str, ModelPerformance]]:
        """Get performance summary for all models.

        Returns:
            Dictionary mapping coin -> {model: ModelPerformance}
        """
        if self._connection is None:
            await self.connect()

        cursor = await self._connection.execute(
            """
            SELECT model, coin, total_predictions, correct_predictions,
                   avg_confidence, avg_error_percent, last_updated
            FROM model_performance ORDER BY model, coin
            """
        )
        rows = await cursor.fetchall()

        summary: Dict[str, Dict[str, ModelPerformance]] = {}
        for row in rows:
            perf = ModelPerformance(
                model=row[0],
                coin=row[1],
                total_predictions=row[2],
                correct_predictions=row[3],
                avg_confidence=row[4],
                avg_error_percent=row[5],
                last_updated=datetime.fromisoformat(row[6]) if isinstance(row[6], str) else row[6],
            )
            if perf.model not in summary:
                summary[perf.model] = {}
            summary[perf.model][perf.coin] = perf

        return summary

    async def evaluate_predictions(
        self, evaluation_window_hours: int = 24
    ) -> List[PredictionRecord]:
        """Evaluate pending predictions against actual outcomes.

        This is a simplified implementation that marks predictions as correct
        based on whether price moved in the predicted direction.

        Args:
            evaluation_window_hours: Lookback window for evaluation

        Returns:
            List of evaluated prediction records
        """
        if self._connection is None:
            await self.connect()

        # In a real implementation, this would fetch actual price data
        # For now, we return empty list as placeholder
        add_log(
            f"Evaluated predictions within {evaluation_window_hours}h window",
            level="DEBUG",
            logger_name="ml_experience",
        )
        return []

    async def archive_predictions(self, archive_after_days: int = 30) -> int:
        """Archive old predictions to reduce database size.

        Args:
            archive_after_days: Age threshold for archiving

        Returns:
            Number of archived predictions
        """
        if self._connection is None:
            await self.connect()

        cutoff_date = datetime.now() - timedelta(days=archive_after_days)

        cursor = await self._connection.execute(
            """
            DELETE FROM predictions
            WHERE prediction_time < ? AND was_correct IS NOT NULL
            """,
            (cutoff_date.isoformat(),),
        )

        archived_count = cursor.rowcount
        await self._connection.commit()

        add_log(
            f"Archived {archived_count} old predictions",
            level="DEBUG",
            data={"archived_count": archived_count},
            logger_name="ml_experience",
        )

        return archived_count


class AdaptiveWeightCalculator:
    """Calculates adaptive weights for ML models based on performance."""

    def __init__(self, db: ExperienceDatabase):
        """Initialize calculator with database connection.

        Args:
            db: ExperienceDatabase instance
        """
        self.db = db

    async def calculate_weights(self, coin: str) -> Dict[str, float]:
        """Calculate adaptive weights for all models trading a coin.

        Args:
            coin: Coin symbol

        Returns:
            Dictionary mapping model name to weight (0-1)
        """
        performance_summary = await self.db.get_performance_summary()
        coin_performance = performance_summary.get(coin, {})

        if not coin_performance:
            # Default equal weights if no performance data
            add_log(
                f"No performance data for {coin}, using default weights",
                level="DEBUG",
                logger_name="ml_experience",
            )
            return {"GRU": 0.25, "LSTM": 0.25, "XGBoost": 0.25, "LightGBM": 0.25}

        # Calculate weights based on accuracy and confidence
        weights: Dict[str, float] = {}
        total_score = 0.0

        for model, perf in coin_performance.items():
            # Score based on accuracy (70%) and confidence (30%)
            accuracy_score = perf.accuracy * 0.7
            confidence_score = min(perf.avg_confidence, 1.0) * 0.3
            score = accuracy_score + confidence_score

            # Apply minimum threshold (models with < 40% accuracy get minimal weight)
            if perf.accuracy < 0.40:
                score = 0.05

            weights[model] = max(score, 0.05)  # Minimum 5% weight
            total_score += weights[model]

        # Normalize weights to sum to 1
        if total_score > 0:
            for model in weights:
                weights[model] /= total_score

        add_log(
            f"Calculated adaptive weights for {coin}: {weights}",
            level="DEBUG",
            data={"weights": weights},
            logger_name="ml_experience",
        )

        return weights


class MLExperienceTracker:
    """Main tracker for ML model experience and adaptive learning."""

    def __init__(
        self,
        db_path: str,
        evaluation_window_hours: int = 24,
        archive_after_days: int = 30,
    ):
        """Initialize ML experience tracker.

        Args:
            db_path: Path to SQLite database
            evaluation_window_hours: Hours to wait before evaluating predictions
            archive_after_days: Days before archiving old predictions
        """
        self.db_path = db_path
        self.evaluation_window_hours = evaluation_window_hours
        self.archive_after_days = archive_after_days
        self.db = ExperienceDatabase(db_path)
        self.weight_calculator = AdaptiveWeightCalculator(self.db)

    async def initialize(self) -> None:
        """Initialize database connection."""
        await self.db.connect()

    async def close(self) -> None:
        """Close database connection."""
        await self.db.close()

    async def get_adaptive_weights(self, coin: str) -> Dict[str, float]:
        """Get adaptive weights for a coin.

        Args:
            coin: Coin symbol

        Returns:
            Dictionary of model -> weight
        """
        return await self.weight_calculator.calculate_weights(coin)

    async def get_performance_report(self) -> Dict[str, Any]:
        """Generate performance report for all models.

        Returns:
            Report dictionary with generated_at and models sections
        """
        performance_summary = await self.db.get_performance_summary()

        report: Dict[str, Any] = {
            "generated_at": datetime.now().isoformat(),
            "models": {},
        }

        for coin, models in performance_summary.items():
            report["models"][coin] = {}
            for model, perf in models.items():
                report["models"][coin][model] = {
                    "total_predictions": perf.total_predictions,
                    "correct_predictions": perf.correct_predictions,
                    "accuracy": round(perf.accuracy, 4),
                    "avg_confidence": round(perf.avg_confidence, 4),
                    "avg_error_percent": round(perf.avg_error_percent, 4),
                    "last_updated": perf.last_updated.isoformat() if isinstance(perf.last_updated, datetime) else perf.last_updated,
                }

        add_log(
            f"Generated performance report: {len(performance_summary)} coins tracked",
            level="INFO",
            data={"coins_tracked": list(performance_summary.keys())},
            logger_name="ml_experience",
        )

        return report

    async def run_maintenance(self) -> Dict[str, int]:
        """Run maintenance tasks (evaluation and archiving).

        Returns:
            Dictionary with archived_predictions and evaluated_predictions counts
        """
        results: Dict[str, int] = {
            "archived_predictions": 0,
            "evaluated_predictions": 0,
        }

        # Archive old predictions
        archived = await self.db.archive_predictions(self.archive_after_days)
        results["archived_predictions"] = archived

        # Evaluate pending predictions
        evaluated = await self.db.evaluate_predictions(self.evaluation_window_hours)
        results["evaluated_predictions"] = len(evaluated)

        add_log(
            "Maintenance completed",
            level="DEBUG",
            data=results,
            logger_name="ml_experience",
        )

        return results

    async def record_prediction(self, record: PredictionRecord) -> int:
        """Record a new prediction.

        Args:
            record: PredictionRecord to store

        Returns:
            ID of the inserted record
        """
        return await self.db.record_prediction(record)
