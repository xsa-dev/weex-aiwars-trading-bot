"""Model performance tracking and dynamic weight adjustment."""

import json
import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass
class ModelPerformance:
    """Performance metrics for a single model."""

    model_name: str
    total_predictions: int = 0
    correct_predictions: int = 0
    accuracy: float = 0.0
    f1_score: float = 0.0
    avg_confidence: float = 0.0
    last_updated: str = ""
    trades: list[dict[str, Any]] = field(default_factory=list)
    is_active: bool = True
    disabled_reason: str = ""


class ModelTracker:
    """Track and analyze ML model performance over time."""

    def __init__(
        self,
        storage_file: str = "model_performance.json",
        accuracy_threshold: float = 0.50,
        min_trades_for_eval: int = 10,
    ):
        self.storage_file = storage_file
        self.accuracy_threshold = accuracy_threshold
        self.min_trades_for_eval = min_trades_for_eval
        self.performances: dict[str, ModelPerformance] = {}
        self.load()

    def load(self):
        """Load performance data from file."""
        if not os.path.exists(self.storage_file):
            return
        try:
            with open(self.storage_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                for model_name, perf_data in data.items():
                    self.performances[model_name] = ModelPerformance(
                        model_name=model_name,
                        total_predictions=perf_data.get("total_predictions", 0),
                        correct_predictions=perf_data.get("correct_predictions", 0),
                        accuracy=perf_data.get("accuracy", 0.0),
                        f1_score=perf_data.get("f1_score", 0.0),
                        avg_confidence=perf_data.get("avg_confidence", 0.0),
                        last_updated=perf_data.get("last_updated", ""),
                        trades=perf_data.get("trades", []),
                        is_active=perf_data.get("is_active", True),
                        disabled_reason=perf_data.get("disabled_reason", ""),
                    )
        except Exception:
            pass

    def save(self):
        """Save performance data to file."""
        data = {}
        for model_name, perf in self.performances.items():
            data[model_name] = {
                "model_name": perf.model_name,
                "total_predictions": perf.total_predictions,
                "correct_predictions": perf.correct_predictions,
                "accuracy": perf.accuracy,
                "f1_score": perf.f1_score,
                "avg_confidence": perf.avg_confidence,
                "last_updated": perf.last_updated,
                "trades": perf.trades[-100:],
                "is_active": perf.is_active,
                "disabled_reason": perf.disabled_reason,
            }
        try:
            with open(self.storage_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception:
            pass

    def record_prediction(
        self,
        model_name: str,
        coin: str,
        predicted_direction: str,
        actual_direction: str,
        confidence: float,
        predicted_change: float,
        actual_change: float,
    ):
        """Record a prediction outcome for a model."""
        if model_name not in self.performances:
            self.performances[model_name] = ModelPerformance(model_name=model_name)

        perf = self.performances[model_name]

        predicted_sign = (
            1
            if predicted_direction == "up"
            else (-1 if predicted_direction == "down" else 0)
        )
        actual_sign = (
            1 if actual_direction == "up" else (-1 if actual_direction == "down" else 0)
        )
        is_correct = predicted_sign == actual_sign

        trade = {
            "timestamp": datetime.now().isoformat(),
            "coin": coin,
            "predicted_direction": predicted_direction,
            "actual_direction": actual_direction,
            "correct": is_correct,
            "confidence": confidence,
            "predicted_change": predicted_change,
            "actual_change": actual_change,
        }
        perf.trades.append(trade)

        perf.total_predictions += 1
        if is_correct:
            perf.correct_predictions += 1

        perf.accuracy = perf.correct_predictions / perf.total_predictions
        total_confidence = sum(t["confidence"] for t in perf.trades)
        perf.avg_confidence = total_confidence / len(perf.trades)
        perf.f1_score = self._calculate_f1(perf.trades)
        perf.last_updated = datetime.now().isoformat()

        if self._should_disable_model(perf):
            perf.is_active = False
            perf.disabled_reason = f"Accuracy {perf.accuracy:.1%} below threshold {self.accuracy_threshold:.0%}"

        self.save()

    def _calculate_f1(self, trades: list[dict]) -> float:
        if not trades:
            return 0.0
        up_trades = [t for t in trades if t.get("predicted_direction") == "up"]
        if up_trades:
            up_correct = sum(1 for t in up_trades if t.get("correct"))
            precision_up = up_correct / len(up_trades)
        else:
            precision_up = 0.0
        actual_up_trades = [t for t in trades if t.get("actual_direction") == "up"]
        if actual_up_trades:
            up_correct = sum(
                1
                for t in trades
                if t.get("predicted_direction") == "up" and t.get("correct")
            )
            recall_up = up_correct / len(actual_up_trades)
        else:
            recall_up = 0.0
        if precision_up + recall_up == 0:
            return 0.0
        return 2 * (precision_up * recall_up) / (precision_up + recall_up)

    def _should_disable_model(self, perf: ModelPerformance) -> bool:
        if perf.total_predictions < self.min_trades_for_eval:
            return False
        return perf.accuracy < self.accuracy_threshold

    def get_active_models(self) -> dict[str, ModelPerformance]:
        return {
            name: perf for name, perf in self.performances.items() if perf.is_active
        }

    def get_inactive_models(self) -> dict[str, ModelPerformance]:
        return {
            name: perf for name, perf in self.performances.items() if not perf.is_active
        }

    def get_best_models(self, top_n: int = 5) -> list[tuple[str, ModelPerformance]]:
        active_models = self.get_active_models()
        sorted_models = sorted(
            active_models.items(), key=lambda x: x[1].accuracy, reverse=True
        )
        return sorted_models[:top_n]

    def get_underperforming_models(self) -> list[tuple[str, ModelPerformance]]:
        active_models = self.get_active_models()
        threshold_buffer = 0.05
        warning_threshold = self.accuracy_threshold + threshold_buffer
        return [
            (name, perf)
            for name, perf in active_models.items()
            if perf.accuracy < warning_threshold
            and perf.total_predictions >= self.min_trades_for_eval
        ]

    def get_evaluation_report(self) -> dict[str, Any]:
        active = self.get_active_models()
        inactive = self.get_inactive_models()
        best = self.get_best_models(3)
        underperforming = self.get_underperforming_models()

        total_predictions = sum(p.total_predictions for p in self.performances.values())
        total_correct = sum(p.correct_predictions for p in self.performances.values())

        return {
            "timestamp": datetime.now().isoformat(),
            "summary": {
                "total_models": len(self.performances),
                "active_models": len(active),
                "inactive_models": len(inactive),
                "total_predictions": total_predictions,
                "overall_accuracy": total_correct / total_predictions
                if total_predictions > 0
                else 0.0,
            },
            "best_models": [
                {
                    "name": name,
                    "accuracy": perf.accuracy,
                    "trades": perf.total_predictions,
                }
                for name, perf in best
            ],
            "underperforming": [
                {
                    "name": name,
                    "accuracy": perf.accuracy,
                    "trades": perf.total_predictions,
                }
                for name, perf in underperforming
            ],
            "inactive_models": [
                {
                    "name": name,
                    "accuracy": perf.accuracy,
                    "reason": perf.disabled_reason,
                }
                for name, perf in inactive.items()
            ],
        }

    def reset_model(self, model_name: str):
        if model_name in self.performances:
            self.performances[model_name] = ModelPerformance(model_name=model_name)
            self.save()

    def reset_all(self):
        self.performances = {}
        self.save()
