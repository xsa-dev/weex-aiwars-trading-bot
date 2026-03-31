"""ML predictions client for localhost:8000.

This module provides async client for fetching predictions from ML server
running on localhost:8000. Supports all available ML models and calculates
ensemble predictions by averaging results from multiple models.
"""

import asyncio
from dataclasses import dataclass
from typing import Any

import aiohttp
from dotenv import load_dotenv

from ai.config import (
    ML_SERVER_HOST,
    ML_PREDICTION_HOURS,
    ML_REQUEST_TIMEOUT,
    ML_AVAILABLE_MODELS,
)
from utils.logger import add_log

load_dotenv()


@dataclass
class MLPrediction:
    """ML prediction result from localhost:8000."""

    coin: str
    model: str
    prediction_hours: int
    current_price: float
    predicted_price: float
    confidence: float
    direction: str
    change_percent: float
    timestamp: str
    raw_response: dict[str, Any] | None = None


@dataclass
class EnsembleMLPrediction:
    """Ensemble prediction combining multiple ML models."""

    coin: str
    direction: str
    confidence: float
    avg_change: float
    model_count: int
    models_used: list[str]
    individual_predictions: list[dict[str, Any]]


class MLClient:
    """Async client for ML server predictions."""

    def __init__(
        self,
        host: str = ML_SERVER_HOST,
        prediction_hours: int = ML_PREDICTION_HOURS,
        timeout: int = ML_REQUEST_TIMEOUT,
        max_retries: int = 2,
    ):
        self.host = host
        self.prediction_hours = prediction_hours
        self.timeout = aiohttp.ClientTimeout(total=timeout)
        self.max_retries = max_retries
        self.session: aiohttp.ClientSession | None = None
        self.models = ML_AVAILABLE_MODELS.copy()

    async def _get_session(self) -> aiohttp.ClientSession:
        if self.session is None or self.session.closed:
            self.session = aiohttp.ClientSession(timeout=self.timeout)
        return self.session

    def _coin_to_symbol(self, coin: str) -> str:
        """Convert coin format (BTCUSDT → BTC/USDT)."""
        symbol_map = {
            "BTCUSDT": "BTC/USDT",
            "ETHUSDT": "ETH/USDT",
            "SOLUSDT": "SOL/USDT",
            "DOGEUSDT": "DOGE/USDT",
            "XRPUSDT": "XRP/USDT",
            "ADAUSDT": "ADA/USDT",
            "BNBUSDT": "BNB/USDT",
            "LTCUSDT": "LTC/USDT",
        }
        coin_clean = coin.replace("cmt_", "").upper()
        return symbol_map.get(coin_clean, f"{coin_clean[:-4]}/USDT")

    async def get_prediction(
        self,
        coin: str,
        model: str = "GRU",
        prediction_hours: int | None = None,
    ) -> MLPrediction | None:
        """Get prediction for a single coin from ML server."""
        session = await self._get_session()
        hours = prediction_hours or self.prediction_hours
        symbol = self._coin_to_symbol(coin)
        url = f"{self.host}/api/predict/crypto-intraday"
        params = {
            "company": symbol,
            "model_type": model,
            "prediction_hours": hours,
        }

        for attempt in range(self.max_retries + 1):
            try:
                async with session.post(url, params=params) as response:
                    if response.status != 200:
                        return None

                    data = await response.json()
                    summary = data.get("prediction_summary", {})
                    next_hour = data.get("next_hour_prediction", {})
                    confidence_scores = data.get("confidence_scores", {})

                    current_price = next_hour.get("open", 0)
                    predicted_price = summary.get("end_price", current_price)

                    change_str = str(summary.get("24h_change", "0%"))
                    change_str = (
                        change_str.replace("↑", "").replace("↓", "").replace("%", "")
                    )
                    try:
                        change_percent = float(change_str)
                    except ValueError:
                        change_percent = 0.0

                    direction = (
                        "up"
                        if change_percent > 0
                        else "down"
                        if change_percent < 0
                        else "neutral"
                    )

                    return MLPrediction(
                        coin=coin,
                        model=model,
                        prediction_hours=hours,
                        current_price=current_price,
                        predicted_price=predicted_price,
                        confidence=confidence_scores.get("close", 0.5),
                        direction=direction,
                        change_percent=change_percent,
                        timestamp=data.get("prediction_date", ""),
                        raw_response=data,
                    )

            except Exception:
                if attempt < self.max_retries:
                    await asyncio.sleep(1 * (attempt + 1))

        return None

    async def get_all_predictions(
        self,
        coins: list[str],
        prediction_hours: int | None = None,
        models: list[str] | None = None,
    ) -> dict[str, list[MLPrediction]]:
        """Get predictions from ALL models for ALL coins in parallel."""
        hours = prediction_hours or self.prediction_hours
        model_list = models or self.models

        total_tasks = len(coins) * len(model_list)
        add_log(
            f"Fetching ML predictions for {len(coins)} coins × {len(model_list)} models ({total_tasks} total)",
            level="DEBUG",
        )

        tasks = []
        for coin in coins:
            for model in model_list:
                task = self.get_prediction(coin, model, hours)
                tasks.append(task)

        # Use gather for proper parallel execution, then report progress
        add_log("Waiting for ML predictions...", level="DEBUG")
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results and count successes
        predictions_by_coin: dict[str, list[MLPrediction]] = {
            coin: [] for coin in coins
        }
        success_count = 0
        none_count = 0
        error_count = 0

        for result in results:
            if isinstance(result, MLPrediction):
                predictions_by_coin[result.coin].append(result)
                success_count += 1
            elif result is None:
                none_count += 1
            elif isinstance(result, Exception):
                error_count += 1

        add_log(
            f"ML predictions complete: {success_count} success, {none_count} none, {error_count} errors",
            level="DEBUG",
        )

        return predictions_by_coin

    async def get_ensemble_prediction(
        self,
        coin: str,
        prediction_hours: int | None = None,
        models: list[str] | None = None,
    ) -> EnsembleMLPrediction:
        """Get ensemble prediction averaging all models."""
        predictions = await self.get_all_predictions([coin], prediction_hours, models)
        coin_predictions = predictions.get(coin, [])

        if not coin_predictions:
            return EnsembleMLPrediction(
                coin=coin,
                direction="neutral",
                confidence=0.0,
                avg_change=0.0,
                model_count=0,
                models_used=[],
                individual_predictions=[],
            )

        total_confidence = sum(p.confidence for p in coin_predictions)
        avg_confidence = total_confidence / len(coin_predictions)
        total_change = sum(p.change_percent for p in coin_predictions)
        avg_change = total_change / len(coin_predictions)

        up_votes = sum(1 for p in coin_predictions if p.direction == "up")
        down_votes = sum(1 for p in coin_predictions if p.direction == "down")
        direction = (
            "up"
            if up_votes > down_votes
            else "down"
            if down_votes > up_votes
            else "neutral"
        )

        return EnsembleMLPrediction(
            coin=coin,
            direction=direction,
            confidence=avg_confidence,
            avg_change=avg_change,
            model_count=len(coin_predictions),
            models_used=[p.model for p in coin_predictions],
            individual_predictions=[
                {
                    "model": p.model,
                    "direction": p.direction,
                    "confidence": p.confidence,
                    "change_percent": p.change_percent,
                }
                for p in coin_predictions
            ],
        )

    async def get_all_ensemble_predictions(
        self,
        coins: list[str],
        prediction_hours: int | None = None,
    ) -> dict[str, EnsembleMLPrediction]:
        """Get ensemble predictions for all coins."""
        add_log(f"Getting ensemble predictions for {len(coins)} coins")

        all_predictions = await self.get_all_predictions(coins, prediction_hours)
        ensemble_predictions: dict[str, EnsembleMLPrediction] = {}

        for coin, predictions in all_predictions.items():
            if predictions:
                total_conf = sum(p.confidence for p in predictions)
                avg_conf = total_conf / len(predictions)
                total_change = sum(p.change_percent for p in predictions)
                avg_change = total_change / len(predictions)
                up_votes = sum(1 for p in predictions if p.direction == "up")
                down_votes = sum(1 for p in predictions if p.direction == "down")
                direction = (
                    "up"
                    if up_votes > down_votes
                    else "down"
                    if down_votes > up_votes
                    else "neutral"
                )

                ensemble_predictions[coin] = EnsembleMLPrediction(
                    coin=coin,
                    direction=direction,
                    confidence=avg_conf,
                    avg_change=avg_change,
                    model_count=len(predictions),
                    models_used=[p.model for p in predictions],
                    individual_predictions=[
                        {
                            "model": p.model,
                            "direction": p.direction,
                            "confidence": p.confidence,
                            "change_percent": p.change_percent,
                        }
                        for p in predictions
                    ],
                )
            else:
                ensemble_predictions[coin] = EnsembleMLPrediction(
                    coin=coin,
                    direction="neutral",
                    confidence=0.0,
                    avg_change=0.0,
                    model_count=0,
                    models_used=[],
                    individual_predictions=[],
                )

        return ensemble_predictions

    async def close(self):
        """Close aiohttp session."""
        if self.session and not self.session.closed:
            await self.session.close()
            self.session = None


async def get_ml_predictions_for_coins(
    coins: list[str],
    prediction_hours: int = ML_PREDICTION_HOURS,
) -> dict[str, EnsembleMLPrediction]:
    """Get ensemble ML predictions for a list of coins."""
    client = MLClient(prediction_hours=prediction_hours)
    try:
        return await client.get_all_ensemble_predictions(coins, prediction_hours)
    finally:
        await client.close()
