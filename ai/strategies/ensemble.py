"""Ensemble combiner using ALL signal sources.

Combines signals from:
- Technical indicators (RSI, MACD, ADX, SuperTrend)
- ML predictions (XGBoost, LSTM, GRU, etc.)
- News sentiment (crypto news)
- LLM analysis (DeepSeek)

Uses EnsembleVoter for weighted voting based on SOURCE_WEIGHTS.
"""

import asyncio
from datetime import datetime
from typing import Any

from ai.config import (
    SOURCE_WEIGHTS,
    DEFAULT_ORDER_SIZE,
    STOP_LOSS_PERCENT,
    TAKE_PROFIT_PERCENT,
)
from ai.ensemble_voter import (
    EnsembleVoter,
    EnsembleSignal,
    TechnicalSignal,
    MLPredictionSignal,
    NewsSentimentSignal,
)
from ai.market_analyzer import Indicators, MarketData, MLSignals
from ai.position_manager import Decision
from utils.logger import add_log


class EnsembleCombiner:
    """Combine ALL signals: Technical + ML + News + LLM via EnsembleVoter."""

    def __init__(self):
        # EnsembleVoter handles weighted voting with SOURCE_WEIGHTS
        self.ensemble_voter = EnsembleVoter(weights=SOURCE_WEIGHTS.copy())

        # Lazy-loaded clients for parallel fetching
        self._ml_client = None
        self._news_client = None

    @property
    def ml_client(self):
        """Lazy load ML client."""
        if self._ml_client is None:
            try:
                from ai.ml_client import MLClient

                self._ml_client = MLClient()
            except Exception as e:
                add_log(f"Failed to init ML client: {e}", level="WARNING")
                self._ml_client = None
        return self._ml_client

    @property
    def news_client(self):
        """Lazy load News client."""
        if self._news_client is None:
            try:
                from ai.news_client import NewsClient

                self._news_client = NewsClient()
            except Exception as e:
                add_log(f"Failed to init News client: {e}", level="WARNING")
                self._news_client = None
        return self._news_client

    async def generate_decisions(
        self,
        market_data: dict[str, MarketData],
        indicators: dict[str, Indicators],
        ml_signals: MLSignals,
        current_positions: list[dict[str, Any]] | None = None,
        portfolio_drawdown: float = 0.0,
    ) -> list[Decision]:
        """Generate decisions using ALL signal sources via EnsembleVoter."""
        decisions = []
        positions_dict = {
            p.get("symbol"): p for p in (current_positions or []) if p.get("symbol")
        }

        coins = list(market_data.keys())

        # ================================================================
        # STEP 1: Fetch ML predictions and News for all coins in parallel
        # ================================================================
        ml_predictions: dict[str, MLPredictionSignal] = {}
        news_sentiments: dict[str, NewsSentimentSignal] = {}

        async def fetch_ml_predictions() -> dict:
            if self.ml_client:
                try:
                    return await self.ml_client.get_all_ensemble_predictions(coins)
                except Exception as e:
                    add_log(f"ML predictions failed: {e}", level="WARNING")
            return {}

        async def fetch_news_sentiments() -> dict:
            if self.news_client:
                try:
                    return await self.news_client.get_all_sentiments(coins)
                except Exception as e:
                    add_log(f"News sentiment failed: {e}", level="WARNING")
            return {}

        # Fetch in parallel
        ml_raw, news_raw = await asyncio.gather(
            fetch_ml_predictions(),
            fetch_news_sentiments(),
        )

        # Convert to EnsembleVoter signal format
        for coin, pred in ml_raw.items():
            if pred:
                ml_predictions[coin] = MLPredictionSignal(
                    direction=pred.direction,  # up/down/neutral
                    confidence=pred.confidence,
                    avg_change=pred.avg_change,
                    model_count=pred.model_count,
                    models_used=pred.models_used,
                )

        for coin, sent in news_raw.items():
            if sent:
                news_sentiments[coin] = NewsSentimentSignal(
                    sentiment=sent.sentiment,  # bullish/bearish/neutral
                    confidence=sent.confidence,
                    total_articles=sent.total_articles,
                    reasoning=f"News: {sent.sentiment} ({sent.total_articles} articles)",
                )

        add_log(
            f"Ensemble sources: ML={len(ml_predictions)}, News={len(news_sentiments)}",
            level="DEBUG",
        )

        # ================================================================
        # STEP 2: Generate decision for each coin using EnsembleVoter
        # ================================================================
        for coin, data in market_data.items():
            ind = indicators.get(coin)
            if not ind:
                continue

            current_position = positions_dict.get(coin)

            # Build Technical signal from indicators
            technical_signal = self._build_technical_signal(coin, ind)

            # Get ML signal (convert direction format)
            ml_pred = ml_predictions.get(coin)
            ml_signal = None
            if ml_pred:
                ml_direction = (
                    "up"
                    if ml_pred.direction in ["up", "long", "buy", "bullish"]
                    else "down"
                    if ml_pred.direction in ["down", "short", "sell", "bearish"]
                    else "neutral"
                )
                ml_signal = MLPredictionSignal(
                    direction=ml_direction,
                    confidence=ml_pred.confidence,
                    avg_change=ml_pred.avg_change,
                    model_count=ml_pred.model_count,
                    models_used=ml_pred.models_used,
                )

            # Get News signal
            news_sig = news_sentiments.get(coin)

            # Get LLM signal from ml_signals (if available)
            llm_signal = None
            if (
                ml_signals
                and hasattr(ml_signals, "source_breakdown")
                and ml_signals.source_breakdown
            ):
                # LLM might be in source_breakdown
                pass  # Will add later if needed

            # ============================================================
            # KEY: Use EnsembleVoter with ALL sources!
            # EnsembleVoter.vote() already returns final_signal and confidence
            # ============================================================
            ensemble = self.ensemble_voter.vote(
                coin=coin,
                price=ind.price,
                technical=technical_signal,
                llm=llm_signal,
                ml=ml_signal,
                news=news_sig,
                current_portfolio_drawdown=portfolio_drawdown,
                current_position=current_position,
            )

            # Convert EnsembleVoter signal (buy/sell/hold) to Decision format
            final_signal = ensemble.signal.upper()  # BUY/SELL/HOLD
            final_confidence = ensemble.confidence

            # Determine action type based on current position
            if current_position:
                position_side = current_position.get("side", "")
                if final_signal == "HOLD":
                    action_type = "HOLD"
                elif (final_signal == "BUY" and position_side == "SHORT") or (
                    final_signal == "SELL" and position_side == "LONG"
                ):
                    action_type = "REVERSE"
                elif final_signal != position_side:
                    action_type = "CLOSE"
                else:
                    action_type = "HOLD"
            else:
                action_type = "OPEN" if final_signal != "HOLD" else "HOLD"

            # Calculate stop loss and take profit
            price = ind.price or 0
            if price and final_signal != "HOLD":
                stop_loss = round(price * STOP_LOSS_PERCENT, 2)
                take_profit = round(price * TAKE_PROFIT_PERCENT, 2)
            else:
                stop_loss = None
                take_profit = None

            # Build reasoning string
            reasoning = self._build_reasoning(
                coin=coin,
                ensemble=ensemble,
                technical=technical_signal,
                ml=ml_signal,
                news=news_sig,
            )

            decision = Decision(
                coin=coin,
                signal=final_signal,
                confidence=final_confidence,
                reasoning=reasoning,
                stop_loss=stop_loss,
                take_profit=take_profit,
                action_type=action_type,
                correlation_id=f"dec_{int(datetime.now().timestamp() * 1000)}_{coin}",
                position_size=DEFAULT_ORDER_SIZE
                * self._get_position_multiplier(final_confidence),
            )
            decisions.append(decision)

            # Log ensemble result
            self._log_ensemble_result(coin, ensemble, ml_signal, news_sig)

        return decisions

    def _build_technical_signal(self, coin: str, ind: Indicators) -> TechnicalSignal:
        """Build TechnicalSignal from calculated indicators."""
        tf_data = ind.timeframes.get("1h", {})

        trend = tf_data.get("overall_trend", "NEUTRAL")
        rsi = tf_data.get("rsi", 50)
        adx = tf_data.get("adx", 0)
        macd_hist = tf_data.get("macd_histogram", 0)
        st_dir = tf_data.get("supertrend_direction", "N")

        # Count bullish/bearish signals
        bullish = 0
        bearish = 0

        if trend == "BULLISH":
            bullish += 2
        elif trend == "BEARISH":
            bearish += 2

        if rsi < 30:  # Oversold - bullish
            bullish += 1
        elif rsi > 70:  # Overbought - bearish
            bearish += 1

        if adx >= 25:
            if trend == "BULLISH":
                bullish += 1
            elif trend == "BEARISH":
                bearish += 1

        if macd_hist > 0:
            bullish += 1
        else:
            bearish += 1

        if st_dir == "UP":
            bullish += 1
        elif st_dir == "DOWN":
            bearish += 1

        if bullish > bearish:
            direction = "bullish"
        elif bearish > bullish:
            direction = "bearish"
        else:
            direction = "neutral"

        total = bullish + bearish
        confidence = 0.5 + (max(bullish, bearish) / total) * 0.4 if total > 0 else 0.5

        reasoning_parts = []
        if direction == "bullish":
            reasoning_parts.append("Technical indicators suggest upward momentum")
        elif direction == "bearish":
            reasoning_parts.append("Technical indicators suggest downward pressure")
        else:
            reasoning_parts.append("Mixed technical signals")

        return TechnicalSignal(
            signal=direction,
            confidence=confidence,
            reasoning=". ".join(reasoning_parts),
            indicators_used={
                "trend": trend,
                "rsi": round(rsi, 1),
                "adx": round(adx, 1),
                "macd": round(macd_hist, 4),
                "supertrend": st_dir,
            },
        )

    def _build_reasoning(
        self,
        coin: str,
        ensemble: EnsembleSignal,
        technical: TechnicalSignal | None,
        ml: MLPredictionSignal | None,
        news: NewsSentimentSignal | None,
    ) -> str:
        """Build human-readable reasoning string."""
        parts = []

        # Source breakdown
        sources = []

        if technical:
            sources.append(
                f"TECH({technical.signal[0].upper()},{technical.confidence:.0%})"
            )

        if ml:
            sources.append(
                f"ML({ml.direction[0].upper()},{ml.confidence:.0%},{ml.model_count}m)"
            )

        if news:
            sources.append(
                f"NEWS({news.sentiment[0].upper()},{news.confidence:.0%},{news.total_articles}a)"
            )

        parts.append(f"Ensemble: {', '.join(sources)}")
        parts.append(f"Final: {ensemble.signal.upper()}({ensemble.confidence:.0%})")
        parts.append(f"Risk: {ensemble.risk_level}")

        return " | ".join(parts)

    def _log_ensemble_result(
        self,
        coin: str,
        ensemble: EnsembleSignal,
        ml: MLPredictionSignal | None,
        news: NewsSentimentSignal | None,
    ):
        """Log ensemble voting result."""
        coin_short = coin.replace("cmt_", "").upper()[:7]

        # Direction counts from ensemble
        breakdown = ensemble.source_breakdown or {}
        buy_sources = []
        sell_sources = []

        for source, data in breakdown.items():
            direction = data.get("direction", "")
            if direction == "buy":
                buy_sources.append(f"{source}({data.get('confidence', 0):.0%})")
            elif direction == "sell":
                sell_sources.append(f"{source}({data.get('confidence', 0):.0%})")

        buy_str = f"BUY({', '.join(buy_sources)})" if buy_sources else "BUY(0)"
        sell_str = f"SELL({', '.join(sell_sources)})" if sell_sources else "SELL(0)"

        add_log(
            f"Ensemble {coin_short}: {ensemble.signal.upper()}({ensemble.confidence:.0%}) | "
            f"{buy_str} vs {sell_str} | "
            f"Risk: {ensemble.risk_level}",
            logger_name="ensemble",
        )

    def _get_position_multiplier(self, confidence: float) -> float:
        """Get position size multiplier based on confidence."""
        if confidence >= 0.80:
            return 1.5
        elif confidence >= 0.70:
            return 1.2
        elif confidence >= 0.65:
            return 0.8
        else:
            return 0.5

    async def close(self):
        """Close async clients."""
        if self._ml_client:
            try:
                await self._ml_client.close()
            except Exception:
                pass
            self._ml_client = None

        if self._news_client:
            try:
                await self._news_client.close()
            except Exception:
                pass
            self._news_client = None
