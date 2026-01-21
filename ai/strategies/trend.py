from typing import Any

from ai.market_analyzer import Indicators, MarketData, MLSignals
from ai.strategies.base import BaseStrategy, StrategySignal


class TrendStrategy(BaseStrategy):
    def __init__(self):
        super().__init__("trend", weight=0.25)

    async def analyze(
        self,
        coin: str,
        market_data: MarketData,
        indicators: Indicators,
        ml_signals: MLSignals | None = None,
        current_position: dict[str, Any] | None = None,
    ) -> StrategySignal:
        tf_1h = indicators.timeframes.get("1h", {})

        ema_9 = tf_1h.get("ema_9", 0)
        ema_21 = tf_1h.get("ema_21", 0)
        ema_50 = tf_1h.get("ema_50", 0)
        ema_200 = tf_1h.get("ema_200", 0)
        price = indicators.price

        bullish, bearish = 0, 0
        reasons = []

        if ema_9 > ema_21:
            bullish += 1
            reasons.append("EMA 9 > 21")
        else:
            bearish += 1
            reasons.append("EMA 9 < 21")

        if ema_21 > ema_50:
            bullish += 1
            reasons.append("EMA 21 > 50")
        else:
            bearish += 1
            reasons.append("EMA 21 < 50")

        if price > ema_9:
            bullish += 0.5
        else:
            bearish += 0.5

        if ema_50 > ema_200:
            bullish += 1
            reasons.append("Golden cross")
        elif ema_50 < ema_200:
            bearish += 1
            reasons.append("Death cross")

        if bullish > bearish:
            signal = "LONG"
            max_signals = bullish
        elif bearish > bullish:
            signal = "SELL"
            max_signals = bearish
        else:
            signal = "HOLD"
            max_signals = 0

        total = bullish + bearish
        confidence = 0.5 + (max_signals / total) * 0.4 if total > 0 else 0.5

        return StrategySignal(
            strategy_name=self.name,
            coin=coin,
            signal=signal,
            confidence=confidence,
            reasoning=f"Trend: {', '.join(reasons)}",
            metadata={
                "ema_9": ema_9,
                "ema_21": ema_21,
                "ema_50": ema_50,
                "ema_200": ema_200,
            },
        )
