from typing import Any

from ai.market_analyzer import Indicators, MarketData, MLSignals
from ai.strategies.base import BaseStrategy, StrategySignal


class BreakoutStrategy(BaseStrategy):
    def __init__(self):
        super().__init__("breakout", weight=0.20)

    async def analyze(
        self,
        coin: str,
        market_data: MarketData,
        indicators: Indicators,
        ml_signals: MLSignals | None = None,
        current_position: dict[str, Any] | None = None,
    ) -> StrategySignal:
        tf_1h = indicators.timeframes.get("1h", {})
        tf_4h = indicators.timeframes.get("4h", {})

        atr = tf_1h.get("atr", 0)
        pivot = tf_1h.get("pivot", 0)
        support_1 = tf_1h.get("support_1", 0)
        resistance_1 = tf_1h.get("resistance_1", 0)
        bb_upper = tf_1h.get("bb_upper", 0)
        bb_lower = tf_1h.get("bb_lower", 0)
        bb_middle = tf_1h.get("bb_middle", 0)
        price = indicators.price

        bullish, bearish = 0, 0
        reasons = []

        if price > resistance_1:
            bullish += 2
            reasons.append(f"Breakout above resistance: {resistance_1:.2f}")
        elif price < support_1:
            bearish += 2
            reasons.append(f"Breakdown below support: {support_1:.2f}")
        elif price > pivot:
            bullish += 0.5
            reasons.append("Above daily pivot")
        else:
            bearish += 0.5
            reasons.append("Below daily pivot")

        tf_4h_trend = tf_4h.get("overall_trend", "NEUTRAL")
        if price > resistance_1 and tf_4h_trend == "BULLISH":
            bullish += 1
            reasons.append("4h trend confirms breakout")
        elif price < support_1 and tf_4h_trend == "BEARISH":
            bearish += 1
            reasons.append("4h trend confirms breakdown")

        if bb_upper > bb_middle > 0:
            bb_position = (
                (price - bb_lower) / (bb_upper - bb_lower)
                if bb_upper != bb_lower
                else 0.5
            )
            if bb_position > 0.9:
                bearish += 1
                reasons.append("Near BB upper band")
            elif bb_position < 0.1:
                bullish += 1
                reasons.append("Near BB lower band")

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
            reasoning=f"Breakout: {', '.join(reasons)}",
            metadata={"atr": atr, "pivot": pivot, "resistance_1": resistance_1},
        )
