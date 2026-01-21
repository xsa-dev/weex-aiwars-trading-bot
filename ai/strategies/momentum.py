from typing import Any

from ai.market_analyzer import Indicators, MarketData, MLSignals
from ai.strategies.base import BaseStrategy, StrategySignal


class MomentumStrategy(BaseStrategy):
    def __init__(self):
        super().__init__("momentum", weight=0.25)

    async def analyze(
        self,
        coin: str,
        market_data: MarketData,
        indicators: Indicators,
        ml_signals: MLSignals | None = None,
        current_position: dict[str, Any] | None = None,
    ) -> StrategySignal:
        tf_1h = indicators.timeframes.get("1h", {})
        tf_15m = indicators.timeframes.get("15m", {})

        rsi = tf_1h.get("rsi", 50)
        rsi_15m = tf_15m.get("rsi", 50)
        rsi_signal = tf_1h.get("rsi_signal", "")
        volume_ratio = tf_1h.get("volume_ratio", 1.0)
        vwap = tf_1h.get("vwap", 0)
        price = indicators.price

        bullish, bearish = 0, 0
        reasons = []

        if rsi < 30:
            bullish += 2
            reasons.append(f"RSI oversold 1h: {rsi:.1f}")
        elif rsi > 70:
            bearish += 2
            reasons.append(f"RSI overbought 1h: {rsi:.1f}")
        elif 40 <= rsi <= 60:
            if rsi_15m < rsi:
                bullish += 0.5
                reasons.append("RSI momentum building")
            elif rsi_15m > rsi:
                bearish += 0.5
                reasons.append("RSI momentum fading")

        if rsi_15m < 35 and rsi >= 45:
            bullish += 1
            reasons.append("Hidden bullish divergence")
        elif rsi_15m > 65 and rsi <= 55:
            bearish += 1
            reasons.append("Hidden bearish divergence")

        if volume_ratio > 1.5:
            if rsi > 55:
                bullish += 1
                reasons.append(f"High volume breakout: {volume_ratio:.1f}x")
            elif rsi < 45:
                bearish += 1
                reasons.append(f"High volume breakdown: {volume_ratio:.1f}x")

        if vwap > 0 and price > vwap:
            bullish += 0.5
            reasons.append("Price above VWAP")
        elif vwap > 0 and price < vwap:
            bearish += 0.5
            reasons.append("Price below VWAP")

        if rsi_signal == "bullish":
            bullish += 1
            reasons.append("RSI signal: bullish")
        elif rsi_signal == "bearish":
            bearish += 1
            reasons.append("RSI signal: bearish")

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
            reasoning=f"Momentum: {', '.join(reasons)}",
            metadata={
                "rsi_1h": rsi,
                "rsi_15m": rsi_15m,
                "volume_ratio": volume_ratio,
                "vwap": vwap,
            },
        )
