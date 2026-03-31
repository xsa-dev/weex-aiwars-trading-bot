from typing import Any

from ai.market_analyzer import Indicators, MarketData, MLSignals
from ai.strategies.base import BaseStrategy, StrategySignal


class TechnicalStrategy(BaseStrategy):
    def __init__(self):
        super().__init__("technical", weight=0.30)

    async def analyze(
        self,
        coin: str,
        market_data: MarketData,
        indicators: Indicators,
        ml_signals: MLSignals | None = None,
        current_position: dict[str, Any] | None = None,
    ) -> StrategySignal:
        tf_1h = indicators.timeframes.get("1h", {})

        trend = tf_1h.get("overall_trend", "NEUTRAL")
        rsi = tf_1h.get("rsi", 50)
        adx = tf_1h.get("adx", 0)
        adx_strength = tf_1h.get("adx_strength", "")
        macd_hist = tf_1h.get("macd_histogram", 0)
        st_dir = tf_1h.get("supertrend_direction", "N")

        bullish, bearish = 0, 0
        reasons = []

        if trend == "BULLISH":
            bullish += 2
            reasons.append("1h trend: BULLISH")
        elif trend == "BEARISH":
            bearish += 2
            reasons.append("1h trend: BEARISH")

        if rsi > 70:
            bearish += 1
            reasons.append(f"RSI overbought: {rsi:.1f}")
        elif rsi < 30:
            bullish += 1
            reasons.append(f"RSI oversold: {rsi:.1f}")

        if adx >= 25:
            if adx_strength == "STRONG":
                if trend == "BULLISH":
                    bullish += 1
                    reasons.append(f"ADX strong bullish: {adx:.1f}")
                elif trend == "BEARISH":
                    bearish += 1
                    reasons.append(f"ADX strong bearish: {adx:.1f}")

        if macd_hist > 0:
            bullish += 1
            reasons.append(f"MACD bullish: {macd_hist:.4f}")
        elif macd_hist < 0:
            bearish += 1
            reasons.append(f"MACD bearish: {macd_hist:.4f}")

        if st_dir == "UP":
            bullish += 1
            reasons.append("SuperTrend UP")
        elif st_dir == "DOWN":
            bearish += 1
            reasons.append("SuperTrend DOWN")

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
            reasoning=f"Technical: {', '.join(reasons)}",
            metadata={
                "rsi": rsi,
                "adx": adx,
                "macd_hist": macd_hist,
                "supertrend": st_dir,
            },
        )
