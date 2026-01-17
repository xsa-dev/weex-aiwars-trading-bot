"""AI-based trading strategy for signal generation."""

from dataclasses import dataclass
from datetime import datetime
from typing import Any

from ai.config import (
    MIN_PROFIT_THRESHOLD,
    MIN_BULLISH_SIGNALS,
    SIGNAL_CONFIDENCE_DEFAULT,
    SIGNAL_CONFIDENCE_STRONG,
    SIGNAL_CONFIDENCE_WEAK,
    STOP_LOSS_PERCENT,
    TAKE_PROFIT_PERCENT,
)
from ai.market_analyzer import Indicators, MLSignals, MarketData


# =============================================================================
# Risk Levels
# =============================================================================


class RiskLevel:
    """Risk assessment levels."""

    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    EXTREME = "EXTREME"


# =============================================================================
# Decision Data Class
# =============================================================================


@dataclass
class Decision:
    """Trading decision for a single coin.

    Attributes:
        coin: Coin symbol (e.g., 'cmt_btcusdt')
        signal: Trading signal (LONG, SELL, HOLD)
        confidence: Signal confidence (0.0-1.0)
        reasoning: Explanation of the decision
        stop_loss: Stop loss price (optional)
        take_profit: Take profit price (optional)
        action_type: Type of action (OPEN, CLOSE, REVERSE, HOLD)
        correlation_id: Unique ID for AI logging
        position_size: Position size (default 0.001)
    """

    coin: str
    signal: str  # LONG, SELL, HOLD
    confidence: float
    reasoning: str
    stop_loss: float | None = None
    take_profit: float | None = None
    action_type: str = "OPEN"  # OPEN, CLOSE, REVERSE, HOLD
    correlation_id: str = ""
    position_size: float = 0.001


# =============================================================================
# Strategy Class
# =============================================================================


class Strategy:
    """AI-based trading strategy for generating trading signals.

    Combines:
    - Technical analysis (indicators)
    - ML market signals
    - Risk assessment

    Returns a list of decisions for all coins.
    """

    def __init__(self, ml_enabled: bool = True):
        self.ml_enabled = ml_enabled

    async def generate_signal(
        self,
        market_data: dict[str, MarketData],
        indicators: dict[str, Indicators],
        ml_signals: MLSignals,
        current_positions: list[dict[str, Any]] | None = None,
    ) -> list[Decision]:
        """Generate trading signals for all coins.

        Args:
            market_data: Dict mapping coin -> MarketData
            indicators: Dict mapping coin -> Indicators
            ml_signals: ML-based market signals
            current_positions: List of current positions (optional)

        Returns:
            List of Decision objects for each coin
        """
        decisions: list[Decision] = []
        positions_dict = self._build_positions_dict(current_positions or [])

        for coin, data in market_data.items():
            ind = indicators.get(coin)
            if not ind:
                continue

            decision = await self._generate_decision(
                coin=coin,
                market_data=data,
                indicators=ind,
                ml_signals=ml_signals,
                current_position=positions_dict.get(coin),
            )
            decisions.append(decision)

        return decisions

    async def _generate_decision(
        self,
        coin: str,
        market_data: MarketData,
        indicators: Indicators,
        ml_signals: MLSignals,
        current_position: dict[str, Any] | None,
    ) -> Decision:
        """Generate a decision for a single coin."""
        price = indicators.price

        # Step 1: Technical analysis - count bullish signals
        ta_signal, ta_confidence, ta_reasoning = self._analyze_indicators(indicators)

        # Step 2: Apply ML market direction filter
        if self.ml_enabled:
            final_signal, final_confidence, final_reasoning = self._combine_with_ml(
                ta_signal=ta_signal,
                ta_confidence=ta_confidence,
                ta_reasoning=ta_reasoning,
                ml_direction=ml_signals.direction,
                ml_confidence=ml_signals.confidence,
                ml_reasoning=ml_signals.reasoning,
            )
        else:
            final_signal, final_confidence, final_reasoning = (
                ta_signal,
                ta_confidence,
                ta_reasoning,
            )

        # Step 3: Check for take profit on existing position
        action_type = "OPEN"
        if current_position:
            action_type, final_signal, final_reasoning = self._check_position_status(
                current_position=current_position,
                ta_signal=ta_signal,
                ml_signals=ml_signals,
                price=price,
            )

        # Step 4: Calculate stop loss and take profit
        stop_loss = round(price * STOP_LOSS_PERCENT, 2) if price else None
        take_profit = round(price * TAKE_PROFIT_PERCENT, 2) if price else None

        # Step 5: Generate correlation ID for AI logging
        correlation_id = f"dec_{int(datetime.now().timestamp() * 1000)}_{coin}"

        return Decision(
            coin=coin,
            signal=final_signal,
            confidence=final_confidence,
            reasoning=final_reasoning,
            stop_loss=stop_loss,
            take_profit=take_profit,
            action_type=action_type,
            correlation_id=correlation_id,
        )

    def _analyze_indicators(self, indicators: Indicators) -> tuple[str, float, str]:
        """Analyze technical indicators for a coin.

        Returns:
            Tuple of (signal, confidence, reasoning)
        """
        signals: list[str] = []

        # Check each timeframe for bullish trend
        if indicators.timeframes.get("1m", {}).get("overall_trend") == "BULLISH":
            signals.append("bullish_1m")
        if indicators.timeframes.get("15m", {}).get("overall_trend") == "BULLISH":
            signals.append("bullish_15m")
        if indicators.timeframes.get("1h", {}).get("overall_trend") == "BULLISH":
            signals.append("bullish_1h")
        if indicators.timeframes.get("4h", {}).get("overall_trend") == "BULLISH":
            signals.append("bullish_4h")

        # Determine signal based on number of bullish signals
        if not signals:
            signal = "HOLD"
            confidence = SIGNAL_CONFIDENCE_DEFAULT
            reasoning = "No clear trend across timeframes"
        elif len(signals) >= MIN_BULLISH_SIGNALS:
            signal = "LONG"
            confidence = SIGNAL_CONFIDENCE_STRONG
            reasoning = f"Strong bullish signal ({len(signals)}/{MIN_BULLISH_SIGNALS}+ timeframes): {', '.join(signals)}"
        else:
            signal = "LONG"
            confidence = SIGNAL_CONFIDENCE_WEAK
            reasoning = (
                f"Weak bullish signal ({len(signals)} timeframes): {', '.join(signals)}"
            )

        return signal, confidence, reasoning

    def _combine_with_ml(
        self,
        ta_signal: str,
        ta_confidence: float,
        ta_reasoning: str,
        ml_direction: str,
        ml_confidence: float,
        ml_reasoning: str,
    ) -> tuple[str, float, str]:
        """Combine technical analysis with ML signals.

        Args:
            ta_signal: Technical analysis signal
            ta_confidence: Technical analysis confidence
            ta_reasoning: Technical analysis reasoning
            ml_direction: ML market direction
            ml_confidence: ML confidence
            ml_reasoning: ML reasoning

        Returns:
            Tuple of (final_signal, final_confidence, final_reasoning)
        """
        # If ML and TA agree, boost confidence
        if ml_direction == "BULLISH" and ta_signal == "LONG":
            # Boost confidence when both agree
            boosted_confidence = min(ta_confidence * (0.5 + ml_confidence * 0.5), 0.95)
            reasoning = (
                f"TA and ML agree: {ta_reasoning}\n\n"
                f"ML Market: {ml_direction} ({ml_confidence:.0%}) - {ml_reasoning}"
            )
            return "LONG", boosted_confidence, reasoning

        elif ml_direction == "BEARISH" and ta_signal == "LONG":
            # Reduce confidence when they disagree
            reduced_confidence = ta_confidence * (1 - ml_confidence * 0.3)
            reasoning = (
                f"TA bullish but ML bearish: {ta_reasoning}\n\n"
                f"ML Market: {ml_direction} ({ml_confidence:.0%}) - {ml_reasoning}"
            )
            return "HOLD", reduced_confidence, reasoning

        elif ml_direction == "BULLISH" and ta_signal == "HOLD":
            # ML says bullish, but TA is neutral - could be entry opportunity
            reasoning = (
                f"TA neutral but ML bullish: {ta_reasoning}\n\n"
                f"ML Market: {ml_direction} ({ml_confidence:.0%}) - {ml_reasoning}"
            )
            return "LONG", ml_confidence * 0.6, reasoning

        # Default to TA signal
        return ta_signal, ta_confidence, ta_reasoning

    def _check_position_status(
        self,
        current_position: dict[str, Any],
        ta_signal: str,
        ml_signals: MLSignals,
        price: float,
    ) -> tuple[str, str, str]:
        """Check if we should close/reverse an existing position.

        Args:
            current_position: Current position data
            ta_signal: Technical analysis signal
            ml_signals: ML signals
            price: Current price

        Returns:
            Tuple of (action_type, signal, reasoning)
        """
        position_side = current_position.get("side", "")
        unrealize_pnl = float(current_position.get("unrealizePnl", 0))

        # Check for take profit (position in profit)
        position_value = abs(float(current_position.get("open_value", 0)))
        if position_value > 0 and unrealize_pnl > 0:
            profit_percent = (unrealize_pnl / position_value) * 100
            if profit_percent >= MIN_PROFIT_THRESHOLD:
                return (
                    "CLOSE",
                    "CLOSE",
                    f"Take profit: +{profit_percent:.2f}% on {position_side} position",
                )

        # Check for reversal signals
        if ta_signal == "LONG" and position_side == "SHORT":
            return (
                "REVERSE",
                "LONG",
                f"Reversal: {position_side} -> LONG (ML: {ml_signals.direction})",
            )
        elif ta_signal == "SELL" and position_side == "LONG":
            return (
                "REVERSE",
                "SELL",
                f"Reversal: {position_side} -> SELL (ML: {ml_signals.direction})",
            )

        # Check for ML direction conflict
        if self.ml_enabled:
            if ml_signals.direction == "BEARISH" and position_side == "LONG":
                # Strong bearish signal from ML, consider closing
                if ml_signals.confidence > 0.7:
                    return (
                        "CLOSE",
                        "CLOSE",
                        f"ML bearish signal ({ml_signals.confidence:.0%}), closing LONG",
                    )
            elif ml_signals.direction == "BULLISH" and position_side == "SHORT":
                # Strong bullish signal from ML, consider closing
                if ml_signals.confidence > 0.7:
                    return (
                        "CLOSE",
                        "CLOSE",
                        f"ML bullish signal ({ml_signals.confidence:.0%}), closing SHORT",
                    )

        # No action needed - hold position
        return "HOLD", "HOLD", f"Holding {position_side} position"

    def _build_positions_dict(
        self, positions: list[dict[str, Any]]
    ) -> dict[str, dict[str, Any]]:
        """Build a dict of positions keyed by symbol."""
        return {pos.get("symbol"): pos for pos in positions if pos.get("symbol")}


# =============================================================================
# Risk Assessment
# =============================================================================


def assess_risk(
    market_data: dict[str, MarketData],
    decisions: list[Decision],
) -> str:
    """Assess overall market risk level.

    Args:
        market_data: Market data for all coins
        decisions: Generated decisions

    Returns:
        Risk level string
    """
    # Count signals
    long_count = sum(1 for d in decisions if d.signal == "LONG")
    sell_count = sum(1 for d in decisions if d.signal == "SELL")

    total = len(decisions)
    if total == 0:
        return RiskLevel.LOW

    long_ratio = long_count / total
    sell_ratio = sell_count / total

    # Determine risk level
    if (long_ratio > 0.7 or sell_ratio > 0.7) and abs(long_ratio - sell_ratio) > 0.5:
        # Strong consensus - could indicate momentum but also risk
        return RiskLevel.MEDIUM
    elif (long_ratio > 0.5 or sell_ratio > 0.5) and abs(long_ratio - sell_ratio) > 0.3:
        # Moderate consensus
        return RiskLevel.LOW
    else:
        # Mixed signals - high uncertainty
        return RiskLevel.MEDIUM
