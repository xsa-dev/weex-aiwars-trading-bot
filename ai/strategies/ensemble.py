from datetime import datetime
from typing import Any

from ai.config import ENSEMBLE_WEIGHTS
from ai.market_analyzer import Indicators, MarketData, MLSignals
from ai.position_manager import Decision
from ai.strategies.base import StrategySignal
from ai.strategies.breakout import BreakoutStrategy
from ai.strategies.momentum import MomentumStrategy
from ai.strategies.technical import TechnicalStrategy
from ai.strategies.trend import TrendStrategy
from utils.logger import add_log


class EnsembleCombiner:
    def __init__(self):
        self.strategies = [
            TechnicalStrategy(),
            TrendStrategy(),
            MomentumStrategy(),
            BreakoutStrategy(),
        ]

    def _get_weight(self, strategy_name: str) -> float:
        return ENSEMBLE_WEIGHTS.get(strategy_name, 1.0)

    async def generate_decisions(
        self,
        market_data: dict[str, MarketData],
        indicators: dict[str, Indicators],
        ml_signals: MLSignals,
        current_positions: list[dict[str, Any]] | None = None,
        strategies_config: dict[str, dict[str, Any]] | None = None,
    ) -> list[Decision]:
        decisions = []
        positions_dict = {
            p.get("symbol"): p for p in (current_positions or []) if p.get("symbol")
        }

        for coin, data in market_data.items():
            ind = indicators.get(coin)
            if not ind:
                continue

            current_position = positions_dict.get(coin)
            strategy_signals: list[StrategySignal] = []

            for strategy in self.strategies:
                if not strategy.enabled:
                    continue
                if strategies_config:
                    config = strategies_config.get(strategy.name, {})
                    if not config.get("enabled", True):
                        continue

                signal = await strategy.analyze(
                    coin=coin,
                    market_data=data,
                    indicators=ind,
                    ml_signals=ml_signals,
                    current_position=current_position,
                )
                strategy_signals.append(signal)

            if not strategy_signals:
                continue

            weighted_scores = {"LONG": 0.0, "SELL": 0.0, "HOLD": 0.0}
            total_weight = 0.0

            for sig in strategy_signals:
                weight = self._get_weight(sig.strategy_name) * sig.confidence
                weighted_scores[sig.signal] += weight
                total_weight += weight

            if total_weight > 0:
                for signal in weighted_scores:
                    weighted_scores[signal] /= total_weight

            final_signal = max(weighted_scores, key=weighted_scores.get)
            final_confidence = weighted_scores[final_signal]

            if final_confidence < 0.5:
                final_signal = "HOLD"

            if current_position:
                position_side = current_position.get("side", "")
                if final_signal == "HOLD":
                    action_type = "HOLD"
                elif (final_signal == "LONG" and position_side == "SHORT") or (
                    final_signal == "SELL" and position_side == "LONG"
                ):
                    action_type = "REVERSE"
                elif final_signal != position_side:
                    action_type = "CLOSE"
                else:
                    action_type = "HOLD"
            else:
                action_type = "OPEN" if final_signal != "HOLD" else "HOLD"

            signal_summary = ", ".join(
                f"{s.signal}({s.confidence:.0%})" for s in strategy_signals
            )
            reasoning = f"Ensemble: {signal_summary}. ML: {ml_signals.direction}({ml_signals.confidence:.0%})"

            price = ind.price or 0
            stop_loss = round(price * 0.98, 2) if price else None
            take_profit = round(price * 1.03, 2) if price else None

            decision = Decision(
                coin=coin,
                signal=final_signal,
                confidence=final_confidence,
                reasoning=reasoning,
                stop_loss=stop_loss,
                take_profit=take_profit,
                action_type=action_type,
                correlation_id=f"dec_{int(datetime.now().timestamp() * 1000)}_{coin}",
            )
            decisions.append(decision)

            # Calculate direction distribution for this coin
            direction_counts = {}
            for s in strategy_signals:
                direction_counts[s.signal] = direction_counts.get(s.signal, 0) + 1
            dist_parts = []
            for d, c in sorted(direction_counts.items(), key=lambda x: -x[1]):
                dist_parts.append(f"{d[0]}:{c}")

            # Per-strategy breakdown
            strat_parts = []
            for s in strategy_signals:
                strat_name = s.strategy_name[:3].upper()
                strat_parts.append(f"{strat_name}:{s.signal[0]}({s.confidence:.0%})")

            coin_short = coin.replace("cmt_", "").upper()[:7]
            add_log(
                f"Ensemble {coin_short}: {final_signal} ({final_confidence:.0%}) | "
                f"{', '.join(dist_parts)} | "
                f"{' '.join(strat_parts)}",
                logger_name="ensemble",
            )

        return decisions
