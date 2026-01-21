from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from ai.market_analyzer import Indicators, MarketData, MLSignals


@dataclass
class StrategySignal:
    strategy_name: str
    coin: str
    signal: str
    confidence: float
    reasoning: str
    metadata: dict[str, Any] | None = None
    timestamp: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()


class BaseStrategy(ABC):
    def __init__(self, name: str, weight: float = 1.0):
        self.name = name
        self.weight = weight

    @abstractmethod
    async def analyze(
        self,
        coin: str,
        market_data: MarketData,
        indicators: Indicators,
        ml_signals: MLSignals | None = None,
        current_position: dict[str, Any] | None = None,
    ) -> StrategySignal:
        pass

    @property
    def enabled(self) -> bool:
        from ai.config import STRATEGIES

        return STRATEGIES.get(self.name, {}).get("enabled", True)
