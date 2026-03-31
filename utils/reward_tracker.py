"""Reward tracking system for trading decisions.

Features:
- Track risk-reward ratio for each trade
- Record PnL when trades close
- Generate RL training data samples
- Store all data in JSON files
"""

import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


TRADES_FILE = Path("trading_rewards.json")
PNL_FILE = Path("pnl_history.json")
TRAINING_FILE = Path("training_data.json")


@dataclass
class TradeReward:
    """Risk-reward tracking for a single trade."""

    trade_id: str
    coin: str
    entry_time: str
    exit_time: str | None
    entry_price: float
    exit_price: float | None
    direction: str
    risk_reward_ratio: float
    stop_loss: float
    take_profit: float
    pnl_percent: float | None
    status: str
    ml_prediction_direction: str | None
    ml_confidence: float | None


@dataclass
class PnLRecord:
    """PnL record for a closed trade."""

    timestamp: str
    trade_id: str
    coin: str
    direction: str
    entry_price: float
    exit_price: float
    pnl_percent: float
    status: str
    duration_hours: float


@dataclass
class RLTrainingSample:
    """RL training sample from a completed trade."""

    state: dict[str, Any]
    action: str
    reward: float
    reward_breakdown: dict[str, float]
    outcome: dict[str, Any]
    timestamp: str


class RewardTracker:
    """Track rewards, PnL, and generate RL training data."""

    def __init__(self):
        self.trades: list[TradeReward] = self._load_trades()
        self.pnl_history: list[dict] = self._load_pnl()
        self.training_data: list[dict] = self._load_training()

    # === TRADE MANAGEMENT ===

    def save_trade_opened(
        self,
        trade_id: str,
        coin: str,
        entry_price: float,
        direction: str,
        stop_loss: float,
        take_profit: float,
        ml_direction: str | None = None,
        ml_confidence: float | None = None,
    ) -> TradeReward:
        """Record when a trade is opened."""
        risk = abs(entry_price - stop_loss)
        reward = abs(take_profit - entry_price)
        rr = reward / risk if risk > 0 else 0.0

        trade = TradeReward(
            trade_id=trade_id,
            coin=coin,
            entry_time=datetime.utcnow().isoformat(),
            exit_time=None,
            entry_price=entry_price,
            exit_price=None,
            direction=direction,
            risk_reward_ratio=rr,
            stop_loss=stop_loss,
            take_profit=take_profit,
            pnl_percent=None,
            status="OPEN",
            ml_prediction_direction=ml_direction,
            ml_confidence=ml_confidence,
        )

        self.trades.append(trade)
        self._save_trades()
        return trade

    def save_trade_closed(
        self,
        trade_id: str,
        exit_price: float,
        close_reason: str,
        actual_pnl_percent: float,
    ) -> TradeReward | None:
        """Record when a trade is closed."""
        for trade in self.trades:
            if trade.trade_id == trade_id and trade.status == "OPEN":
                trade.exit_time = datetime.utcnow().isoformat()
                trade.exit_price = exit_price
                trade.pnl_percent = actual_pnl_percent
                trade.status = close_reason

                self._add_pnl_record(trade)
                self._generate_training_sample(trade)

                self._save_trades()
                self._save_pnl()
                self._save_training()
                return trade
        return None

    def get_open_trade(self, trade_id: str) -> TradeReward | None:
        """Get an open trade by ID."""
        for trade in self.trades:
            if trade.trade_id == trade_id and trade.status == "OPEN":
                return trade
        return None

    def get_trade_by_coin(self, coin: str) -> list[TradeReward]:
        """Get all trades for a coin."""
        return [t for t in self.trades if t.coin == coin]

    # === PnL TRACKING ===

    def _add_pnl_record(self, trade: TradeReward):
        """Add record to PnL history."""
        record = {
            "timestamp": datetime.utcnow().isoformat(),
            "trade_id": trade.trade_id,
            "coin": trade.coin,
            "direction": trade.direction,
            "entry_price": trade.entry_price,
            "exit_price": trade.exit_price,
            "pnl_percent": trade.pnl_percent,
            "status": trade.status,
            "duration_hours": self._calculate_duration_hours(trade),
        }
        self.pnl_history.append(record)

    def _calculate_duration_hours(self, trade: TradeReward) -> float:
        """Calculate trade duration in hours."""
        if not trade.exit_time:
            return 0.0
        entry = datetime.fromisoformat(trade.entry_time)
        exit = datetime.fromisoformat(trade.exit_time)
        return (exit - entry).total_seconds() / 3600

    # === RL TRAINING DATA ===

    def _generate_training_sample(self, trade: TradeReward):
        """Generate RL training sample from closed trade."""
        if trade.status == "OPEN" or trade.exit_price is None:
            return

        pnl_reward = trade.pnl_percent / 100 if trade.pnl_percent else 0.0

        direction_correct = (
            trade.direction == "LONG" and trade.pnl_percent and trade.pnl_percent > 0
        ) or (trade.direction == "SELL" and trade.pnl_percent and trade.pnl_percent < 0)
        direction_reward = 1.0 if direction_correct else -0.5

        risk_reward = (
            1.0 if trade.status == "TP" else -1.0 if trade.status == "SL" else 0.0
        )

        total_reward = 0.4 * pnl_reward + 0.3 * direction_reward + 0.3 * risk_reward

        sample = {
            "state": {
                "coin": trade.coin,
                "direction": trade.direction,
                "entry_price": trade.entry_price,
                "ml_direction": trade.ml_prediction_direction,
                "ml_confidence": trade.ml_confidence,
                "risk_reward_ratio": trade.risk_reward_ratio,
            },
            "action": trade.direction,
            "reward": round(total_reward, 4),
            "reward_breakdown": {
                "pnl_reward": round(pnl_reward, 4),
                "direction_reward": direction_reward,
                "risk_reward": risk_reward,
            },
            "outcome": {
                "exit_price": trade.exit_price,
                "pnl_percent": trade.pnl_percent,
                "close_reason": trade.status,
                "duration_hours": self._calculate_duration_hours(trade),
            },
            "timestamp": trade.exit_time,
        }

        self.training_data.append(sample)

    # === FILE OPERATIONS ===

    def _load_trades(self) -> list[TradeReward]:
        if TRADES_FILE.exists():
            with open(TRADES_FILE) as f:
                data = json.load(f)
                return [TradeReward(**t) for t in data]
        return []

    def _save_trades(self):
        with open(TRADES_FILE, "w") as f:
            json.dump([asdict(t) for t in self.trades], f, indent=2)

    def _load_pnl(self) -> list[dict]:
        if PNL_FILE.exists():
            with open(PNL_FILE) as f:
                return json.load(f)
        return []

    def _save_pnl(self):
        with open(PNL_FILE, "w") as f:
            json.dump(self.pnl_history, f, indent=2)

    def _load_training(self) -> list[dict]:
        if TRAINING_FILE.exists():
            with open(TRAINING_FILE) as f:
                return json.load(f)
        return []

    def _save_training(self):
        with open(TRAINING_FILE, "w") as f:
            json.dump(self.training_data, f, indent=2)

    # === STATISTICS ===

    def get_statistics(self) -> dict[str, Any]:
        """Get trading statistics."""
        closed = [t for t in self.trades if t.status != "OPEN"]
        if not closed:
            return {"total_trades": 0, "win_rate": 0.0, "avg_pnl": 0.0}

        wins = sum(1 for t in closed if t.pnl_percent and t.pnl_percent > 0)
        avg_pnl = sum(t.pnl_percent for t in closed if t.pnl_percent) / len(closed)

        return {
            "total_trades": len(closed),
            "win_rate": wins / len(closed),
            "avg_pnl_percent": avg_pnl,
            "tp_count": sum(1 for t in closed if t.status == "TP"),
            "sl_count": sum(1 for t in closed if t.status == "SL"),
            "open_count": sum(1 for t in self.trades if t.status == "OPEN"),
            "total_pnl": sum(t.pnl_percent for t in closed if t.pnl_percent),
        }

    def get_recent_training_samples(self, count: int = 100) -> list[dict]:
        """Get most recent training samples."""
        return self.training_data[-count:]

    def clear_all(self):
        """Clear all data (for testing)."""
        self.trades = []
        self.pnl_history = []
        self.training_data = []
        self._save_trades()
        self._save_pnl()
        self._save_training()
