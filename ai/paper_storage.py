"""Paper trading storage - saves trades to disk without executing."""

import json
from pathlib import Path
from datetime import datetime
from typing import Any

PAPER_TRADES_FILE = Path("paper_trades.json")


def _load_trades() -> list[dict[str, Any]]:
    """Load paper trades from disk."""
    if PAPER_TRADES_FILE.exists():
        try:
            with open(PAPER_TRADES_FILE) as f:
                return json.load(f)
        except Exception:
            return []
    return []


def _save_trades(trades: list[dict[str, Any]]) -> None:
    """Save paper trades to disk."""
    with open(PAPER_TRADES_FILE, "w") as f:
        json.dump(trades, f, indent=2)


def save_paper_trade(
    coin: str,
    action: str,
    signal: str,
    price: float,
    size: float,
    confidence: float,
    reasoning: str,
) -> dict[str, Any]:
    """Save a paper trade and return the saved record."""
    trade = {
        "timestamp": datetime.now().isoformat(),
        "coin": coin,
        "action": action,
        "signal": signal,
        "price": price,
        "size": size,
        "confidence": confidence,
        "reasoning": reasoning,
        "status": "PENDING",
        "executed_at": None,
    }

    trades = _load_trades()
    trades.append(trade)
    _save_trades(trades)

    return trade


def get_paper_trades(limit: int = 50) -> list[dict[str, Any]]:
    """Get all paper trades."""
    trades = _load_trades()
    return sorted(trades, key=lambda x: x["timestamp"], reverse=True)[:limit]


def clear_paper_trades() -> None:
    """Clear all paper trades."""
    _save_trades([])


def get_paper_trades_count() -> int:
    """Get count of paper trades."""
    return len(_load_trades())
