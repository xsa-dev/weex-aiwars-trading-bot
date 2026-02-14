"""PnL calculation utilities for trading.

Provides:
- Trade PnL calculation (LONG and SELL)
- Portfolio PnL calculation
- Fee calculations
"""

from dataclasses import dataclass


@dataclass
class PnLResult:
    """PnL calculation result."""

    gross_pnl: float
    fees: float
    net_pnl: float
    roi_percent: float


@dataclass
class PortfolioPnLResult:
    """Portfolio PnL result."""

    gross_pnl: float
    fees: float
    net_pnl: float
    roi_percent: float
    balance_before: float
    balance_after: float


def calculate_pnl(
    entry_price: float,
    exit_price: float,
    size: float,
    direction: str,
    fee_rate: float = 0.0005,
) -> PnLResult:
    """Calculate PnL for a trade.

    Args:
        entry_price: Entry price
        exit_price: Exit price
        size: Position size
        direction: LONG or SELL
        fee_rate: Trading fee rate (0.05% = 0.0005 for Weex)

    Returns:
        PnLResult with gross PnL, fees, net PnL, and ROI%
    """
    if direction == "LONG":
        gross_pnl = (exit_price - entry_price) * size
    else:
        gross_pnl = (entry_price - exit_price) * size

    fees = (entry_price + exit_price) * size * fee_rate
    net_pnl = gross_pnl - fees

    margin = entry_price * size
    roi = (net_pnl / margin) * 100 if margin > 0 else 0.0

    return PnLResult(
        gross_pnl=gross_pnl,
        fees=fees,
        net_pnl=net_pnl,
        roi_percent=roi,
    )


def calculate_portfolio_pnl(
    balance_before: float,
    balance_after: float,
    fees: float = 0.0,
) -> PortfolioPnLResult:
    """Calculate portfolio PnL from balance changes.

    Args:
        balance_before: Account balance before trades
        balance_after: Account balance after trades
        fees: Total fees paid

    Returns:
        PortfolioPnLResult with PnL metrics
    """
    gross_pnl = balance_after - balance_before
    net_pnl = gross_pnl - fees
    roi = (net_pnl / balance_before * 100) if balance_before > 0 else 0.0

    return PortfolioPnLResult(
        gross_pnl=gross_pnl,
        fees=fees,
        net_pnl=net_pnl,
        roi_percent=roi,
        balance_before=balance_before,
        balance_after=balance_after,
    )


def calculate_liquidation_price(
    entry_price: float,
    size: float,
    leverage: float = 1.0,
    maintenance_margin: float = 0.005,
) -> float:
    """Calculate estimated liquidation price.

    Args:
        entry_price: Entry price
        size: Position size
        leverage: Leverage multiplier
        maintenance_margin: Maintenance margin rate

    Returns:
        Estimated liquidation price
    """
    if leverage <= 1.0:
        return 0.0

    maintenance_margin_rate = maintenance_margin * leverage
    liquidation_distance = entry_price * (1 / leverage - maintenance_margin_rate)

    if entry_price > 0:
        return entry_price - liquidation_distance
    return 0.0


def calculate_margin_requirement(
    price: float,
    size: float,
    leverage: float = 1.0,
) -> float:
    """Calculate required margin for a position.

    Args:
        price: Asset price
        size: Position size
        leverage: Leverage multiplier

    Returns:
        Required margin in quote currency
    """
    notional = price * size
    return notional / leverage
