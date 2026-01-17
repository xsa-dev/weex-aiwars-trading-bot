"""Position management and order execution."""

from dataclasses import dataclass
from typing import Any

from weex_client import WeexAsyncClient, config
from weex_client.models import PlaceOrderRequest

from ai.ai_log_stub import AILogStub
from ai.config import PAPER_TRADING
from ai.market_analyzer import MarketData, RateLimiter
from ai.paper_storage import save_paper_trade
from ai.strategy import Decision
from utils.logger import add_log, logger

settings = config.load_config()


# =============================================================================
# Result Classes
# =============================================================================


@dataclass
class ActionResult:
    """Result of executing an action."""

    coin: str
    success: bool
    action_type: str
    order_id: str | None = None
    error: str | None = None
    details: dict[str, Any] | None = None


# =============================================================================
# Position Manager
# =============================================================================


class PositionManager:
    """Manages positions and executes trading decisions."""

    def __init__(
        self,
        client: WeexAsyncClient,
        ai_logger: AILogStub | None = None,
        rate_limiter: RateLimiter | None = None,
    ):
        self.client = client
        self.ai_logger = ai_logger
        self.rate_limiter = rate_limiter or RateLimiter()
        self.config = settings

    async def execute_actions(
        self,
        decisions: list[Decision],
        market_data: dict[str, MarketData],
    ) -> list[ActionResult]:
        """Execute all trading decisions.

        Args:
            decisions: List of trading decisions
            market_data: Market data for reference

        Returns:
            List of ActionResult objects
        """
        results: list[ActionResult] = []

        # Paper trading mode - save to disk instead of executing
        if PAPER_TRADING:
            add_log(f"PAPER TRADING: Processing {len(decisions)} decisions")
            for decision in decisions:
                if decision.action_type == "HOLD":
                    continue

                market = market_data.get(decision.coin)
                # Extract price from ticker dict (MarketData doesn't have .price attribute)
                if market and hasattr(market, 'ticker') and isinstance(market.ticker, dict):
                    current_price = float(market.ticker.get('last') or market.ticker.get('price') or market.ticker.get('lastPrice') or 0)
                else:
                    current_price = 0

                # Save paper trade
                save_paper_trade(
                    coin=decision.coin,
                    action=decision.action_type,
                    signal=decision.signal,
                    price=current_price,
                    size=decision.position_size,
                    confidence=decision.confidence,
                    reasoning=decision.reasoning,
                )

                add_log(
                    f"PAPER: {decision.coin} {decision.action_type} {decision.signal} @ {current_price}",
                    data={
                        "coin": decision.coin,
                        "action": decision.action_type,
                        "signal": decision.signal,
                        "price": current_price,
                        "confidence": decision.confidence,
                    },
                    coin=decision.coin,
                )

                results.append(
                    ActionResult(
                        coin=decision.coin,
                        success=True,
                        action_type=decision.action_type,
                        details={
                            "mode": "PAPER",
                            "price": current_price,
                            "signal": decision.signal,
                            "confidence": decision.confidence,
                        },
                    )
                )

            return results

        # Real trading mode
        # for decision in decisions:
        #     try:
        #         result = await self._execute_decision(decision, market_data)
        #         results.append(result)
        #     except Exception as e:
        #         results.append(
        #             ActionResult(
        #                 coin=decision.coin,
        #                 success=False,
        #                 action_type=decision.action_type,
        #                 error=str(e),
        #             )
        #         )

        # return results

    async def _execute_decision(
        self,
        decision: Decision,
        market_data: dict[str, MarketData],
    ) -> ActionResult:
        """Execute a single trading decision."""
        coin = decision.coin
        action_type = decision.action_type

        # Skip HOLD actions
        if action_type == "HOLD":
            add_log(f"{coin}: Holding position (no action needed)")
            return ActionResult(
                coin=coin,
                success=True,
                action_type="HOLD",
            )

        # Get order size from market data
        market = market_data.get(coin)
        min_order_size = self._get_min_order_size(market)
        order_size = self._calculate_order_size(min_order_size)

        if action_type == "CLOSE":
            return await self._close_position(coin, decision, order_size)

        elif action_type == "REVERSE":
            return await self._reverse_position(coin, decision, order_size)

        elif action_type == "OPEN":
            return await self._open_position(coin, decision, order_size)

        else:
            return ActionResult(
                coin=coin,
                success=False,
                action_type=action_type,
                error=f"Unknown action type: {action_type}",
            )

    async def _open_position(
        self,
        coin: str,
        decision: Decision,
        size: str,
    ) -> ActionResult:
        """Open a new position."""
        side = "BUY" if decision.signal == "LONG" else "SELL"

        add_log(
            f"{coin}: Opening {decision.signal} position",
            data={"size": size, "confidence": decision.confidence},
            coin=coin,
        )

        try:
            await self.rate_limiter.wait()

            order_request = PlaceOrderRequest(
                symbol=coin,
                client_oid=decision.correlation_id,
                size=size,
                side=side,
                type="2",  # market order
                order_type="2",  # market order
                match_price="1",  # use market price
            )

            response = await self.client.place_order(order_request)

            # Extract order_id
            order_id = (
                response.get("data", {}).get("orderId")
                or response.get("orderId")
                or response.get("order_id")
            )

            # Log AI decision
            if self.ai_logger and order_id:
                await self.ai_logger.log_decision(
                    order_request={
                        "symbol": coin,
                        "side": side,
                        "size": size,
                        "order_type": "MARKET",
                        "order_id": order_id,
                    },
                    decision_data={
                        "stage": "Decision Making",
                        "signal": decision.signal,
                        "confidence": decision.confidence,
                        "reasoning": decision.reasoning,
                        "stop_loss": decision.stop_loss,
                        "take_profit": decision.take_profit,
                        "correlation_id": decision.correlation_id,
                    },
                )

            add_log(
                f"{coin}: Position opened successfully",
                data={"order_id": order_id, "side": side, "size": size},
                coin=coin,
            )

            return ActionResult(
                coin=coin,
                success=True,
                action_type="OPEN",
                order_id=order_id or "",
                details={
                    "side": side,
                    "size": size,
                    "signal": decision.signal,
                    "confidence": decision.confidence,
                },
            )

        except Exception as e:
            add_log(
                f"{coin}: Failed to open position: {e}",
                level="ERROR",
                coin=coin,
            )
            return ActionResult(
                coin=coin,
                success=False,
                action_type="OPEN",
                error=str(e),
            )

    async def _close_position(
        self,
        coin: str,
        decision: Decision,
        size: str,
    ) -> ActionResult:
        """Close an existing position."""
        add_log(f"{coin}: Closing position", data={"size": size}, coin=coin)

        try:
            await self.rate_limiter.wait()

            # Use close_positions API
            response = await self.client.close_positions(
                symbol=coin,
                side="LONG",  # Close LONG by selling
                size=size,
            )

            order_id = (
                response.get("data", {}).get("orderId")
                or response.get("orderId")
                or response.get("order_id")
            )

            # Log AI decision
            if self.ai_logger:
                await self.ai_logger.log_decision(
                    order_request={
                        "symbol": coin,
                        "side": "CLOSE",
                        "size": size,
                        "order_type": "MARKET",
                        "order_id": order_id,
                    },
                    decision_data={
                        "stage": "Position Management",
                        "action": "CLOSE",
                        "reason": decision.reasoning,
                        "signal": decision.signal,
                        "confidence": decision.confidence,
                        "correlation_id": decision.correlation_id,
                    },
                )

            add_log(
                f"{coin}: Position closed successfully",
                data={"order_id": order_id},
                coin=coin,
            )

            return ActionResult(
                coin=coin,
                success=True,
                action_type="CLOSE",
                order_id=order_id or "",
            )

        except Exception as e:
            add_log(
                f"{coin}: Failed to close position: {e}",
                level="ERROR",
                coin=coin,
            )
            return ActionResult(
                coin=coin,
                success=False,
                action_type="CLOSE",
                error=str(e),
            )

    async def _reverse_position(
        self,
        coin: str,
        decision: Decision,
        size: str,
    ) -> ActionResult:
        """Reverse an existing position (close SHORT and open LONG, or vice versa)."""
        add_log(f"{coin}: Reversing position", data={"size": size}, coin=coin)

        # First close, then open new position
        close_result = await self._close_position(coin, decision, size)

        if not close_result.success:
            return ActionResult(
                coin=coin,
                success=False,
                action_type="REVERSE",
                error=f"Failed to close for reversal: {close_result.error}",
            )

        # Wait a bit before opening
        await self.rate_limiter.wait()

        # Open new position in opposite direction
        open_result = await self._open_position(coin, decision, size)

        if open_result.success:
            return ActionResult(
                coin=coin,
                success=True,
                action_type="REVERSE",
                order_id=open_result.order_id,
                details={
                    "close_order_id": close_result.order_id,
                    "open_order_id": open_result.order_id,
                },
            )
        else:
            return ActionResult(
                coin=coin,
                success=False,
                action_type="REVERSE",
                error=f"Close succeeded but open failed: {open_result.error}",
            )

    def _get_min_order_size(self, market: MarketData | None) -> float:
        """Get minimum order size from market data."""
        if market and market.market_overview:
            contracts = market.market_overview.get("contracts")
            if contracts:
                return float(contracts.min_order_size)
        return 0.001  # default

    def _calculate_order_size(self, min_size: float) -> str:
        """Calculate order size respecting minimum."""
        default_size = 0.001
        requested_size = max(default_size, min_size)
        return str(max(requested_size, min_size))


# =============================================================================
# Standalone Functions
# =============================================================================


async def close_all_positions() -> dict[str, dict]:
    """Close all open positions via API.

    Returns:
        Dict mapping symbol -> result dict
    """
    add_log("Closing all positions...", level="WARNING")

    client = WeexAsyncClient(config=settings)
    ai_logger = AILogStub(client=client, logger=logger)
    rate_limiter = RateLimiter()

    results: dict[str, dict] = {}

    try:
        await rate_limiter.wait()
        positions = await client.get_all_positions()

        for pos in positions:
            symbol = pos.get("symbol")
            side = pos.get("side")
            size = pos.get("size")

            if not symbol or not side or not size:
                continue

            try:
                await rate_limiter.wait()
                response = await client.close_positions(
                    symbol=symbol,
                    side=side,
                    size=str(size),
                )

                # Handle both dict and list responses
                if isinstance(response, list):
                    order_id = response[0].get("successOrderId") if response else None
                elif isinstance(response, dict):
                    order_id = (
                        response.get("data", {}).get("orderId")
                        or response.get("orderId")
                        or response.get("order_id")
                    )
                else:
                    order_id = None

                results[symbol] = {
                    "success": True,
                    "order_id": order_id,
                    "size": size,
                    "side": side,
                }

                add_log(
                    f"Closed {symbol} {side} {size}, order_id={order_id}",
                    level="WARNING",
                    coin=symbol,
                )

                # Log AI decision
                await ai_logger.log_decision(
                    order_request={
                        "symbol": symbol,
                        "side": "CLOSE",
                        "size": size,
                        "order_type": "MARKET",
                        "order_id": order_id,
                    },
                    decision_data={
                        "stage": "Manual Close",
                        "action": "CLOSE_ALL",
                        "reason": "manual_intervention",
                        "signal": "CLOSE",
                        "confidence": 1.0,
                        "position_side": side,
                        "position_size": size,
                        "close_order_id": order_id,
                    },
                )

            except Exception as e:
                results[symbol] = {
                    "success": False,
                    "error": str(e),
                    "size": size,
                    "side": side,
                }
                add_log(
                    f"Failed to close {symbol}: {e}",
                    level="ERROR",
                    coin=symbol,
                )

    finally:
        await client.close()

    return results
