"""Position management and order execution."""

from dataclasses import dataclass
from typing import Any

from weex_client import WeexAsyncClient, config
from weex_client.models import PlaceOrderRequest

from ai.ai_log import AILogStub
from ai.config import PAPER_TRADING
from ai.market_analyzer import MarketData, RateLimiter
from ai.paper_storage import save_paper_trade
from utils.logger import add_log, logger
from utils.reward_tracker import RewardTracker
from utils.pnl_calculator import calculate_pnl

settings = config.load_config()


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
        enable_reward_tracking: bool = True,
    ):
        self.client = client
        self.ai_logger = ai_logger
        self.rate_limiter = rate_limiter or RateLimiter()
        self.config = settings
        self.enable_reward_tracking = enable_reward_tracking
        self.reward_tracker = RewardTracker() if enable_reward_tracking else None

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
                if (
                    market
                    and hasattr(market, "ticker")
                    and isinstance(market.ticker, dict)
                ):
                    current_price = float(
                        market.ticker.get("last")
                        or market.ticker.get("price")
                        or market.ticker.get("lastPrice")
                        or 0
                    )
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
        for decision in decisions:
            try:
                result = await self._execute_decision(decision, market_data)
                results.append(result)
            except Exception as e:
                results.append(
                    ActionResult(
                        coin=decision.coin,
                        success=False,
                        action_type=decision.action_type,
                        error=str(e),
                    )
                )

        return results

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
        # Weex futures API uses 'type' field:
        # 1 = Open long, 2 = Open short, 3 = Close long, 4 = Close short
        # decision.signal contains "BUY" or "SELL" from ensemble.py
        order_type_value = "1" if decision.signal == "BUY" else "2"

        add_log(
            f"{coin}: Opening {decision.signal} position",
            data={"size": size, "confidence": decision.confidence},
            coin=coin,
        )

        try:
            await self.rate_limiter.wait()

            # Add stop loss and take profit if available
            stop_loss_price = str(decision.stop_loss) if decision.stop_loss else None

            order_request = PlaceOrderRequest(
                symbol=coin,
                client_oid=decision.correlation_id,
                size=size,
                type=order_type_value,  # 1=Open long, 2=Open short
                order_type="1",  # Post-Only order (required by weex_client validation)
                match_price="1",  # Use market price
                preset_stop_loss_price=stop_loss_price,  # Stop loss
                preset_take_profit_price=str(decision.take_profit)
                if decision.take_profit
                else None,  # Take profit
            )

            response = await self.client.place_order(order_request)

            # Extract order_id
            # Handle close_positions response (returns list of results)
            order_id = None
            if isinstance(response, list) and len(response) > 0:
                # Response is a list of close results
                first_result = response[0]
                order_id = first_result.get("successOrderId") or first_result.get(
                    "orderId"
                )
                if not first_result.get("success"):
                    raise Exception(
                        f"Close failed: {first_result.get('errorMessage', 'Unknown error')}"
                    )
            elif isinstance(response, dict):
                # Response is a dict with order_id at top level (from weex_client place_order)
                order_id = (
                    response.get("order_id")
                    or response.get("data", {}).get("orderId")
                    or response.get("orderId")
                )

            # Log AI decision
            if self.ai_logger and order_id:
                await self.ai_logger.log_decision(
                    order_request={
                        "symbol": coin,
                        "type": order_type_value,
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
                data={
                    "order_id": order_id,
                    "type": order_type_value,
                    "size": size,
                    "stop_loss": decision.stop_loss,
                },
                coin=coin,
            )

            # Track trade opening for reward system
            if self.reward_tracker and order_id:
                entry_price_val = (
                    decision.stop_loss * 0.98 if decision.stop_loss else 0
                )  # approximate from SL
                self.reward_tracker.save_trade_opened(
                    trade_id=decision.correlation_id or str(order_id),
                    coin=coin,
                    entry_price=entry_price_val,
                    direction=decision.signal,
                    stop_loss=decision.stop_loss
                    or (entry_price_val * 0.98 if entry_price_val else 0),
                    take_profit=decision.take_profit
                    or (entry_price_val * 1.03 if entry_price_val else 0),
                )

            return ActionResult(
                coin=coin,
                success=True,
                action_type="OPEN",
                order_id=order_id or "",
                details={
                    "type": order_type_value,
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

            # Fetch current position to get side and size
            positions = await self.client.get_all_positions()
            current_position = None
            for pos in positions:
                if pos.get("symbol") == coin:
                    current_position = pos
                    break

            if not current_position:
                raise Exception(f"No open position found for {coin}")

            position_side = current_position.get("side")
            position_size = current_position.get("size")

            if not position_side or not position_size:
                raise Exception(f"Invalid position data for {coin}: side={position_side}, size={position_size}")

            add_log(
                f"{coin}: Closing {position_side} position, size={position_size}",
                coin=coin,
            )

            # Use close_positions API with required parameters
            response = await self.client.close_positions(
                symbol=coin,
                side=position_side,  # LONG or SHORT
                size=str(position_size),
            )

            # Handle close_positions response (returns list of results)
            order_id = None
            if isinstance(response, list) and len(response) > 0:
                # Response is a list of close results
                first_result = response[0]
                order_id = first_result.get("successOrderId") or first_result.get(
                    "orderId"
                )
                if not first_result.get("success"):
                    raise Exception(
                        f"Close failed: {first_result.get('errorMessage', 'Unknown error')}"
                    )
            elif isinstance(response, dict):
                # Fallback for dict response
                order_id = response.get("data", {}).get("orderId") or response.get(
                    "orderId"
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

    async def _place_take_profit(
        self,
        symbol: str,
        position_side: str,
        size: str,
        trigger_price: str,
        correlation_id: str,
    ) -> None:
        """Place a take profit order using the TP/SL API."""
        tp_client_oid = f"tp_{correlation_id}"

        # Weex TP/SL API expects uppercase positionSide
        tp_payload = {
            "symbol": symbol,
            "clientOrderId": tp_client_oid,
            "planType": "profit_plan",
            "triggerPrice": trigger_price,
            "executePrice": "0",  # Market price
            "size": size,
            "positionSide": position_side.upper(),  # Ensure uppercase
        }

        add_log(
            f"{symbol}: Placing TP order at {trigger_price}",
            data={"payload": tp_payload},
            coin=symbol,
        )

        # Call Weex TP/SL API
        tp_response = await self.client.request(
            method="POST",
            url_or_path="/capi/v2/order/placeTpSlOrder",
            json=tp_payload,
        )

        add_log(
            f"{symbol}: Take profit order placed at {trigger_price}",
            data={"tp_response": tp_response},
            coin=symbol,
        )


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
    AILogStub(client=client, logger=logger)
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
                # weex_client requires side and size parameters
                response = await client.close_positions(
                    symbol=symbol,
                    side=side,  # LONG or SHORT
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
