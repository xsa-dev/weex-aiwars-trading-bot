"""Trading loop orchestrator.

This module provides the main trading loop that orchestrates:
1. MarketAnalyzer - collects data and calculates indicators
2. Strategy - generates trading signals using AI
3. PositionManager - executes trading decisions
"""

import asyncio
from datetime import datetime

from weex_client import WeexAsyncClient, config
from weex_client.exceptions import WEEXRateLimitError

from ai.ai_log_stub import AILogStub
from ai.config import LOOP_DELAY
from ai.market_analyzer import MarketAnalyzer, RateLimiter
from ai.position_manager import PositionManager
from ai.strategy import Strategy
from utils.logger import add_log

settings = config.load_config()
trade_task: asyncio.Task | None = None


async def trade_loop():
    """Main trading loop orchestrator.

    Flow:
    1. Get ML market signals (1x per loop)
    2. Fetch market data for all coins (parallel)
    3. Calculate indicators for all coins
    4. Generate trading decisions (List[Decision])
    5. Execute all decisions
    """
    # Lazy import to avoid circular dependency
    from api.dependencies import USER_COINS, bot_status

    add_log("Trade loop started")

    rate_limiter = RateLimiter(base_delay=0.015)

    while bot_status["running"]:
        try:
            # Initialize components
            client = WeexAsyncClient(config=settings)
            ai_logger = AILogStub(client=client, logger=None)
            analyzer = MarketAnalyzer(client, rate_limiter)
            strategy = Strategy(ml_enabled=True)
            position_mgr = PositionManager(
                client=client,
                ai_logger=ai_logger,
                rate_limiter=rate_limiter,
            )

            # Fetch balance and positions
            add_log("Fetching balance and positions...")

            try:
                await rate_limiter.wait()
                balance = await client.get_account_balance()

                await rate_limiter.wait()
                positions = await client.get_all_positions()
            except Exception as e:
                add_log(
                    f"Failed to fetch balance/positions: {e}",
                    level="ERROR",
                    error_type="api_fetch_error",
                )
                await asyncio.sleep(10)
                continue

            add_log(
                f"Balance: {balance}, Positions: {positions}",
                data={"positions_count": len(positions)},
            )

            # =================================================================
            # STEP 1: Get ML market signals (1x per loop)
            # =================================================================
            add_log("Getting ML market signals...")
            ml_signals = await analyzer.get_ml_market_signals()
            add_log(
                f"ML Market: {ml_signals.direction} ({ml_signals.confidence:.0%})",
                data={"reasoning": ml_signals.reasoning},
            )

            # =================================================================
            # STEP 2: Fetch market data for all coins (parallel)
            # =================================================================
            market_data = await analyzer.get_all_market_data(USER_COINS)

            # =================================================================
            # STEP 3: Calculate indicators for all coins
            # =================================================================
            indicators = analyzer.calculate_all_indicators(market_data)

            # =================================================================
            # STEP 4: Generate trading decisions (List[Decision])
            # =================================================================
            decisions = await strategy.generate_signal(
                market_data=market_data,
                indicators=indicators,
                ml_signals=ml_signals,
                current_positions=positions,
            )

            # Log decisions summary
            decision_summary = {
                d.coin: {
                    "signal": d.signal,
                    "action": d.action_type,
                    "confidence": d.confidence,
                }
                for d in decisions
            }
            add_log(
                f"Generated {len(decisions)} trading decisions", data=decision_summary
            )

            # =================================================================
            # STEP 5: Execute all decisions
            # =================================================================
            results = await position_mgr.execute_actions(
                decisions=decisions,
                market_data=market_data,
            )

            # Log results summary
            success_count = sum(1 for r in results if r.success)
            add_log(f"Executed {success_count}/{len(results)} actions successfully")

            # Close client
            await client.close()

            # =================================================================
            # Loop delay
            # =================================================================
            bot_status["last_action"] = f"Processed {len(USER_COINS)} coins"
            add_log(f"Sleeping {LOOP_DELAY} seconds...")
            await asyncio.sleep(LOOP_DELAY)

        except WEEXRateLimitError as e:
            rate_limiter.on_rate_limit()
            add_log(
                f"Rate limit hit: {e}, delay now {rate_limiter.current_delay * 1000:.0f}ms",
                level="WARNING",
            )
            await asyncio.sleep(5)

        except asyncio.CancelledError:
            add_log("Trade loop cancelled")
            break

        except Exception as e:
            error_msg = str(e)
            add_log(f"Error: {error_msg}", level="ERROR")
            await asyncio.sleep(5)

    add_log("Trade loop stopped")


async def start_trading():
    """Start trading loop."""
    # Lazy import to avoid circular dependency
    from api.dependencies import bot_status

    global trade_task
    if trade_task and not trade_task.done():
        add_log("Trade loop already running")
        return

    bot_status["running"] = True
    bot_status["started_at"] = datetime.now().isoformat()
    trade_task = asyncio.create_task(trade_loop())
    add_log("Bot started - trade loop initiated")


async def stop_trading():
    """Stop trading loop."""
    # Lazy import to avoid circular dependency
    from api.dependencies import bot_status

    global trade_task
    add_log("Bot shutting down...")
    bot_status["running"] = False

    if trade_task and not trade_task.done():
        trade_task.cancel()
        try:
            await asyncio.wait_for(trade_task, timeout=5.0)
        except asyncio.TimeoutError:
            add_log("Trade loop did not cancel in time")
        except asyncio.CancelledError:
            pass

    add_log("Bot stopped")
