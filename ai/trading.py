"""Trading loop orchestrator.

This module provides the main trading loop that orchestrates:
1. MarketAnalyzer - collects data and calculates indicators
2. EnsembleCombiner - generates trading signals from multiple strategies
3. PositionManager - executes trading decisions
"""

import asyncio
import time
from datetime import datetime
from typing import Any

from weex_client import WeexAsyncClient, config
from weex_client.exceptions import WEEXRateLimitError

from ai.ai_log import AILogStub
from ai.config import LOOP_DELAY
from ai.market_analyzer import MarketAnalyzer, RateLimiter
from ai.monitoring import metrics, start_metrics_server
from ai.position_manager import PositionManager
from ai.strategies.ensemble import EnsembleCombiner
from utils.logger import add_log

settings = config.load_config()
trade_task: asyncio.Task | None = None


def calculate_portfolio_drawdown(
    balance: list[dict[str, Any]], positions: list[dict[str, Any]]
) -> float:
    """Calculate current portfolio drawdown from balance and positions.

    Args:
        balance: Account balance response
        positions: Open positions list

    Returns:
        Drawdown as a fraction (0.0-1.0), where 0.10 = 10% drawdown
    """
    try:
        # Find USDT equity
        usdt_balance = None
        for b in balance:
            if b.get("coinName") == "USDT":
                usdt_balance = b
                break

        if not usdt_balance:
            return 0.0

        current_equity = float(balance[0].get("equity", 0))
        unrealized_pnl = float(balance[0].get("unrealizePnl", 0))

        # Estimate initial equity (current equity minus unrealized PnL)
        initial_equity = current_equity - unrealized_pnl

        if initial_equity <= 0:
            return 0.0

        # Drawdown = (initial - current) / initial
        drawdown = (initial_equity - current_equity) / initial_equity
        return max(0.0, drawdown)  # Never return negative drawdown

    except (ValueError, KeyError, IndexError):
        return 0.0


async def trade_loop():
    """Main trading loop orchestrator.

    Flow:
    1. Get ML market signals (1x per loop)
    2. Fetch market data for all coins (parallel)
    3. Calculate indicators for all coins
    4. Generate trading decisions (List[Decision]) using EnsembleCombiner
    5. Execute all decisions
    """
    # Lazy import to avoid circular dependency
    from api.dependencies import USER_COINS, bot_status

    add_log("Trade loop started")

    rate_limiter = RateLimiter(base_delay=0.015)
    cycle_start_time = time.time()
    client: WeexAsyncClient | None = None

    while bot_status["running"]:
        try:
            # Initialize components
            client = WeexAsyncClient(config=settings)
            ai_logger = AILogStub(client=client, logger=None)
            analyzer = MarketAnalyzer(client, rate_limiter)
            ensemble = EnsembleCombiner()
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
            # STEP 1: Fetch market data for all coins (parallel)
            # =================================================================
            market_data = await analyzer.get_all_market_data(USER_COINS)

            # =================================================================
            # STEP 2: Calculate indicators for all coins
            # =================================================================
            indicators = await analyzer.calculate_all_indicators(market_data)

            # =================================================================
            # STEP 3: Get ML market signals using ensemble voting
            # =================================================================
            add_log("Getting ensemble ML market signals...")
            portfolio_drawdown = calculate_portfolio_drawdown(balance, positions)
            ml_signals = await analyzer.get_ml_market_signals(
                market_data=market_data,
                indicators=indicators,
                portfolio_drawdown=portfolio_drawdown,
                current_positions=positions,
            )

            # Log summary of ML signals
            if ml_signals:
                add_log(
                    f"ML Market: {ml_signals.direction} ({ml_signals.confidence:.0%})",
                    data={"reasoning": ml_signals.reasoning},
                )

            # =================================================================
            # STEP 4: Generate trading decisions using EnsembleCombiner
            # =================================================================
            decisions = await ensemble.generate_decisions(
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
            # STEP 5: Get LLM analysis for all coins (before execution)
            # =================================================================
            llm_signals: dict[str, Any] = {}
            if analyzer.llm_client:
                try:
                    # Build coins data for LLM analysis
                    coins_data = []
                    for coin in USER_COINS:
                        ind = indicators.get(coin)
                        data = market_data.get(coin)
                        if not ind or not data:
                            continue

                        tf_1h = ind.timeframes.get("1h", {})

                        coins_data.append(
                            {
                                "coin": coin,
                                "price": ind.price or 0,
                                "rsi": tf_1h.get("rsi", 50),
                                "macd_hist": tf_1h.get("macd_histogram", 0),
                                "adx": tf_1h.get("adx", 0),
                                "trend": tf_1h.get("overall_trend", "NEUTRAL"),
                            }
                        )

                    add_log(f"Fetching LLM analysis for {len(coins_data)} coins...")
                    llm_signals = await analyzer.llm_client.get_all_signals(coins_data)
                    add_log(f"LLM analysis received for {len(llm_signals)} coins")

                    # Log LLM signals
                    for coin, sig in llm_signals.items():
                        add_log(
                            f"LLM {coin}: {sig.signal} ({sig.confidence:.0%}) - {sig.reasoning[:80]}"
                        )
                except Exception as e:
                    add_log(f"LLM analysis failed: {e}", level="WARNING")
            else:
                add_log("LLM client not available", level="WARNING")

            # =================================================================
            # STEP 6: Execute all decisions
            # =================================================================
            results = await position_mgr.execute_actions(
                decisions=decisions,
                market_data=market_data,
            )

            # Log results summary
            success_count = sum(1 for r in results if r.success)
            add_log(f"Executed {success_count}/{len(results)} actions successfully")

            # =================================================================
            # Update metrics
            # =================================================================
            cycle_duration_ms = (time.time() - cycle_start_time) * 1000
            non_hold_decisions = sum(1 for d in decisions if d.action_type != "HOLD")
            
            metrics.trading_cycle_completed(
                coins_analyzed=len(decisions),
                decisions=non_hold_decisions,
                duration_ms=cycle_duration_ms,
                success=(success_count == len(results)),
            )

            # Close client
            await client.close()
            client = None

            # =================================================================
            # Loop delay
            # =================================================================
            bot_status["last_action"] = f"Processed {len(USER_COINS)} coins"
            add_log(f"Sleeping {LOOP_DELAY} seconds...")
            cycle_start_time = time.time()
            await asyncio.sleep(LOOP_DELAY)

        except WEEXRateLimitError as e:
            rate_limiter.on_rate_limit()
            add_log(
                f"Rate limit hit: {e}, delay now {rate_limiter.current_delay * 1000:.0f}ms",
                level="WARNING",
            )
            await asyncio.sleep(5)
            # Ensure client is closed
            if client is not None:
                try:
                    await client.close()
                except Exception:
                    pass
                client = None

        except asyncio.CancelledError:
            add_log("Trade loop cancelled")
            break

        except Exception as e:
            error_msg = str(e)
            add_log(f"Error: {error_msg}", level="ERROR")
            await asyncio.sleep(5)
            # Ensure client is closed
            if client is not None:
                try:
                    await client.close()
                except Exception:
                    pass
                client = None

    # Final cleanup
    if client is not None:
        try:
            await client.close()
        except Exception:
            pass

    add_log("Trade loop stopped")


async def start_trading():
    """Start trading loop."""
    # Lazy import to avoid circular dependency
    from api.dependencies import bot_status

    global trade_task
    if trade_task and not trade_task.done():
        add_log("Trade loop already running")
        return

    # Initialize Prometheus metrics server (one-time)
    if "metrics_initialized" not in bot_status:
        try:
            start_metrics_server(9090)
            add_log("Prometheus metrics server started on port 9090")
        except Exception as e:
            add_log(f"Failed to start metrics server: {e}", level="WARNING")
        bot_status["metrics_initialized"] = True

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
