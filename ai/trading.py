import asyncio
import time
from datetime import datetime

from dotenv import load_dotenv
from weex_client import WeexAsyncClient, config
from weex_client.exceptions import WEEXRateLimitError

from ai.ai_log_stub import AILogStub
from api.dependencies import USER_COINS, bot_status
from utils.logger import add_log, add_order_book_snapshot, format_order_book_log, logger
from utils.indicators import candles_to_df, calculate_indicators

load_dotenv()
settings = config.load_config()
trade_task: asyncio.Task | None = None


class RateLimiter:
    def __init__(self, base_delay: float = 0.015):
        self.base_delay = base_delay
        self.current_delay = base_delay
        self.last_request: float = 0
        self.rate_limit_count: int = 0

    async def wait(self):
        now = time.monotonic()
        elapsed = now - self.last_request
        if elapsed < self.current_delay:
            await asyncio.sleep(self.current_delay - elapsed)
        self.last_request = time.monotonic()

    def on_rate_limit(self):
        self.current_delay = min(self.current_delay + 0.01, 0.1)
        self.rate_limit_count += 1

    def reset(self):
        self.current_delay = self.base_delay


async def trade_loop():
    """Background trading loop."""
    add_log("Trade loop started")
    rate_limiter = RateLimiter(base_delay=0.015)

    while bot_status["running"]:
        try:
            client = WeexAsyncClient(config=settings)
            ai_commit_route = AILogStub(logger=logger)

            add_log("Fetching balance and positions...")

            await rate_limiter.wait()
            balance = await client.get_account_balance()

            await rate_limiter.wait()
            positions = await client.get_all_positions()

            add_log(f"Balance: {balance}, Positions: {positions}")

            for i, coin in enumerate(USER_COINS):
                if i > 0:
                    await asyncio.sleep(0.1)

                try:
                    await rate_limiter.wait()
                    ticker = await client.get_ticker(coin)

                    await rate_limiter.wait()
                    candles_1m = await client.get_kline(coin, granularity="1m")

                    await rate_limiter.wait()
                    candles_15m = await client.get_kline(coin, granularity="15m")

                    await rate_limiter.wait()
                    candles_1h = await client.get_kline(coin, granularity="1h")

                    await rate_limiter.wait()
                    candles_4h = await client.get_kline(coin, granularity="4h")

                    await rate_limiter.wait()
                    candles_history_30m = await client.get_history_kline(coin, "30m")

                    await rate_limiter.wait()
                    candles_history_1h = await client.get_history_kline(coin, "1h")

                    add_log(
                        f"{coin}: candles: 1m={len(candles_1m)}, 15m={len(candles_15m)}, 1h={len(candles_1h)}, 4h={len(candles_4h)}"
                    )

                    indicators_1m = calculate_indicators(
                        candles_to_df(candles_1m), "1m"
                    )
                    indicators_15m = calculate_indicators(
                        candles_to_df(candles_15m), "15m"
                    )
                    indicators_1h = calculate_indicators(
                        candles_to_df(candles_1h), "1h"
                    )
                    indicators_4h = calculate_indicators(
                        candles_to_df(candles_4h), "4h"
                    )

                    price_float = 0.0
                    if isinstance(ticker, dict):
                        price = (
                            ticker.get("last")
                            or ticker.get("price")
                            or ticker.get("lastPrice")
                            or "N/A"
                        )
                        if price != "N/A":
                            try:
                                price_float = float(price)
                            except (ValueError, TypeError):
                                pass
                    elif isinstance(ticker, list) and len(ticker) > 0:
                        first = ticker[0]
                        if isinstance(first, dict):
                            price = first.get("last") or first.get("price") or "N/A"
                            if price != "N/A":
                                try:
                                    price_float = float(price)
                                except (ValueError, TypeError):
                                    pass

                    add_log(
                        f"{coin}: RSI={indicators_1h.get('rsi', 'N/A')} ({indicators_1h.get('rsi_signal', '')}), "
                        f"ADX={indicators_1h.get('adx', 'N/A')} ({indicators_1h.get('adx_strength', '')}), "
                        f"MACD={indicators_1h.get('macd_histogram', 0):+.2f}, "
                        f"ATR={indicators_1h.get('atr', 'N/A')}, "
                        f"BB={price_float:.2f}/{indicators_1h.get('bb_middle', 'N/A')} ({indicators_1h.get('bb_position', 'N/A')}% in band), "
                        f"Trend={indicators_1h.get('overall_trend', 'N')} | "
                        f"ST={indicators_1h.get('supertrend_value', 'N/A')} ({indicators_1h.get('supertrend_direction', '')})"
                    )

                    bot_status["indicators"] = bot_status.get("indicators", {})
                    bot_status["indicators"][coin] = {
                        "1m": indicators_1m,
                        "15m": indicators_15m,
                        "1h": indicators_1h,
                        "4h": indicators_4h,
                    }

                    await rate_limiter.wait()
                    next_funding_time = await client.get_next_funding_time(coin)

                    await rate_limiter.wait()
                    funding_rate = await client.get_current_funding_rate(coin)

                    await rate_limiter.wait()
                    funding_history = await client.get_funding_history(coin)

                    fr_val = funding_rate[0].funding_rate if funding_rate else "N/A"
                    add_log(f"{coin}: funding: {fr_val}%")

                    await rate_limiter.wait()
                    order_book = await client.get_order_book(coin)

                    await rate_limiter.wait()
                    try:
                        open_interest = await client.get_open_interest(coin)
                    except Exception as e:
                        add_log(
                            f"Open interest unavailable for {coin}: {e}",
                            level="WARNING",
                        )
                        open_interest = None

                    await rate_limiter.wait()
                    try:
                        market_overview = await client.get_market_overview(coin)
                    except Exception as e:
                        add_log(
                            f"Market overview unavailable for {coin}: {e}",
                            level="WARNING",
                        )
                        market_overview = None

                    if market_overview and market_overview.get("contracts"):
                        c = market_overview["contracts"]
                        add_log(
                            f"{coin}: contracts: lev={c.max_leverage}x, "
                            f"fees={c.maker_fee_rate}/{c.taker_fee_rate}, "
                            f"size={c.min_order_size}-{c.max_order_size}, "
                            f"tick={c.tick_size}, inc={c.size_increment}"
                        )

                    price = None
                    if isinstance(ticker, dict):
                        price = (
                            ticker.get("last")
                            or ticker.get("price")
                            or ticker.get("lastPrice")
                            or "N/A"
                        )

                        oi_val = open_interest.base_volume if open_interest else "N/A"
                        if oi_val is not None and oi_val != "N/A" and len(oi_val) > 6:
                            oi_val = f"{float(oi_val) / 1e6:.1f}M"

                        vol_24h = ticker.get("volume_24h") if ticker else "N/A"
                        if (
                            vol_24h is not None
                            and vol_24h != "N/A"
                            and len(vol_24h) > 6
                        ):
                            vol_24h = f"{float(vol_24h) / 1e9:.1f}B"

                        add_log(f"{coin}: market: ${price}, oi={oi_val}, vol={vol_24h}")

                        bids = order_book.get("bids", []) if order_book else []
                        asks = order_book.get("asks", []) if order_book else []

                        bid_vol = sum(float(b[1]) for b in bids) if bids else 0
                        ask_vol = sum(float(a[1]) for a in asks) if asks else 0
                        best_bid = bids[0][0] if bids else "N/A"
                        best_ask = asks[0][0] if asks else "N/A"

                        if best_bid != "N/A" and best_ask != "N/A":
                            spread_val = float(best_ask) - float(best_bid)
                            spread_pct = (spread_val / float(best_bid)) * 100
                            spread_str = f"{spread_val:.1f} ({spread_pct:.2f}%)"
                        else:
                            spread_str = "N/A"

                        if ask_vol > 0:
                            imbalance = bid_vol / ask_vol
                            if imbalance > 1.2:
                                trend = "📈"
                            elif imbalance < 0.8:
                                trend = "📉"
                            else:
                                trend = "⚖️"
                        else:
                            imbalance = 0
                            trend = "❓"

                        log_line = format_order_book_log(
                            coin, price, bid_vol, ask_vol, spread_str, imbalance, trend
                        )
                        add_log(log_line)

                        snapshot = {
                            "timestamp": datetime.now().isoformat(),
                            "price": price,
                            "bid_vol": bid_vol,
                            "ask_vol": ask_vol,
                            "best_bid": best_bid,
                            "best_ask": best_ask,
                            "spread": spread_str,
                            "imbalance": imbalance,
                        }
                        add_order_book_snapshot(coin, snapshot)

                    elif isinstance(ticker, list) and len(ticker) > 0:
                        first = ticker[0]
                        if isinstance(first, dict):
                            price = first.get("last") or first.get("price") or "N/A"
                            add_log(f"{coin}: price={price} (from list)")
                        else:
                            add_log(
                                f"{coin}: list first item not dict: {first}",
                                level="WARNING",
                            )
                    else:
                        add_log(
                            f"{coin}: unexpected format {type(ticker)}", level="WARNING"
                        )

                except Exception as e:
                    add_log(f"{coin} ticker error: {e}", level="ERROR")

            await ai_commit_route.log_decision(
                order_request={},
                decision_data={
                    "signal": "hold",
                    "stage": "Strategy Generation Next 1",
                    "Alpha": "ML Generation Ansamble - Terra Appolox",
                },
            )

            rate_limiter.reset()
            add_log(
                f"Rate limiter resets, rate limit hits: {rate_limiter.rate_limit_count}"
            )

            loop_delay = 60
            bot_status["last_action"] = f"Processed {len(USER_COINS)} coins"
            add_log(f"Sleeping {loop_delay} seconds...")
            await asyncio.sleep(loop_delay)

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
    global trade_task
    bot_status["running"] = True
    bot_status["started_at"] = datetime.now().isoformat()
    add_log("Bot started via FastAPI")
    trade_task = asyncio.create_task(trade_loop())


async def stop_trading():
    """Stop trading loop."""
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
