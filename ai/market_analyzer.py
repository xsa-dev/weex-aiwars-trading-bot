"""Market data collection, indicators calculation, and ML signal integration."""

import asyncio
import time
from dataclasses import dataclass, field
from functools import wraps
from typing import Any

from weex_client import WeexAsyncClient

from ai.config import CANDLES_CONFIG, TIMEFRAMES
from utils.indicators import calculate_indicators, candles_to_df
from utils.logger import add_log


# =============================================================================
# Rate Limiter (moved from trading.py)
# =============================================================================


class RateLimiter:
    """Rate limiter for API requests."""

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


# =============================================================================
# API Retry Decorator
# =============================================================================


def retry_on_api_error(max_retries: int = 3, delay: float = 1.0):
    """Decorator for retrying API calls on errors."""

    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            last_error = None
            for attempt in range(max_retries):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    last_error = e
                    add_log(
                        f"API error (attempt {attempt + 1}/{max_retries}): {e}",
                        level="WARNING",
                    )
                    if attempt < max_retries - 1:
                        await asyncio.sleep(delay * (attempt + 1))
            raise last_error

        return wrapper

    return decorator


# =============================================================================
# Safe Data Fetching Functions (moved from trading.py)
# =============================================================================


@retry_on_api_error(max_retries=3, delay=1.0)
async def safe_get_kline(
    client: WeexAsyncClient,
    coin: str,
    granularity: str,
    limit: int,
    max_retries: int = 3,
) -> list[list[str]]:
    """Safely fetch kline data with retry logic."""
    for attempt in range(max_retries):
        try:
            result = await client.get_kline(coin, granularity=granularity, limit=limit)
            if result is not None and len(result) > 0:
                return result
            else:
                add_log(
                    f"{coin} {granularity}: received empty data (attempt {attempt + 1}/{max_retries})",
                    level="WARNING",
                )
                if attempt < max_retries - 1:
                    await asyncio.sleep(1)
        except Exception as e:
            add_log(
                f"Error fetching {coin} {granularity} (attempt {attempt + 1}/{max_retries}): {e}",
                level="WARNING",
            )
            if attempt < max_retries - 1:
                await asyncio.sleep(1)
    return []


@retry_on_api_error(max_retries=3, delay=1.0)
async def safe_get_history_kline(
    client: WeexAsyncClient, coin: str, granularity: str, max_retries: int = 3
) -> list[list[str]]:
    """Safely fetch history kline data with retry logic."""
    for attempt in range(max_retries):
        try:
            result = await client.get_history_kline(coin, granularity)
            if result is not None and len(result) > 0:
                return result
            else:
                add_log(
                    f"{coin} {granularity}: received empty history data (attempt {attempt + 1}/{max_retries})",
                    level="WARNING",
                )
                if attempt < max_retries - 1:
                    await asyncio.sleep(1)
        except Exception as e:
            add_log(
                f"Error fetching {coin} {granularity} history (attempt {attempt + 1}/{max_retries}): {e}",
                level="WARNING",
            )
            if attempt < max_retries - 1:
                await asyncio.sleep(1)
    return []


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class MarketData:
    """Market data for a single coin."""

    coin: str
    ticker: dict[str, Any]
    candles: dict[str, list[list[str]]]
    order_book: dict[str, Any]
    funding_rate: str | None
    open_interest: dict[str, Any] | None
    market_overview: dict[str, Any] | None
    next_funding_time: str | None


@dataclass
class Indicators:
    """Technical indicators for a coin."""

    coin: str
    price: float
    timeframes: dict[str, dict[str, Any]] = field(default_factory=dict)
    overall_trend: str = "N/A"


@dataclass
class MLSignals:
    """ML-based market signals."""

    direction: str = "NEUTRAL"  # BULLISH, BEARISH, NEUTRAL
    confidence: float = 0.5
    reasoning: str = ""
    model_name: str = ""
    timestamp: str = ""


# =============================================================================
# Market Analyzer
# =============================================================================


class MarketAnalyzer:
    """Collects market data, calculates indicators, and gets ML signals."""

    def __init__(self, client: WeexAsyncClient, rate_limiter: RateLimiter):
        self.client = client
        self.rate_limiter = rate_limiter

    async def get_all_market_data(self, coins: list[str]) -> dict[str, MarketData]:
        """Fetch market data for all coins in parallel.

        Args:
            coins: List of coin symbols (e.g., ['cmt_btcusdt', 'cmt_ethusdt'])

        Returns:
            Dict mapping coin -> MarketData
        """
        add_log(f"Fetching market data for {len(coins)} coins...")

        # Create tasks for all coins
        tasks = [self._get_market_data(coin) for coin in coins]

        # Execute in parallel
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Build dict, handling exceptions
        market_data: dict[str, MarketData] = {}
        for coin, result in zip(coins, results):
            if isinstance(result, Exception):
                add_log(f"Error fetching {coin}: {result}", level="ERROR")
            else:
                market_data[coin] = result

        add_log(f"Market data fetched for {len(market_data)}/{len(coins)} coins")
        return market_data

    async def _get_market_data(self, coin: str) -> MarketData:
        """Get market data for a single coin."""
        await self.rate_limiter.wait()
        ticker = await self.client.get_ticker(coin)

        # Fetch candles for all timeframes
        candles: dict[str, list[list[str]]] = {}
        for tf in TIMEFRAMES[:4]:  # 1m, 15m, 1h, 4h
            await self.rate_limiter.wait()
            if tf == "1m":
                candles[tf] = await safe_get_kline(
                    self.client, coin, tf, CANDLES_CONFIG["1m"]
                )
            elif tf == "15m":
                candles[tf] = await safe_get_kline(
                    self.client, coin, tf, CANDLES_CONFIG["15m"]
                )
            elif tf == "1h":
                candles[tf] = await safe_get_kline(
                    self.client, coin, tf, CANDLES_CONFIG["1h"]
                )
            elif tf == "4h":
                candles[tf] = await safe_get_kline(
                    self.client, coin, tf, CANDLES_CONFIG["4h"]
                )

        # Fetch 1D and 1W history
        for tf in ["1d", "1w"]:
            await self.rate_limiter.wait()
            try:
                candles[tf] = await safe_get_history_kline(self.client, coin, tf)
            except Exception:
                candles[tf] = []

        # Fetch order book
        await self.rate_limiter.wait()
        order_book = await self.client.get_order_book(coin)

        # Fetch funding rate
        await self.rate_limiter.wait()
        funding = await self.client.get_current_funding_rate(coin)
        funding_rate = funding[0].funding_rate if funding else None

        # Fetch open interest
        await self.rate_limiter.wait()
        try:
            open_interest = await self.client.get_open_interest(coin)
        except Exception:
            open_interest = None

        # Fetch market overview
        await self.rate_limiter.wait()
        try:
            market_overview = await self.client.get_market_overview(coin)
        except Exception:
            market_overview = None

        # Fetch next funding time
        await self.rate_limiter.wait()
        try:
            next_funding = await self.client.get_next_funding_time(coin)
            next_funding_time = next_funding[0].settle_time if next_funding else None
        except Exception:
            next_funding_time = None

        return MarketData(
            coin=coin,
            ticker=ticker,
            candles=candles,
            order_book=order_book,
            funding_rate=funding_rate,
            open_interest=open_interest,
            market_overview=market_overview,
            next_funding_time=next_funding_time,
        )

    def calculate_all_indicators(
        self, market_data: dict[str, MarketData]
    ) -> dict[str, Indicators]:
        """Calculate technical indicators for all coins.

        Args:
            market_data: Dict mapping coin -> MarketData

        Returns:
            Dict mapping coin -> Indicators
        """
        indicators: dict[str, Indicators] = {}

        for coin, data in market_data.items():
            coin_indicators = self._calculate_indicators_for_coin(coin, data)
            indicators[coin] = coin_indicators

        return indicators

    def _calculate_indicators_for_coin(self, coin: str, data: MarketData) -> Indicators:
        """Calculate indicators for a single coin."""
        # Extract price from ticker
        price = "N/A"
        if isinstance(data.ticker, dict):
            price = (
                data.ticker.get("last")
                or data.ticker.get("price")
                or data.ticker.get("lastPrice")
                or "N/A"
            )

        try:
            price_float = float(price) if price != "N/A" else 0.0
        except (ValueError, TypeError):
            price_float = 0.0

        # Calculate indicators for each timeframe
        timeframes: dict[str, dict[str, Any]] = {}

        for tf, candles in data.candles.items():
            if candles:
                df = candles_to_df(candles)
                if not df.empty:
                    ind = calculate_indicators(df, tf)
                    timeframes[tf] = ind

        # Get overall trend from 1h
        overall_trend = "NEUTRAL"
        if "1h" in timeframes:
            overall_trend = timeframes["1h"].get("overall_trend", "NEUTRAL")

        return Indicators(
            coin=coin,
            price=price_float,
            timeframes=timeframes,
            overall_trend=overall_trend,
        )

    async def get_ml_market_signals(self) -> MLSignals:
        """Get ML-based market signals.

        This queries external AI/ML models to determine overall market direction.

        TODO: Implement actual ML integration
        - OpenAI/Anthropic API for LLM analysis
        - Or local ML model (Ollama/llama.cpp)
        - Or Weex AI API if available

        Returns:
            MLSignals with direction, confidence, and reasoning
        """
        # Placeholder implementation
        # TODO: Replace with actual ML API call

        return MLSignals(
            direction="NEUTRAL",
            confidence=0.5,
            reasoning="ML signals not yet implemented",
            model_name="placeholder",
            timestamp="",
        )


# =============================================================================
# Utility Functions
# =============================================================================


def format_indicators_log(coin: str, price: float, indicators: dict) -> str:
    """Format indicators for logging."""
    lines = [f"{coin} @ ${price:.2f}"]

    for tf in TIMEFRAMES:
        data = indicators.get(tf, {})
        if not data:
            lines.append(f"  {tf}: N/A")
            continue

        rsi = data.get("rsi", "N/A")
        rsi_signal = data.get("rsi_signal", "")
        adx = data.get("adx", "N/A")
        adx_strength = data.get("adx_strength", "")
        macd = data.get("macd_histogram", 0)
        st_dir = data.get("supertrend_direction", "N")
        st_val = data.get("supertrend_value", "N/A")
        trend = data.get("overall_trend", "N")

        line = f"  {tf}: RSI={rsi}"
        if rsi_signal:
            line += f" ({rsi_signal[:4]})"
        line += f", ADX={adx}"
        if adx_strength:
            line += f" ({adx_strength[:4]})"
        line += f", MACD={macd:+.2f}, ST={st_dir}({st_val}), Trend={trend}"
        lines.append(line)

    return "\n".join(lines)


def indicators_to_structured(coin: str, price: float, indicators: dict) -> dict:
    """Convert indicators to structured data for JSON logging."""
    structured = {
        "coin": coin,
        "price": price,
        "timeframes": {},
        "overall_trend": "N/A",
    }

    for tf in TIMEFRAMES:
        data = indicators.get(tf, {})
        if data:
            structured["timeframes"][tf] = {
                "rsi": data.get("rsi"),
                "rsi_signal": data.get("rsi_signal"),
                "adx": data.get("adx"),
                "adx_strength": data.get("adx_strength"),
                "macd_histogram": data.get("macd_histogram"),
                "supertrend_direction": data.get("supertrend_direction"),
                "supertrend_value": data.get("supertrend_value"),
                "overall_trend": data.get("overall_trend"),
            }

    if "1h" in indicators:
        structured["overall_trend"] = indicators["1h"].get("overall_trend", "N/A")

    return structured
