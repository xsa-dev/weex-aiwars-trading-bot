"""Market data collection, indicators calculation, and ML signal integration."""

import asyncio
from asyncio import to_thread
import time
from dataclasses import dataclass, field
from functools import wraps
from typing import Any

from weex_client import WeexAsyncClient

from ai.config import CANDLES_CONFIG, TIMEFRAMES
from ai.ensemble_voter import TechnicalSignal
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

    coin: str = ""
    direction: str = "NEUTRAL"  # BULLISH, BEARISH, NEUTRAL
    confidence: float = 0.5
    reasoning: str = ""
    model_name: str = ""
    timestamp: str = ""
    source_breakdown: dict[str, dict[str, Any]] | None = None
    risk_level: str = "MEDIUM"
    stop_loss: float | None = None
    take_profit: float | None = None


# =============================================================================
# Market Analyzer
# =============================================================================


class MarketAnalyzer:
    """Collects market data, calculates indicators, and gets ML signals."""

    def __init__(self, client: WeexAsyncClient, rate_limiter: RateLimiter):
        self.client = client
        self.rate_limiter = rate_limiter

        # Ensemble clients (lazy loaded)
        self._ml_client = None
        self._news_client = None
        self._llm_client = None
        self._ensemble_voter = None
        self._model_tracker = None

    @property
    def ml_client(self):
        """Lazy load ML client for localhost:8000."""
        if self._ml_client is None:
            try:
                from ai.ml_client import MLClient

                self._ml_client = MLClient()
            except Exception as e:
                add_log(f"Failed to initialize ML client: {e}", level="ERROR")
                self._ml_client = None
        return self._ml_client

    @property
    def news_client(self):
        """Lazy load News client for localhost:3001."""
        if self._news_client is None:
            try:
                from ai.news_client import NewsClient

                self._news_client = NewsClient()
            except Exception as e:
                add_log(f"Failed to initialize News client: {e}", level="ERROR")
                self._news_client = None
        return self._news_client

    @property
    def ensemble_voter(self):
        """Lazy load Ensemble voter."""
        if self._ensemble_voter is None:
            try:
                from ai.ensemble_voter import EnsembleVoter

                self._ensemble_voter = EnsembleVoter()
            except Exception as e:
                add_log(f"Failed to initialize Ensemble voter: {e}", level="ERROR")
                self._ensemble_voter = None
        return self._ensemble_voter

    @property
    def model_tracker(self):
        """Lazy load Model tracker."""
        if self._model_tracker is None:
            try:
                from ai.model_tracker import ModelTracker

                self._model_tracker = ModelTracker()
            except Exception as e:
                add_log(f"Failed to initialize Model tracker: {e}", level="ERROR")
                self._model_tracker = None
        return self._model_tracker

    @property
    def llm_client(self):
        """Lazy load LLM client for DeepSeek/VSE API."""
        if self._llm_client is None:
            try:
                from ai.llm_client import LLMClient

                self._llm_client = LLMClient()
            except Exception as e:
                add_log(f"Failed to initialize LLM client: {e}", level="ERROR")
                self._llm_client = None
        return self._llm_client

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

    async def calculate_all_indicators(
        self, market_data: dict[str, MarketData]
    ) -> dict[str, Indicators]:
        """Calculate technical indicators for all coins in parallel.

        Args:
            market_data: Dict mapping coin -> MarketData

        Returns:
            Dict mapping coin -> Indicators
        """
        if not market_data:
            return {}

        # Create tasks for all coins
        tasks = [
            self._calculate_indicators_for_coin(coin, data)
            for coin, data in market_data.items()
        ]

        # Execute in parallel
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Build dict, handling exceptions
        indicators: dict[str, Indicators] = {}
        for (coin, data), result in zip(market_data.items(), results):
            if isinstance(result, Exception):
                add_log(
                    f"Error calculating indicators for {coin}: {result}",
                    level="ERROR",
                    coin=coin,
                    error_type="indicator_error",
                )
            else:
                indicators[coin] = result

        add_log(f"Indicators calculated for {len(indicators)}/{len(market_data)} coins")
        return indicators

    def _build_positions_dict(
        self, current_positions: list[dict[str, Any]]
    ) -> dict[str, dict[str, Any]]:
        """Convert list of positions to dict keyed by coin symbol.

        Args:
            current_positions: List of position dictionaries

        Returns:
            Dict mapping coin symbol -> position data
        """
        positions_dict: dict[str, dict[str, Any]] = {}
        for position in current_positions:
            coin = position.get("symbol") or position.get("coin")
            if coin:
                positions_dict[coin] = position
        return positions_dict

    def _build_technical_signal(self, coin: str, ind: Indicators) -> TechnicalSignal:
        """Convert technical indicators to a TechnicalSignal for ensemble voting.

        Args:
            coin: Coin symbol
            ind: Calculated indicators

        Returns:
            TechnicalSignal with direction, confidence, and reasoning
        """
        # Get 1h timeframe as primary reference
        tf_data = ind.timeframes.get("1h", {})

        # Determine overall direction from indicators
        trend = tf_data.get("overall_trend", "NEUTRAL")
        rsi = tf_data.get("rsi", 50)
        rsi_signal = tf_data.get("rsi_signal", "")
        adx = tf_data.get("adx", 0)
        adx_strength = tf_data.get("adx_strength", "")
        macd_hist = tf_data.get("macd_histogram", 0)
        st_dir = tf_data.get("supertrend_direction", "N")
        st_value = tf_data.get("supertrend_value", 0)

        # Calculate technical direction and confidence
        bullish_signals = 0
        bearish_signals = 0
        indicators_used: dict[str, Any] = {}

        # Trend analysis
        if trend == "BULLISH":
            bullish_signals += 2
            indicators_used["trend"] = "BULLISH"
        elif trend == "BEARISH":
            bearish_signals += 2
            indicators_used["trend"] = "BEARISH"
        else:
            indicators_used["trend"] = "NEUTRAL"

        # RSI analysis
        if rsi > 70:
            bearish_signals += 1
            indicators_used["rsi"] = f"{rsi:.1f} (overbought)"
        elif rsi < 30:
            bullish_signals += 1
            indicators_used["rsi"] = f"{rsi:.1f} (oversold)"
        else:
            indicators_used["rsi"] = f"{rsi:.1f}"

        if rsi_signal:
            indicators_used["rsi_signal"] = rsi_signal

        # ADX analysis
        if adx >= 25:
            if adx_strength == "STRONG":
                if trend == "BULLISH":
                    bullish_signals += 1
                elif trend == "BEARISH":
                    bearish_signals += 1
            indicators_used["adx"] = f"{adx:.1f} ({adx_strength})"
        else:
            indicators_used["adx"] = f"{adx:.1f} (weak)"

        # MACD analysis
        if macd_hist > 0:
            bullish_signals += 1
            indicators_used["macd"] = f"+{macd_hist:.4f}"
        elif macd_hist < 0:
            bearish_signals += 1
            indicators_used["macd"] = f"{macd_hist:.4f}"
        else:
            indicators_used["macd"] = "0.0000"

        # SuperTrend analysis
        if st_dir == "UP":
            bullish_signals += 1
            indicators_used["supertrend"] = f"UP ({st_value:.2f})"
        elif st_dir == "DOWN":
            bearish_signals += 1
            indicators_used["supertrend"] = f"DOWN ({st_value:.2f})"
        else:
            indicators_used["supertrend"] = f"N ({st_value:.2f})"

        # Determine signal
        if bullish_signals > bearish_signals:
            direction = "bullish"
        elif bearish_signals > bullish_signals:
            direction = "bearish"
        else:
            direction = "neutral"

        # Calculate confidence (0.5-0.9 range based on signal strength)
        total_signals = bullish_signals + bearish_signals
        if total_signals > 0:
            max_signals = max(bullish_signals, bearish_signals)
            confidence = 0.5 + (max_signals / total_signals) * 0.4
        else:
            confidence = 0.5

        # Generate reasoning
        reasoning_parts = []
        if direction == "bullish":
            reasoning_parts.append("Technical indicators suggest upward momentum")
        elif direction == "bearish":
            reasoning_parts.append("Technical indicators suggest downward pressure")
        else:
            reasoning_parts.append("Mixed technical signals")

        if adx >= 25:
            reasoning_parts.append(f"ADX {adx:.1f} indicates strong trend")

        reasoning = ". ".join(reasoning_parts)

        return TechnicalSignal(
            signal=direction,
            confidence=confidence,
            reasoning=reasoning,
            indicators_used=indicators_used,
        )

    async def _calculate_indicators_for_coin(
        self, coin: str, data: MarketData
    ) -> Indicators:
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
                try:
                    df = await to_thread(candles_to_df, candles)
                    if not df.empty:
                        ind = await to_thread(calculate_indicators, df, tf)
                        timeframes[tf] = ind
                except Exception as e:
                    add_log(
                        f"Error calculating indicators for {coin} {tf}: {e}",
                        level="ERROR",
                        coin=coin,
                        error_type="indicator_error",
                    )

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

    async def get_ml_market_signals(
        self,
        market_data: dict[str, MarketData],
        indicators: dict[str, Indicators],
        portfolio_drawdown: float = 0.0,
        current_positions: list[dict[str, Any]] | None = None,
    ) -> MLSignals:
        """Get ML-based market signals using ensemble voting.

        Combines:
        - Technical analysis (RSI, ADX, MACD, SuperTrend)
        - ML predictions (from localhost:8000)
        - News sentiment (from localhost:3001)
        - LLM analysis (TBD - from DeepSeek)

        Args:
            market_data: Dict mapping coin -> MarketData
            indicators: Dict mapping coin -> Indicators
            portfolio_drawdown: Current portfolio drawdown (0.0-1.0)
            current_positions: List of current positions

        Returns:
            Single MLSignals representing aggregate market sentiment
        """
        import time
        from ai.config import ML_PREDICTION_HOURS

        add_log("Getting ensemble ML market signals...")

        coins = list(market_data.keys())
        signals: dict[str, MLSignals] = {}

        # Fetch ML predictions, News sentiment, and LLM analysis in parallel
        add_log("Fetching ML predictions, News, and LLM analysis in parallel...")

        ml_predictions: dict[str, Any] = {}
        news_sentiments: dict[str, Any] = {}

        async def get_ml_predictions_task() -> dict[str, Any]:
            if self.ml_client:
                try:
                    add_log("Fetching ML predictions from localhost:8000...")
                    predictions = await self.ml_client.get_all_ensemble_predictions(
                        coins, prediction_hours=ML_PREDICTION_HOURS
                    )
                    add_log(f"ML predictions received for {len(predictions)} coins")
                    return predictions
                except Exception as e:
                    add_log(
                        f"Failed to get ML predictions: {e}",
                        level="WARNING",
                        error_type="ml_prediction_error",
                    )
            else:
                add_log("ML client not available", level="WARNING")
            return {}

        async def get_news_sentiment_task() -> dict[str, Any]:
            if self.news_client:
                try:
                    add_log(
                        "Fetching news sentiment from https://free-crypto-news-api.cloudpub.ru..."
                    )
                    sentiments = await self.news_client.get_all_sentiments(coins)
                    add_log(f"News sentiment received for {len(sentiments)} coins")
                    return sentiments
                except Exception as e:
                    add_log(
                        f"Failed to get news sentiment: {e}",
                        level="WARNING",
                        error_type="news_sentiment_error",
                    )
            else:
                add_log("News client not available", level="WARNING")
            return {}



        # Run ML and News tasks in parallel (LLM handled separately in trading.py)
        ml_predictions, news_sentiments = await asyncio.gather(
            get_ml_predictions_task(),
            get_news_sentiment_task(),
        )

        # Build positions dict
        positions_dict = self._build_positions_dict(current_positions or [])

        for coin in coins:
            try:
                ind = indicators.get(coin)
                data = market_data.get(coin)

                if not ind or not data:
                    continue

                # Build technical signal
                technical_signal = self._build_technical_signal(coin, ind)

                # Get ML and News signals
                ml_signal = ml_predictions.get(coin)
                news_signal = news_sentiments.get(coin)

                # Get current position
                current_position = positions_dict.get(coin)

                # Generate ensemble signal
                if self.ensemble_voter:
                    ensemble = self.ensemble_voter.vote(
                        coin=coin,
                        price=ind.price,
                        technical=technical_signal,
                        llm=None,  # LLM handled separately in trading.py
                        ml=ml_signal,
                        news=news_signal,
                        current_portfolio_drawdown=portfolio_drawdown,
                        current_position=current_position,
                    )

                    signals[coin] = MLSignals(
                        direction=ensemble.signal.upper(),
                        confidence=ensemble.confidence,
                        reasoning=ensemble.reasoning,
                        model_name="ensemble",
                        timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
                        source_breakdown=ensemble.source_breakdown,
                        risk_level=ensemble.risk_level,
                        stop_loss=ensemble.stop_loss,
                        take_profit=ensemble.take_profit,
                    )
                else:
                    # Fallback to technical-only signal
                    signals[coin] = MLSignals(
                        direction=technical_signal.signal.upper(),
                        confidence=technical_signal.confidence,
                        reasoning=technical_signal.reasoning,
                        model_name="technical",
                        timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
                    )

                add_log(
                    f"Signal for {coin}: {signals[coin].direction} ({signals[coin].confidence:.0%})",
                    coin=coin,
                )

            except Exception as e:
                add_log(
                    f"Error generating signal for {coin}: {e}",
                    level="ERROR",
                    coin=coin,
                    error_type="signal_generation_error",
                )

        add_log(f"Ensemble signals generated for {len(signals)}/{len(coins)} coins")

        # Aggregate signals into a single market-wide MLSignals
        if not signals:
            return MLSignals(
                direction="NEUTRAL",
                confidence=0.5,
                reasoning="No signals generated",
                model_name="ensemble",
                timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
            )

        # Count directions
        directions: dict[str, int] = {}
        total_confidence = 0.0
        for sig in signals.values():
            direction = sig.direction.upper()
            directions[direction] = directions.get(direction, 0) + 1
            total_confidence += sig.confidence

        # Get most common direction
        from collections import Counter

        most_common = Counter(directions).most_common(1)[0]
        aggregate_direction = most_common[0]
        aggregate_confidence = total_confidence / len(signals)

        # Generate aggregate reasoning
        direction_counts = ", ".join(f"{d}: {c}" for d, c in sorted(directions.items()))
        aggregate_reasoning = (
            f"Market aggregate: {aggregate_direction} ({aggregate_confidence:.0%} confidence). "
            f"Signal distribution: {direction_counts}. "
            f"Analyzed {len(signals)} coins."
        )

        return MLSignals(
            direction=aggregate_direction,
            confidence=aggregate_confidence,
            reasoning=aggregate_reasoning,
            model_name="ensemble",
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
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
