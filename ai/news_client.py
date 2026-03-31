"""News sentiment client for nirholas/free-crypto-news API.

API endpoint: http://localhost:3001/api/news
"""

import asyncio
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import aiohttp

from ai.config import (
    NEWS_SERVER_HOST,
    NEWS_API_ENDPOINT,
    NEWS_REQUEST_TIMEOUT,
    NEWS_CACHE_DURATION,
    COIN_KEYWORDS,
)
from utils.logger import add_log


@dataclass
class NewsArticle:
    """Single news article."""

    title: str
    link: str
    description: str | None
    source: str
    pub_date: str
    category: str


@dataclass
class NewsSentiment:
    """News sentiment result for a coin."""

    coin: str
    sentiment: str
    confidence: float
    articles: list[NewsArticle]
    total_articles: int
    timestamp: str


class NewsClient:
    """Async client for crypto news sentiment analysis."""

    BULLISH_KEYWORDS = [
        "bullish",
        "surge",
        "gain",
        "rise",
        "pump",
        "breakout",
        "adoption",
        "partnership",
        "launch",
        "record",
        "high",
        "rally",
        "moon",
        "soar",
        "jump",
        "upgrade",
        "approved",
    ]

    BEARISH_KEYWORDS = [
        "bearish",
        "drop",
        "fall",
        "dump",
        "crash",
        "ban",
        "hack",
        "scam",
        "warning",
        "risk",
        "investigation",
        "regulation",
        "crackdown",
        "lose",
        "plunge",
        "tumble",
    ]

    def __init__(
        self,
        host: str = NEWS_SERVER_HOST,
        endpoint: str = NEWS_API_ENDPOINT,
        timeout: int = NEWS_REQUEST_TIMEOUT,
        cache_duration: int = NEWS_CACHE_DURATION,
    ):
        self.host = host
        self.endpoint = endpoint
        self.timeout = aiohttp.ClientTimeout(total=timeout)
        self.cache_duration = cache_duration
        self.session: aiohttp.ClientSession | None = None
        self._cache: dict[str, dict[str, Any]] = {}

    async def _get_session(self) -> aiohttp.ClientSession:
        if self.session is None or self.session.closed:
            self.session = aiohttp.ClientSession(timeout=self.timeout)
        return self.session

    def _is_cached(self, coin: str) -> bool:
        if coin not in self._cache:
            return False
        cached = self._cache[coin]
        timestamp = cached.get("timestamp")
        if not timestamp:
            return False
        cached_time = datetime.fromisoformat(timestamp)
        age = (datetime.now() - cached_time).total_seconds()
        return age < self.cache_duration

    def _get_keywords(self, coin: str) -> list[str]:
        return COIN_KEYWORDS.get(
            coin, [coin.replace("cmt_", "").replace("usdt", "").lower()]
        )

    def _analyze_sentiment(self, articles: list[dict]) -> tuple[str, float, int, int]:
        if not articles:
            return "neutral", 0.5, 0, 0

        bullish_count = 0
        bearish_count = 0

        for article in articles:
            title = article.get("title", "").lower()
            description = (
                article.get("description", "").lower()
                if article.get("description")
                else ""
            )
            text = f"{title} {description}"

            is_bullish = any(kw in text for kw in self.BULLISH_KEYWORDS)
            is_bearish = any(kw in text for kw in self.BEARISH_KEYWORDS)

            if is_bullish and not is_bearish:
                bullish_count += 1
            elif is_bearish and not is_bullish:
                bearish_count += 1

        total = bullish_count + bearish_count

        if total == 0:
            return "neutral", 0.5, 0, 0

        if bullish_count > bearish_count:
            confidence = min(0.5 + (bullish_count / total) * 0.5, 0.9)
            return "bullish", confidence, bullish_count, bearish_count
        elif bearish_count > bullish_count:
            confidence = min(0.5 + (bearish_count / total) * 0.5, 0.9)
            return "bearish", confidence, bullish_count, bearish_count
        else:
            return "neutral", 0.5, bullish_count, bearish_count

    async def get_sentiment(self, coin: str) -> NewsSentiment | None:
        """Get sentiment for a specific coin."""
        if self._is_cached(coin):
            add_log(f"Using cached news for {coin}")
            return self._cache[coin]["data"]

        session = await self._get_session()
        url = f"{self.host}{self.endpoint}"

        try:
            async with session.get(url) as response:
                if response.status != 200:
                    add_log(
                        f"News API returned {response.status} for {coin}",
                        level="WARNING",
                    )
                    return None

                data = await response.json()
                articles_data = data.get("articles", [])

                keywords = self._get_keywords(coin)
                filtered_articles = []

                for article in articles_data:
                    title = article.get("title", "").lower()
                    description = (
                        article.get("description", "").lower()
                        if article.get("description")
                        else ""
                    )
                    text = f"{title} {description}"

                    if any(kw in text for kw in keywords):
                        filtered_articles.append(article)

                sentiment, confidence, bullish, bearish = self._analyze_sentiment(
                    filtered_articles
                )

                articles = [
                    NewsArticle(
                        title=a.get("title", ""),
                        link=a.get("link", ""),
                        description=a.get("description"),
                        source=a.get("source", ""),
                        pub_date=a.get("pubDate", ""),
                        category=a.get("category", "general"),
                    )
                    for a in filtered_articles[:20]
                ]

                result = NewsSentiment(
                    coin=coin,
                    sentiment=sentiment,
                    confidence=confidence,
                    articles=articles,
                    total_articles=len(filtered_articles),
                    timestamp=datetime.now().isoformat(),
                )

                self._cache[coin] = {
                    "data": result,
                    "timestamp": datetime.now(),
                }

                add_log(
                    f"News sentiment for {coin}: {sentiment} ({confidence:.0%}), {len(articles)} articles",
                    coin=coin,
                )
                return result

        except asyncio.TimeoutError:
            add_log(f"News API timeout for {coin}", level="WARNING", coin=coin)
        except Exception as e:
            add_log(
                f"News API error for {coin}: {e}",
                level="ERROR",
                coin=coin,
                error_type="news_api_error",
            )

        return None

    async def get_all_sentiments(self, coins: list[str]) -> dict[str, NewsSentiment]:
        """Get sentiment for all coins in parallel."""
        add_log(f"Fetching news sentiment for {len(coins)} coins")

        tasks = [self.get_sentiment(coin) for coin in coins]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        sentiments: dict[str, NewsSentiment] = {}

        for coin, result in zip(coins, results):
            if isinstance(result, NewsSentiment):
                sentiments[coin] = result
            elif isinstance(result, Exception):
                add_log(
                    f"Error getting news for {coin}: {result}", level="ERROR", coin=coin
                )
                sentiments[coin] = NewsSentiment(
                    coin=coin,
                    sentiment="neutral",
                    confidence=0.5,
                    articles=[],
                    total_articles=0,
                    timestamp=datetime.now().isoformat(),
                )

        return sentiments

    def clear_cache(self, coin: str | None = None):
        """Clear news cache."""
        if coin:
            self._cache.pop(coin, None)
        else:
            self._cache.clear()
        add_log(f"News cache cleared for {'all' if coin is None else coin}")

    async def close(self):
        """Close aiohttp session."""
        if self.session and not self.session.closed:
            await self.session.close()
            self.session = None


async def get_news_sentiment_for_coins(coins: list[str]) -> dict[str, NewsSentiment]:
    """Get news sentiment for a list of coins."""
    client = NewsClient()
    try:
        return await client.get_all_sentiments(coins)
    finally:
        await client.close()
