"""LLM client for DeepSeek/VSE API integration using OpenAI SDK.

Generates trading signals based on market analysis and technical indicators.
"""

import json
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from openai import AsyncOpenAI

from ai.config import (
    LLM_MODEL,
    LLM_REQUEST_TIMEOUT,
    VSE_LLM_API_BASE,
    VSE_LLM_API_KEY,
)
from ai.ensemble_voter import LLMSignal
from utils.logger import add_log


@dataclass
class LLMAnalysisResult:
    """Result from LLM analysis."""

    signal: str  # BULLISH, BEARISH, NEUTRAL
    confidence: float  # 0.0-1.0
    reasoning: str
    model: str
    timestamp: str


# System prompt for WEEX competition trading
SYSTEM_PROMPT = """Act as an expert crypto trader specializing in short-term leveraged trading for WEEX competition.

Risk Management Rules:
- Maximum risk per trade: 1% of total balance (10 USDT loss if SL hit)
- Calculation: loss at SL = margin × 20 × 0.02 = margin × 0.4 → max margin per trade = 25 USDT
- Total portfolio risk: no more than 8% of balance (80 USDT max simultaneous risk)
- Only open positions with confidence ≥ 65%
- Allocate margin proportionally to confidence level (higher confidence → larger allocation, but never exceed 25 USDT per trade)

For each pair analyze: last 24-48h price, volume, volatility, RSI, trends, sentiment.
Give 24h forecast. Recommend: buy (long), sell (short), or hold.
If buy: TP = +5%, SL = -2%. If sell: TP = -5%, SL = +2%.
Predict outcome: "TP hit", "SL hit", or "hold".
Confidence 0-100%, brief reason.

Output STRICTLY in this JSON format:
{
  "portfolio": {
    "balance_usdt": 1000,
    "leverage": 20,
    "max_risk_per_trade_usdt": 10,
    "max_total_risk_usdt": 80,
    "total_allocated_margin_usdt": float,
    "total_risk_usdt": float,
    "allocations": {
      "BTCUSDT": {"suggested_margin_usdt": float, "position_size_usdt": float, "risk_usdt": float},
      "ETHUSDT": {"suggested_margin_usdt": float, "position_size_usdt": float, "risk_usdt": float},
      "SOLUSDT": {"suggested_margin_usdt": float, "position_size_usdt": float, "risk_usdt": float},
      "DOGEUSDT": {"suggested_margin_usdt": float, "position_size_usdt": float, "risk_usdt": float},
      "XRPUSDT": {"suggested_margin_usdt": float, "position_size_usdt": float, "risk_usdt": float},
      "ADAUSDT": {"suggested_margin_usdt": float, "position_size_usdt": float, "risk_usdt": float},
      "BNBUSDT": {"suggested_margin_usdt": float, "position_size_usdt": float, "risk_usdt": float},
      "LTCUSDT": {"suggested_margin_usdt": float, "position_size_usdt": float, "risk_usdt": float}
    }
  },
  "signals": {
    "BTCUSDT": {"signal": "buy/sell/hold", "tp_price": float, "sl_price": float, "predicted_outcome": "TP hit/SL hit/hold", "confidence": int, "reason": "string"},
    "ETHUSDT": {"signal": "buy/sell/hold", "tp_price": float, "sl_price": float, "predicted_outcome": "TP hit/SL hit/hold", "confidence": int, "reason": "string"},
    "SOLUSDT": {"signal": "buy/sell/hold", "tp_price": float, "sl_price": float, "predicted_outcome": "TP hit/SL hit/hold", "confidence": int, "reason": "string"},
    "DOGEUSDT": {"signal": "buy/sell/hold", "tp_price": float, "sl_price": float, "predicted_outcome": "TP hit/SL hit/hold", "confidence": int, "reason": "string"},
    "XRPUSDT": {"signal": "buy/sell/hold", "tp_price": float, "sl_price": float, "predicted_outcome": "TP hit/SL hit/hold", "confidence": int, "reason": "string"},
    "ADAUSDT": {"signal": "buy/sell/hold", "tp_price": float, "sl_price": float, "predicted_outcome": "TP hit/SL hit/hold", "confidence": int, "reason": "string"},
    "BNBUSDT": {"signal": "buy/sell/hold", "tp_price": float, "sl_price": float, "predicted_outcome": "TP hit/SL hit/hold", "confidence": int, "reason": "string"},
    "LTCUSDT": {"signal": "buy/sell/hold", "tp_price": float, "sl_price": float, "predicted_outcome": "TP hit/SL hit/hold", "confidence": int, "reason": "string"}
  }
}

IMPORTANT: Output ONLY valid JSON, no markdown, no text, no explanations."""


class LLMClient:
    """Async client for LLM-based trading analysis using OpenAI SDK."""

    def __init__(
        self,
        api_key: str | None = None,
        api_base: str | None = None,
        model: str | None = None,
    ):
        """Initialize LLM client using OpenAI SDK."""
        self.api_key = api_key or VSE_LLM_API_KEY
        self.api_base = api_base or VSE_LLM_API_BASE
        self.model = model or LLM_MODEL

        if not self.api_key:
            add_log("LLM API key not configured", level="WARNING")
            self.client = None
        else:
            self.client = AsyncOpenAI(
                api_key=self.api_key,
                base_url=self.api_base,
                timeout=LLM_REQUEST_TIMEOUT,
            )

    async def close(self):
        """Close the OpenAI client."""
        if self.client:
            await self.client.close()

    def _extract_json(self, text: str | None) -> dict[str, Any] | None:
        """Extract JSON from LLM response text."""
        # Clean markdown code blocks
        if not text:
            raise Exception("Text is none! Json cant be processed! Check LLM response!")
        text = re.sub(r"^```json\s*", "", text, flags=re.MULTILINE)
        text = re.sub(r"^```\s*", "", text, flags=re.MULTILINE)
        text = re.sub(r"\s*```\s*$", "", text, flags=re.MULTILINE)
        text = text.strip()

        # Try direct JSON parsing
        try:
            data = json.loads(text)
            if "signals" in data and "portfolio" in data:
                return data
        except json.JSONDecodeError:
            pass

        # Try to find JSON in text
        try:
            start = text.find("{")
            end = text.rfind("}")
            if start != -1 and end != -1:
                json_str = text[start : end + 1]
                data = json.loads(json_str)
                if "signals" in data and "portfolio" in data:
                    return data
        except json.JSONDecodeError:
            pass

        return None

    async def analyze_all_coins(
        self,
        coins_data: dict[str, dict[str, Any]],
    ) -> dict[str, LLMSignal]:
        """Analyze all coins at once with portfolio-level reasoning."""
        if not self.client:
            return {}

        # Build market data summary
        market_summary = "Current Market Data:\n"
        for coin, data in coins_data.items():
            price = data.get("price", 0)
            rsi = data.get("rsi", 50)
            trend = data.get("trend", "NEUTRAL")
            market_summary += f"- {coin}: ${price:.2f}, RSI {rsi:.0f}, Trend {trend}\n"

        current_date = datetime.now().strftime("%B %-d, %Y")
        user_prompt = f"""{market_summary}

Analyze each pair and provide trading signals and portfolio allocation.
Current date: {current_date}.
Output in the exact JSON format specified in the system prompt."""

        # Prepare request for logging
        request_data = {
            "model": self.model,
            "message_count": 2,
            "coins": list(coins_data.keys()),
            "temperature": 0.3,
            "max_tokens": 10000,
        }

        add_log(
            "LLM request initiated",
            data=request_data,
            coin=",".join(coins_data.keys()),
        )

        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.3,
                max_tokens=10000,
            )

            # Extract response metadata for logging
            response_meta = {
                "model": response.model,
                "finish_reason": response.choices[0].finish_reason,
                "content_length": len(response.choices[0].message.content) if response.choices[0].message.content else 0,
            }

            # Log token usage if available
            if hasattr(response, "usage") and response.usage:
                response_meta["prompt_tokens"] = response.usage.prompt_tokens
                response_meta["completion_tokens"] = response.usage.completion_tokens
                response_meta["total_tokens"] = response.usage.total_tokens

            add_log(
                "LLM response received",
                data=response_meta,
                coin=",".join(coins_data.keys()),
            )

            content = response.choices[0].message.content

            # Log content preview (first 200 chars)
            add_log(
                "LLM content preview",
                data={"preview": content[:200] + "..." if len(content) > 200 else content},
                coin=",".join(coins_data.keys()),
            )

            result = self._extract_json(content)

            if result is None:
                add_log(
                    "LLM parsing failed",
                    data={
                        "full_response": content[:500] if content else "Empty response",
                        "model": self.model,
                    },
                    level="ERROR",
                    coin=",".join(coins_data.keys()),
                )
                return {}

            # Log parsing success
            add_log(
                "LLM parsing successful",
                data={
                    "signals_count": len(result.get("signals", {})),
                    "coins_analyzed": list(result.get("signals", {}).keys()),
                },
                coin=",".join(coins_data.keys()),
            )

            # Convert to LLMSignal dict
            signals: dict[str, LLMSignal] = {}
            if "signals" in result:
                for coin, sig_data in result["signals"].items():
                    signal = sig_data.get("signal", "HOLD").upper()
                    confidence = sig_data.get("confidence", 50) / 100.0
                    reason = sig_data.get("reason", "")

                    signals[coin] = LLMSignal(
                        signal=signal,
                        confidence=confidence,
                        reasoning=reason,
                        model=self.model,
                    )

                    add_log(f"LLM {coin}: {signal} ({confidence:.0%}) - {reason[:50]}")

            return signals

        except Exception as e:
            add_log(
                f"LLM error: {e}",
                level="ERROR",
                coin=",".join(coins_data.keys()),
                error_type="llm_failure",
            )
            return {}

    async def analyze_market(
        self,
        coin: str,
        price: float,
        price_change_24h: float,
        rsi: float,
        macd_hist: float,
        adx: float,
        trend: str,
        news_sentiment: str | None = None,
        ml_direction: str | None = None,
    ) -> LLMSignal | None:
        """Analyze single coin (fallback method)."""
        if not self.client:
            return None

        # Simple rule-based analysis as fallback
        signal = "NEUTRAL"
        confidence = 0.5
        reason = "insufficient data"

        if rsi < 35:
            signal = "BULLISH"
            confidence = 0.65
            reason = "RSI oversold"
        elif rsi > 65:
            signal = "BEARISH"
            confidence = 0.65
            reason = "RSI overbought"

        if macd_hist > 0 and signal == "NEUTRAL":
            signal = "BULLISH"
            confidence = 0.60
            reason = "MACD positive"
        elif macd_hist < 0 and signal == "NEUTRAL":
            signal = "BEARISH"
            confidence = 0.60
            reason = "MACD negative"

        return LLMSignal(
            signal=signal,
            confidence=confidence,
            reasoning=reason,
            model=self.model,
        )

    async def get_all_signals(
        self,
        coins_data: list[dict[str, Any]],
    ) -> dict[str, LLMSignal]:
        """Get LLM signals for multiple coins."""
        if not self.client:
            return {}

        # Build coins dict
        coins_dict: dict[str, dict[str, Any]] = {}
        for coin_data in coins_data:
            coin = coin_data.get("coin", "")
            if coin:
                coins_dict[coin] = coin_data

        # Use batch analysis
        return await self.analyze_all_coins(coins_dict)


async def create_llm_client() -> LLMClient:
    """Factory function to create LLM client."""
    return LLMClient()
