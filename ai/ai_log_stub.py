"""
AI Logging stub for WEEX AI Wars compliance.

This module provides AI decision logging functionality with:
- Weex API integration for AI log upload
- Retry logic with exponential backoff
- Fallback to local logging on API failure
- Order-to-log linking for compliance
"""

import asyncio
import time
from datetime import datetime
from typing import Any, Dict

from weex_client import WeexAsyncClient


class AILogStub:
    """
    AI Logging stub for WEEX AI Wars compliance.

    Responsibilities:
    - Capture AI decision logs
    - Upload to Weex API endpoint with retry
    - Link decisions to orders
    - Fallback to local logging on API failure
    """

    MODEL_VERSION = "SemiAuto-v1.0.1"

    STAGES = [
        "Data Collection",
        "Indicator Calculation",
        "Signal Generation",
        "Strategy Generation",
        "Decision Making",
        "Risk Assessment",
        "Order Placement",
        "Order Execution",
        "Portfolio Management",
    ]

    # Retry configuration
    MAX_RETRIES = 3
    INITIAL_DELAY = 1.0  # seconds
    MAX_DELAY = 10.0  # seconds
    BACKOFF_MULTIPLIER = 2.0

    def __init__(
        self,
        client: WeexAsyncClient | None = None,
        logger=None,
    ):
        """
        Initialize AI Log stub.

        Args:
            client: Optional WeexAsyncClient instance (will be created if not provided)
            logger: Optional logger instance
        """
        self._logger = logger
        self._client = client
        self._pending_logs: dict[str, dict] = {}
        self._upload_stats = {
            "success": 0,
            "errors": 0,
            "retries": 0,
            "skipped": 0,
        }

    def _get_client(self) -> WeexAsyncClient:
        """Get Weex client, raising error if not configured."""
        if self._client is None:
            raise ValueError("WeexAsyncClient not configured")
        return self._client

    async def log_decision(
        self,
        order_request: Dict[str, Any],
        decision_data: Dict[str, Any],
    ) -> bool:
        """
        Log AI decision with upload to Weex API and retry logic.

        Args:
            order_request: Order details (symbol, quantity, etc.)
            decision_data: AI decision details (signal, confidence, etc.)

        Returns:
            True if logged successfully (local + API), False on final failure
        """
        # Build payload
        payload = self._build_payload(
            order_request=order_request,
            decision_data=decision_data,
        )

        # Store locally first
        correlation_id = decision_data.get("correlation_id") or order_request.get(
            "client_order_id"
        )
        if correlation_id:
            self._pending_logs[correlation_id] = {
                "payload": payload,
                "timestamp": datetime.utcnow().isoformat(),
                "order_request": order_request,
                "decision_data": decision_data,
            }

        # Log locally
        if self._logger:
            self._logger.warning(
            f"[AI_LOG] Decision: {decision_data.get('signal', 'N/A')} | "
            f"Symbol: {order_request.get('symbol', 'N/A')} | "
            f"Stage: {decision_data.get('stage', 'Decision Making')} | "
            f"Model: {self.MODEL_VERSION}"
        )

        # Try to upload to Weex API with retry
        success = await self._upload_with_retry(
            payload=payload,
            correlation_id=correlation_id,
        )

        return success

    async def _upload_with_retry(
        self,
        payload: dict[str, Any],
        correlation_id: str | None = None,
    ) -> bool:
        """
        Upload AI log with exponential backoff retry.

        Args:
            payload: AI log payload
            correlation_id: Optional correlation ID for tracking

        Returns:
            True if uploaded successfully, False if skipped after retries
        """
        delay = self.INITIAL_DELAY
        last_error = None

        for attempt in range(self.MAX_RETRIES + 1):  # +1 for initial attempt
            try:
                client = self._get_client()

                response = await client.upload_ai_log(
                    stage=payload["stage"],
                    model=payload["model"],
                    input_data=payload["input"],
                    output_data=payload["output"],
                    explanation=payload["explanation"],
                    order_id=payload.get("orderId"),
                )

                if response.is_success:
                    self._upload_stats["success"] += 1
                    self._logger.info(
                        "[AI_LOG] Uploaded successfully to Weex",
                        stage=payload["stage"],
                        order_id=payload.get("orderId"),
                        attempt=attempt + 1,
                    )
                    return True
                else:
                    last_error = f"API error: {response.msg}"
                    if self._logger:
                        self._logger.warning(
                            f"[AI_LOG] Weex API error: {response.msg}",
                            code=response.code,
                            attempt=attempt + 1,
                        )

            except Exception as e:
                last_error = str(e)
                if self._logger:
                    self._logger.warning(
                        f"[AI_LOG] Upload attempt {attempt + 1} failed: {e}",
                    )

            # Check if we should retry
            if attempt < self.MAX_RETRIES:
                self._upload_stats["retries"] += 1
                await asyncio.sleep(delay)
                delay = min(delay * self.BACKOFF_MULTIPLIER, self.MAX_DELAY)

        # All retries exhausted - skip this log
        self._upload_stats["errors"] += 1
        self._upload_stats["skipped"] += 1
        if self._logger:
            self._logger.warning(
                f"[AI_LOG] Skipped after {self.MAX_RETRIES + 1} attempts (last_error: {last_error})"
            )
        return False

    def _build_payload(
        self,
        order_request: Dict[str, Any],
        decision_data: Dict[str, Any],
    ) -> dict[str, Any]:
        """Build AI log payload from decision data."""

        stage = decision_data.get("stage", "Decision Making")
        signal = decision_data.get("signal", "HOLD")
        confidence = decision_data.get("confidence", 0.0)
        indicators = decision_data.get("indicators", {})
        reasoning = decision_data.get("reasoning", "")
        symbol = order_request.get("symbol", "N/A")

        # Build input (what was fed to the AI model)
        input_data = {
            "prompt": decision_data.get(
                "prompt", f"Analyze {symbol} market conditions"
            ),
            "market_data": {
                "symbol": symbol,
                "side": order_request.get("side"),
                "size": order_request.get("size"),
                "order_type": order_request.get("order_type"),
            },
            "indicators": self._serialize_indicators(indicators),
            "temporal_context": {
                "timestamp": datetime.utcnow().isoformat(),
                "stage": stage,
            },
        }

        # Build output (what the AI model produced)
        output_data = {
            "signal": signal,
            "confidence": confidence,
            "position_size": order_request.get("size"),
            "stop_loss": decision_data.get("stop_loss"),
            "take_profit": decision_data.get("take_profit"),
            "risk_reward_ratio": decision_data.get("risk_reward_ratio"),
            "timeframes_analyzed": decision_data.get("timeframes", []),
            "signals_triggered": decision_data.get("signals", []),
        }

        # Build explanation (natural language summary)
        explanation = self._build_explanation(
            stage=stage,
            signal=signal,
            confidence=confidence,
            indicators=indicators,
            reasoning=reasoning,
            symbol=symbol,
        )

        return {
            "orderId": int(time.time() * 1000),
            "stage": stage,
            "model": self.MODEL_VERSION,
            "input": input_data,
            "output": output_data,
            "explanation": explanation,
        }

    def _serialize_indicators(self, indicators: Dict[str, Any]) -> Dict[str, Any]:
        """Serialize indicators dict for JSON logging."""
        if not indicators:
            return {}

        serialized = {}
        for tf, data in indicators.items():
            if isinstance(data, dict):
                # Extract key values for logging
                serialized[tf] = {
                    "rsi": data.get("rsi"),
                    "overall_trend": data.get("overall_trend"),
                    "macd_histogram": data.get("macd_histogram"),
                    "adx": data.get("adx"),
                    "supertrend_direction": data.get("supertrend_direction"),
                }
            else:
                serialized[tf] = data

        return serialized

    def _build_explanation(
        self,
        stage: str,
        signal: str,
        confidence: float,
        indicators: Dict[str, Any],
        reasoning: str,
        symbol: str,
    ) -> str:
        """Build natural language explanation (max 1000 chars)."""

        parts = []

        # Stage context
        parts.append(f"Stage: {stage}")

        # Decision summary
        if symbol and symbol != "N/A":
            parts.append(f"Symbol: {symbol}")

        parts.append(f"Decision: {signal}")
        parts.append(f"Confidence: {confidence:.1%}")

        # Key indicator values from 1h timeframe
        if indicators:
            tf_1h = indicators.get("1h") if isinstance(indicators, dict) else None
            if tf_1h and isinstance(tf_1h, dict):
                rsi = tf_1h.get("rsi")
                trend = tf_1h.get("overall_trend")
                if rsi is not None:
                    parts.append(f"RSI(14): {rsi}")
                if trend:
                    parts.append(f"Trend: {trend}")

        # Reasoning or default
        if reasoning:
            parts.append(f"Reasoning: {reasoning}")
        elif signal != "HOLD":
            parts.append("Multi-timeframe analysis indicates favorable conditions")

        explanation = " | ".join(parts)

        # Truncate if needed (max 1000 chars)
        if len(explanation) > 1000:
            explanation = explanation[:997] + "..."

        return explanation

    async def link_order_to_log(
        self,
        client_order_id: str,
        order_id: str,
        execution_details: Dict[str, Any] | None = None,
    ) -> bool:
        """
        Link a Weex order ID to a pending AI log.

        Args:
            client_order_id: Client-side order ID
            order_id: Weex order ID
            execution_details: Optional execution info

        Returns:
            True if linked and uploaded successfully
        """
        if client_order_id not in self._pending_logs:
            if self._logger:
                self._logger.warning(f"[AI_LOG] No pending log found for {client_order_id}")
            return False

        # Update pending log with order ID
        self._pending_logs[client_order_id]["order_id"] = order_id
        self._pending_logs[client_order_id]["execution_details"] = execution_details
        self._pending_logs[client_order_id]["updated_at"] = (
            datetime.utcnow().isoformat()
        )

        # Rebuild payload with order ID
        payload = self._pending_logs[client_order_id]["payload"]
        payload["orderId"] = int(order_id) if order_id else int(time.time() * 1000)

        # Upload with retry
        success = await self._upload_with_retry(
            payload=payload,
            correlation_id=client_order_id,
        )

        if success:
            self._logger.info(f"[AI_LOG] Linked order: {client_order_id} -> {order_id}")

        return success

    async def log_execution_result(
        self,
        order_id: str,
        execution_result: Dict[str, Any],
    ) -> bool:
        """
        Log order execution result as a new AI log entry.

        Args:
            order_id: Weex order ID
            execution_result: Fill price, fees, status, etc.

        Returns:
            True if logged successfully
        """
        payload = {
            "orderId": int(order_id) if order_id else int(time.time() * 1000),
            "stage": "Order Execution",
            "model": self.MODEL_VERSION,
            "input": {
                "order_id": order_id,
                "execution_result": execution_result,
            },
            "output": {
                "status": execution_result.get("status"),
                "fill_price": execution_result.get("fill_price"),
                "filled_qty": execution_result.get("filled_qty"),
                "fees": execution_result.get("fees"),
            },
            "explanation": f"Order {order_id} execution: {execution_result.get('status', 'unknown')}",
        }

        return await self._upload_with_retry(payload=payload)

    def get_pending_count(self) -> int:
        """Get count of pending AI logs."""
        return len(self._pending_logs)

    def get_stats(self) -> Dict[str, int]:
        """Get upload statistics."""
        return self._upload_stats.copy()

    def clear_stats(self) -> None:
        """Reset upload statistics."""
        self._upload_stats = {
            "success": 0,
            "errors": 0,
            "retries": 0,
            "skipped": 0,
        }
