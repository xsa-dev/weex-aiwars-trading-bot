from datetime import datetime
from typing import Any, Dict


class AILogStub:
    def __init__(self, logger):
        self.logger = logger
        self.pending_logs = {}

    async def log_decision(
        self, order_request: Dict[str, Any], decision_data: Dict[str, Any]
    ):
        """AI log stub - only WARNING logs"""
        self.logger.warning(
            f"[AI_LOG_STUB] Decision: {decision_data.get('signal', 'N/A')} | "
            f"Symbol: {order_request.get('symbol', 'N/A')} | "
            f"Quantity: {order_request.get('quantity', 'N/A')} | "
            f"Model: Alpha-Pipeline-v1.0.0a | "
            f"Stage: {decision_data.get('stage', 'Decision Making')}"
        )

        # Store for possible future linking
        client_order_id = order_request.get("client_order_id", "")
        if client_order_id:
            self.pending_logs[client_order_id] = {
                "decision_data": decision_data,
                "timestamp": datetime.now(),
            }

    async def link_order_to_log(self, client_order_id: str, order_id: str):
        """Stub for linking order ID"""
        if client_order_id in self.pending_logs:
            self.logger.warning(
                f"[AI_LOG_STUB] Linked: {client_order_id} -> {order_id}"
            )
            del self.pending_logs[client_order_id]

    def get_pending_count(self) -> int:
        """Get count of pending AI logs"""
        return len(self.pending_logs)
