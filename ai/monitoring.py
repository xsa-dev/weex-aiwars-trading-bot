from prometheus_client import Counter, Histogram, Gauge, start_http_server
import threading


trading_cycles_total = Counter(
    "trading_cycles_total", "Total trading cycles", ["status"]
)
decisions_total = Counter(
    "decisions_total", "Trading decisions", ["coin", "action", "signal"]
)
executions_total = Counter("executions_total", "Trade executions", ["coin", "success"])
api_calls_total = Counter("api_calls_total", "API calls", ["api", "status"])
trading_cycle_duration = Histogram(
    "trading_cycle_duration_seconds",
    "Cycle duration",
    buckets=[5, 10, 30, 60, 120, 180, 300, 600],
)
api_call_duration = Histogram(
    "api_call_duration_seconds",
    "API call duration",
    ["api"],
    buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0],
)
active_positions = Gauge("active_positions", "Active positions")
portfolio_value = Gauge("portfolio_value_usdt", "Portfolio value in USDT")
scheduler_status = Gauge("scheduler_status", "Scheduler status", ["status"])
last_trade_timestamp = Gauge("last_trade_timestamp", "Last trade timestamp")
rate_limiter_delay = Gauge("rate_limiter_delay_ms", "Rate limiter delay in ms")


class MetricsCollector:
    def __init__(self, port: int = 9090):
        self._port = port
        self._server_started = False
        self._lock = threading.Lock()

    def start_server(self) -> None:
        with self._lock:
            if not self._server_started:
                start_http_server(self._port)
                self._server_started = True

    def trading_cycle_completed(
        self, coins_analyzed: int, decisions: int, duration_ms: float, success: bool
    ) -> None:
        status = "success" if success else "failed"
        trading_cycles_total.labels(status=status).inc()
        trading_cycle_duration.observe(duration_ms / 1000)

    def scheduler_health_check(
        self,
        is_healthy: bool,
        last_run_ago_seconds: float | None,
        error_count: int,
        success_count: int,
    ) -> None:
        status = "running" if is_healthy else "error"
        scheduler_status.labels(status=status).set(1 if is_healthy else 0)


metrics = MetricsCollector()


def start_metrics_server(port: int = 9090) -> None:
    metrics.start_server(port)
