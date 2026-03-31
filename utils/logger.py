import inspect
import json
import logging
from collections import deque
from datetime import datetime
from logging.handlers import RotatingFileHandler
from typing import Any, Deque, Dict, List, Optional


def _get_caller_logger_name() -> str:
    """Get logger name from caller's module using inspect."""
    frame = inspect.currentframe()
    if frame is None:
        return "unknown"

    # Walk up the stack looking for the first frame that's not from utils.logger
    # Skip: _get_caller_logger_name -> add_log -> caller
    i = 0
    while frame is not None:
        frame = frame.f_back
        i += 1
        if frame is None:
            return "unknown"

        module = frame.f_globals.get("__name__", "")

        # Skip utils.logger internals (this function and add_log)
        if module == "utils.logger":
            continue

        # Skip logging module internals and builtins, but allow __main__
        if (
            module
            and module != "__main__"
            and not module.startswith("_")
            and "logging" not in module
        ):
            return module.split(".")[-1] if "." in module else module
        elif module == "__main__":
            return "main"

    return "unknown"


class JSONFormatter(logging.Formatter):
    """Custom JSON formatter for structured logging."""

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        logger_name = getattr(record, "logger", "trading")

        log_data: dict[str, Any] = {
            "timestamp": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z",
            "level": record.levelname,
            "logger": logger_name,
            "message": record.getMessage(),
        }

        # Add extra fields if present
        data = getattr(record, "data", None)
        if data and isinstance(data, dict):
            log_data["data"] = data

        coin = getattr(record, "coin", None)
        if coin:
            log_data["coin"] = coin

        error_type = getattr(record, "error_type", None)
        if error_type:
            log_data["error_type"] = error_type

        return json.dumps(log_data, ensure_ascii=False)


def setup_logger():
    """Setup logger with JSON file output and readable console output."""
    logger = logging.getLogger("trading")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    # JSON formatter for file
    json_formatter = JSONFormatter()

    # Readable formatter for console
    console_formatter = logging.Formatter(
        "[%(asctime)s] %(levelname)s [%(logger)s]: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # File handler with JSON output
    file_handler = RotatingFileHandler(
        "trading.log", maxBytes=5 * 1024 * 1024, backupCount=100, encoding="utf-8"
    )
    file_handler.setFormatter(json_formatter)
    logger.addHandler(file_handler)

    # Console handler with readable output
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    return logger


logger = setup_logger()
logs: Deque[Dict[str, Any]] = deque(maxlen=500)
order_book_history: Dict[str, List[Dict[str, Any]]] = {}


def add_log(
    message: str,
    level: str = "INFO",
    data: Optional[Dict[str, Any]] = None,
    coin: Optional[str] = None,
    error_type: Optional[str] = None,
    logger_name: Optional[str] = None,
):
    """Add structured log to memory, file, and console.

    Args:
        message: Log message
        level: Log level (INFO, WARNING, ERROR, DEBUG)
        data: Optional structured data dict for JSON logging
        coin: Optional coin symbol for filtering
        error_type: Optional error type for classification
        logger_name: Optional logger name (auto-detected from caller if not provided)
    """
    # Auto-detect logger name from caller if not provided
    if logger_name is None:
        logger_name = _get_caller_logger_name()

    timestamp = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")

    # Store structured log in memory
    log_entry = {
        "timestamp": timestamp,
        "level": level.upper(),
        "logger": logger_name,
        "message": message,
        "data": data,
        "coin": coin,
        "error_type": error_type,
    }
    logs.append(log_entry)

    log_level = getattr(logging, level.upper(), logging.INFO)
    logger.log(log_level, message, extra={"logger": logger_name})


def add_order_book_snapshot(coin: str, data: Dict[str, Any], max_entries: int = 100):
    """Add order book snapshot to history."""
    if coin not in order_book_history:
        order_book_history[coin] = []
    order_book_history[coin].append(data)
    if len(order_book_history[coin]) > max_entries:
        order_book_history[coin].pop(0)


def get_order_book_history(coin: str, limit: int = 50) -> List[Dict[str, Any]]:
    """Get order book history for a coin."""
    return order_book_history.get(coin, [])[-limit:]


def format_order_book_log(
    coin: str,
    price: str,
    bid_vol: float,
    ask_vol: float,
    spread: str,
    imbalance: float,
    trend: str,
) -> str:
    """Format order book data into a single log line."""
    return f"{coin}: ${price} | spread={spread} | bids={bid_vol:.2f} vol | asks={ask_vol:.2f} vol |imb={imbalance:.2f}{trend}"


def get_logs(
    coin: Optional[str] = None, level: Optional[str] = None, limit: int = 100
) -> List[Dict[str, Any]]:
    """Get filtered logs.

    Args:
        coin: Filter by coin symbol
        level: Filter by log level
        limit: Maximum number of logs to return

    Returns:
        List of filtered log entries
    """
    filtered = list(logs)

    if coin:
        filtered = [entry for entry in filtered if entry.get("coin") == coin]

    if level:
        filtered = [entry for entry in filtered if entry.get("level") == level.upper()]

    return filtered[-limit:]


def export_logs_json(coin: Optional[str] = None, limit: int = 500) -> str:
    """Export logs as JSON string.

    Args:
        coin: Optional coin filter
        limit: Maximum number of logs

    Returns:
        JSON string of logs
    """
    filtered = get_logs(coin=coin, limit=limit)
    return json.dumps(filtered, ensure_ascii=False, indent=2)
