import logging
from logging.handlers import RotatingFileHandler
from datetime import datetime
from collections import deque
from typing import Deque, Dict, List, Any


def setup_logger():
    """Setup logger with file and console output."""
    logger = logging.getLogger("trading")
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(
        "[%(asctime)s] %(levelname)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )

    file_handler = RotatingFileHandler(
        "trading.log", maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8"
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger


logger = setup_logger()
logs: Deque[str] = deque(maxlen=500)
order_book_history: Dict[str, List[Dict[str, Any]]] = {}


def add_log(message: str, level: str = "INFO"):
    """Add log to memory deque, file, and console."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = f"[{timestamp}] {message}"

    logs.append(log_entry)

    log_level = getattr(logging, level.upper(), logging.INFO)
    logger.log(log_level, message)


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
