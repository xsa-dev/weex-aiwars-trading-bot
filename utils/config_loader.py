import yaml
from pathlib import Path
from typing import Any

_config: dict[str, Any] | None = None
CONFIG_PATH = Path(__file__).parent.parent / "config.yaml"


def load_config() -> dict[str, Any]:
    global _config
    if _config is None:
        with open(CONFIG_PATH, "r") as f:
            _config = yaml.safe_load(f) or {}
    return _config


def get(key: str, default=None) -> Any:
    cfg = load_config()
    return cfg.get(key, default)


def reload():
    global _config
    _config = None
    return load_config()
