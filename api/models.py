from pydantic import BaseModel
from typing import Optional


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


class RestartResponse(BaseModel):
    status: str
    message: str


class LogEntry(BaseModel):
    timestamp: str
    message: str


class StatusResponse(BaseModel):
    running: bool
    started_at: str | None
    last_action: str | None
    coins: list[str]
    log_count: int


class HealthResponse(BaseModel):
    status: str
    uptime_seconds: float | None


class IndicatorData(BaseModel):
    timeframe: str
    calculated_at: str
    rsi: Optional[float] = None
    rsi_signal: Optional[str] = None
    supertrend_direction: Optional[str] = None
    supertrend_value: Optional[float] = None
    adx: Optional[float] = None
    adx_strength: Optional[str] = None
    macd: Optional[float] = None
    macd_signal: Optional[float] = None
    macd_histogram: Optional[float] = None
    macd_trend: Optional[str] = None
    momentum: Optional[float] = None
    atr: Optional[float] = None
    bb_upper: Optional[float] = None
    bb_middle: Optional[float] = None
    bb_lower: Optional[float] = None
    bb_position: Optional[float] = None
    stoch_rsi_k: Optional[float] = None
    stoch_rsi_d: Optional[float] = None
    vwma: Optional[float] = None
    overall_trend: Optional[str] = None
    error: Optional[str] = None


class CoinIndicators(BaseModel):
    coin: str
    timeframes: dict[str, IndicatorData]
