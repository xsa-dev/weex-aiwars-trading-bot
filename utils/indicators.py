import pandas as pd
from typing import Any
from utils.config_loader import load_config


def candles_to_df(candles: list[list[str]]) -> pd.DataFrame:
    if not candles:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

    data = []
    for c in candles:
        if isinstance(c, list) and len(c) >= 6:
            data.append(
                {
                    "open": float(c[1]),
                    "high": float(c[2]),
                    "low": float(c[3]),
                    "close": float(c[4]),
                    "volume": float(c[5]),
                }
            )

    return pd.DataFrame(data)


def calculate_rsi(prices: pd.Series, period: int = 14) -> float:
    delta = prices.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return round(float(rsi.iloc[-1]), 1)


def calculate_sma(prices: pd.Series, period: int) -> pd.Series:
    return prices.rolling(window=period).mean()


def calculate_ema(prices: pd.Series, period: int) -> pd.Series:
    return prices.ewm(span=period, adjust=False).mean()


def calculate_supertrend(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    length: int = 10,
    multiplier: float = 3.0,
) -> tuple[str, float]:
    atr = calculate_atr(high, low, close, length)

    basis = (high + low) / 2
    upper = basis + multiplier * atr
    lower = basis - multiplier * atr

    trend = ["up"] * len(close)
    for i in range(1, len(close)):
        if close.iloc[i] > upper.iloc[i - 1]:
            trend[i] = "up"
        elif close.iloc[i] < lower.iloc[i - 1]:
            trend[i] = "down"
        else:
            trend[i] = trend[i - 1]
            if trend[i] == "up" and lower.iloc[i] < lower.iloc[i - 1]:
                lower.iloc[i] = lower.iloc[i - 1]
            if trend[i] == "down" and upper.iloc[i] > upper.iloc[i - 1]:
                upper.iloc[i] = upper.iloc[i - 1]

    return trend[-1], round(
        float(lower.iloc[-1] if trend[-1] == "up" else upper.iloc[-1]), 2
    )


def calculate_adx(
    high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14
) -> float:
    plus_dm = high.diff()
    minus_dm = -low.diff()

    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm < 0] = 0

    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    atr = tr.rolling(window=period, min_periods=period).mean()

    plus_di = 100 * (plus_dm.rolling(window=period, min_periods=period).mean() / atr)
    minus_di = 100 * (minus_dm.rolling(window=period, min_periods=period).mean() / atr)

    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = dx.rolling(window=period, min_periods=period).mean()

    return round(float(adx.iloc[-1]), 1)


def calculate_macd(
    close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9
) -> tuple[float, float, float]:
    ema_fast = calculate_ema(close, fast)
    ema_slow = calculate_ema(close, slow)

    macd_line = ema_fast - ema_slow
    signal_line = calculate_ema(macd_line, signal)
    histogram = macd_line - signal_line

    return (
        round(float(macd_line.iloc[-1]), 2),
        round(float(signal_line.iloc[-1]), 2),
        round(float(histogram.iloc[-1]), 2),
    )


def calculate_momentum(close: pd.Series, period: int = 10) -> float:
    return round(
        float(close.iloc[-1] - close.iloc[-period - 1] if len(close) > period else 0), 2
    )


def calculate_atr(
    high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14
) -> pd.Series:
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=period, min_periods=period).mean()
    return atr


def calculate_bollinger_bands(
    close: pd.Series, length: int = 20, std: float = 2
) -> tuple[float, float, float]:
    sma = close.rolling(window=length).mean()
    std_dev = close.rolling(window=length).std()

    upper = sma + std_dev * std
    lower = sma - std_dev * std

    return (
        round(float(upper.iloc[-1]), 2),
        round(float(sma.iloc[-1]), 2),
        round(float(lower.iloc[-1]), 2),
    )


def calculate_stoch_rsi(close: pd.Series, period: int = 14) -> tuple[float, float]:
    """Calculate Stochastic RSI."""
    calculate_rsi(close, period)

    rsi_min = close.rolling(window=period).min()
    rsi_max = close.rolling(window=period).max()

    stoch_rsi = (close - rsi_min) / (rsi_max - rsi_min)

    k = stoch_rsi.iloc[-1] * 100
    # k is a scalar (float), not a Series - need to keep history
    k * 100  # k already contains the last value
    
    # Calculate d as 3-period SMA of k values
    k_series = stoch_rsi * 100  # Full Series for rolling
    d = k_series.rolling(window=3).mean().iloc[-1] if len(k_series) >= 3 else k

    return round(float(k), 1), round(float(d), 1)


def calculate_vwma(close: pd.Series, volume: pd.Series, period: int = 20) -> float | None:
    """Calculate Volume Weighted Moving Average."""
    try:
        vwma_series = (close * volume).rolling(window=period).sum() / volume.rolling(window=period).sum()
        if vwma_series.empty:
            return None
        vwma = vwma_series.iloc[-1]
        if pd.isna(vwma) or (isinstance(vwma, float) and vwma != vwma):  # Check for NaN
            return None
        return round(float(vwma), 2)
    except (ZeroDivisionError, ValueError):
        return None


def calculate_indicators(df: pd.DataFrame, tf: str) -> dict[str, Any]:
    cfg = load_config().get("indicators", {})
    result = {"timeframe": tf, "calculated_at": pd.Timestamp.now().isoformat()}

    if df.empty or len(df) < 20:
        result["error"] = "Insufficient data"
        return result

    try:
        close = df["close"]
        high = df["high"]
        low = df["low"]
        volume = df["volume"]

        rsi_cfg = cfg.get("rsi", {})
        result["rsi"] = calculate_rsi(close, rsi_cfg.get("period", 14))
        result["rsi_signal"] = interpret_rsi(result["rsi"], rsi_cfg)

        st_cfg = cfg.get("supertrend", {})
        st_dir, st_val = calculate_supertrend(
            high,
            low,
            close,
            length=st_cfg.get("length", 10),
            multiplier=st_cfg.get("multiplier", 3.0),
        )
        result["supertrend_direction"] = st_dir
        result["supertrend_value"] = st_val

        adx_cfg = cfg.get("adx", {})
        result["adx"] = calculate_adx(high, low, close, adx_cfg.get("period", 14))
        result["adx_strength"] = interpret_adx(
            result["adx"], adx_cfg.get("threshold", 25)
        )

        macd_cfg = cfg.get("macd", {})
        macd, macd_sig, macd_hist = calculate_macd(
            close,
            fast=macd_cfg.get("fast", 12),
            slow=macd_cfg.get("slow", 26),
            signal=macd_cfg.get("signal", 9),
        )
        result["macd"] = macd
        result["macd_signal"] = macd_sig
        result["macd_histogram"] = macd_hist
        result["macd_trend"] = "bullish" if macd_hist > 0 else "bearish"

        mom_cfg = cfg.get("momentum", {})
        result["momentum"] = calculate_momentum(close, mom_cfg.get("period", 10))

        atr_cfg = cfg.get("atr", {})
        result["atr"] = round(
            float(calculate_atr(high, low, close, atr_cfg.get("period", 14)).iloc[-1]),
            2,
        )

        bb_cfg = cfg.get("bollinger", {})
        bb_upper, bb_middle, bb_lower = calculate_bollinger_bands(
            close, length=bb_cfg.get("length", 20), std=bb_cfg.get("std", 2)
        )
        result["bb_upper"] = bb_upper
        result["bb_middle"] = bb_middle
        result["bb_lower"] = bb_lower
        bb_range = bb_upper - bb_lower
        if bb_range > 0:
            result["bb_position"] = round(
                (close.iloc[-1] - bb_lower) / bb_range * 100, 1
            )
        else:
            result["bb_position"] = 50.0

        stoch_cfg = cfg.get("stoch_rsi", {})
        stoch_k, stoch_d = calculate_stoch_rsi(close, stoch_cfg.get("period", 14))
        result["stoch_rsi_k"] = stoch_k
        result["stoch_rsi_d"] = stoch_d

        vwma_cfg = cfg.get("vwma", {})
        result["vwma"] = calculate_vwma(close, volume, vwma_cfg.get("period", 20))

    except Exception as e:
        result["error"] = str(e)
    
    # ВСЕГДА рассчитываем тренд, даже если есть ошибки в индикаторах
    if "close" in df.columns:
        result["overall_trend"] = calculate_overall_trend(result, df["close"].iloc[-1])

    return result


def interpret_rsi(rsi: float, cfg: dict) -> str:
    if rsi >= cfg.get("overbought", 70):
        return "overbought"
    elif rsi <= cfg.get("oversold", 30):
        return "oversold"
    else:
        return "neutral"


def interpret_adx(adx: float, threshold: float) -> str:
    if adx >= 50:
        return "very_strong"
    elif adx >= threshold:
        return "strong"
    elif adx >= 20:
        return "moderate"
    else:
        return "weak"


def calculate_overall_trend(indicators: dict, price: float) -> str:
    score = 0

    if indicators.get("rsi", 50) > 55:
        score += 1
    elif indicators.get("rsi", 50) < 45:
        score -= 1

    if indicators.get("supertrend_direction") == "up":
        score += 2
    else:
        score -= 2

    if indicators.get("macd_histogram", 0) > 0:
        score += 1
    else:
        score -= 1

    if indicators.get("adx", 0) >= 25:
        if indicators.get("supertrend_direction") == "up":
            score += 1
        else:
            score -= 1

    vwma = indicators.get("vwma")
    if vwma is not None and price > vwma:
        score += 0.5
    elif vwma is not None:
        score -= 0.5
    # Если VWMA=None, score не меняется

    if score >= 3:
        return "BULLISH"
    elif score <= -2:
        return "BEARISH"
    else:
        return "NEUTRAL"
