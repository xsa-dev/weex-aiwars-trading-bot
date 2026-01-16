from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException
from api.auth import verify_token
from api.dependencies import bot_status, USER_COINS
from api.models import StatusResponse, HealthResponse, CoinIndicators, IndicatorData

router = APIRouter(tags=["Status"])


@router.get("/health", response_model=HealthResponse)
def health():
    uptime = None
    if bot_status["started_at"]:
        started = datetime.fromisoformat(bot_status["started_at"])
        uptime = (datetime.now() - started).total_seconds()
    return {"status": "healthy", "uptime_seconds": uptime}


@router.get(
    "/status", response_model=StatusResponse, dependencies=[Depends(verify_token)]
)
def get_status():
    return {
        "running": bot_status["running"],
        "started_at": bot_status["started_at"],
        "last_action": bot_status["last_action"],
        "coins": USER_COINS,
        "log_count": 0,
    }


@router.get(
    "/api/v1/indicators/{coin}",
    response_model=CoinIndicators,
    dependencies=[Depends(verify_token)],
)
async def get_indicators(coin: str):
    if "indicators" not in bot_status or coin not in bot_status.get("indicators", {}):
        raise HTTPException(status_code=404, detail=f"No indicators for {coin}")

    raw = bot_status["indicators"][coin]
    timeframes = {}
    for tf, data in raw.items():
        timeframes[tf] = IndicatorData(
            timeframe=tf,
            calculated_at=data.get("calculated_at", ""),
            rsi=data.get("rsi"),
            rsi_signal=data.get("rsi_signal"),
            supertrend_direction=data.get("supertrend_direction"),
            supertrend_value=data.get("supertrend_value"),
            adx=data.get("adx"),
            adx_strength=data.get("adx_strength"),
            macd=data.get("macd"),
            macd_signal=data.get("macd_signal"),
            macd_histogram=data.get("macd_histogram"),
            macd_trend=data.get("macd_trend"),
            momentum=data.get("momentum"),
            atr=data.get("atr"),
            bb_upper=data.get("bb_upper"),
            bb_middle=data.get("bb_middle"),
            bb_lower=data.get("bb_lower"),
            bb_position=data.get("bb_position"),
            stoch_rsi_k=data.get("stoch_rsi_k"),
            stoch_rsi_d=data.get("stoch_rsi_d"),
            vwma=data.get("vwma"),
            overall_trend=data.get("overall_trend"),
            error=data.get("error"),
        )

    return CoinIndicators(coin=coin, timeframes=timeframes)


@router.get(
    "/api/v1/indicators",
    response_model=dict[str, CoinIndicators],
    dependencies=[Depends(verify_token)],
)
async def get_all_indicators():
    if "indicators" not in bot_status:
        raise HTTPException(status_code=503, detail="Indicators not available")

    result = {}
    for coin, raw in bot_status["indicators"].items():
        timeframes = {}
        for tf, data in raw.items():
            timeframes[tf] = IndicatorData(
                timeframe=tf,
                calculated_at=data.get("calculated_at", ""),
                rsi=data.get("rsi"),
                rsi_signal=data.get("rsi_signal"),
                supertrend_direction=data.get("supertrend_direction"),
                supertrend_value=data.get("supertrend_value"),
                adx=data.get("adx"),
                adx_strength=data.get("adx_strength"),
                macd=data.get("macd"),
                macd_signal=data.get("macd_signal"),
                macd_histogram=data.get("macd_histogram"),
                macd_trend=data.get("macd_trend"),
                momentum=data.get("momentum"),
                atr=data.get("atr"),
                bb_upper=data.get("bb_upper"),
                bb_middle=data.get("bb_middle"),
                bb_lower=data.get("bb_lower"),
                bb_position=data.get("bb_position"),
                stoch_rsi_k=data.get("stoch_rsi_k"),
                stoch_rsi_d=data.get("stoch_rsi_d"),
                vwma=data.get("vwma"),
                overall_trend=data.get("overall_trend"),
                error=data.get("error"),
            )
        result[coin] = CoinIndicators(coin=coin, timeframes=timeframes)

    return result
