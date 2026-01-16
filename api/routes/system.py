from fastapi import APIRouter, Depends, HTTPException
from api.auth import verify_token
from api.dependencies import bot_status, USER_COINS
from api.models import RestartResponse, LogEntry
from ai.trading import start_trading, stop_trading
from utils.logger import logs, add_log, get_order_book_history

router = APIRouter(tags=["System"])


@router.get(
    "/logs", response_model=list[LogEntry], dependencies=[Depends(verify_token)]
)
def get_logs():
    return [
        {
            "timestamp": log.split("]")[0].strip("["),
            "message": log.split("]", 1)[1].strip(),
        }
        for log in list(logs)
    ]


@router.get("/orderbook/{coin}", dependencies=[Depends(verify_token)])
def get_orderbook_history(coin: str, limit: int = 50):
    """Get order book history for a specific coin."""
    if coin not in USER_COINS:
        raise HTTPException(status_code=404, detail=f"Coin {coin} not found")
    return get_order_book_history(coin, limit)


@router.get("/orderbook", dependencies=[Depends(verify_token)])
def get_all_orderbook_history(limit: int = 10):
    """Get order book history for all coins."""
    return {coin: get_order_book_history(coin, limit) for coin in USER_COINS}


@router.get("/config", dependencies=[Depends(verify_token)])
def get_config():
    return {
        "coins": USER_COINS,
        "api_timeout": 30,
        "log_limit": 100,
    }


@router.post(
    "/restart", response_model=RestartResponse, dependencies=[Depends(verify_token)]
)
async def restart():
    add_log("Restart requested via API")
    await stop_trading()
    await start_trading()
    bot_status["last_action"] = "restarted"
    return {"status": "ok", "message": "Restart signal sent"}
