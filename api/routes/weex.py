import asyncio
from fastapi import APIRouter, Depends
from api.auth import verify_token
from api.dependencies import USER_COINS
from weex_client import WeexAsyncClient, config

settings = config.load_config()
router = APIRouter(tags=["Weex"])


@router.get("/balance", dependencies=[Depends(verify_token)])
async def get_balance():
    client = WeexAsyncClient(config=settings)
    return await client.get_account_balance()


@router.get("/positions", dependencies=[Depends(verify_token)])
async def get_positions():
    client = WeexAsyncClient(config=settings)
    return await client.get_all_positions()


@router.get("/tickers", dependencies=[Depends(verify_token)])
async def get_tickers():
    client = WeexAsyncClient(config=settings)
    tasks = [asyncio.create_task(client.get_ticker(coin)) for coin in USER_COINS]
    return dict(zip(USER_COINS, await asyncio.gather(*tasks)))
