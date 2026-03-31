from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes import auth_router, status_router, weex_router, system_router
from ai.trading import start_trading, stop_trading

app = FastAPI(
    title="Trading Bot API",
    docs_url="/docs",
    redoc_url="/redoc",
    on_startup=[start_trading],
    on_shutdown=[stop_trading],
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth_router, prefix="/auth")
app.include_router(status_router)
app.include_router(weex_router)
app.include_router(system_router)


@app.post("/close-all-positions")
async def close_all_positions():
    """Закрывает все открытые позиции."""
    from ai.position_manager import close_all_positions

    results = await close_all_positions()
    return {"results": results}
