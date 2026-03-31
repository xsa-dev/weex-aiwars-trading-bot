import logging

import uvicorn
from dotenv import load_dotenv

from api.app import app

load_dotenv(".env")


app.debug = True

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    print("🚀 Trading Bot API starting...")
    print("📖 Swagger UI: http://localhost:8888/docs")
    uvicorn.run("main:app", host="0.0.0.0", port=8888, reload=True, reload_delay=30)
