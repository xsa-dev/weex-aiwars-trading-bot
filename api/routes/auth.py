from fastapi import APIRouter
from api.models import TokenResponse

router = APIRouter(tags=["Auth"])


@router.post("/auth/token", response_model=TokenResponse)
async def login():
    return {"access_token": "secret_token_12345", "token_type": "bearer"}
