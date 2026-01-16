from api.routes.auth import router as auth_router
from api.routes.status import router as status_router
from api.routes.weex import router as weex_router
from api.routes.system import router as system_router

__all__ = [
    "auth_router",
    "status_router",
    "weex_router",
    "system_router",
]
