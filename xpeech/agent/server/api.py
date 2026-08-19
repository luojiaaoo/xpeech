from fastapi import Depends

from .auth import require_jwt
from .auth import router as auth_router
from .routes.chat import router as chat_router
from .routes.files import preview_router
from .routes.files import router as files_router
from .routes.statistics import router as statistics_router
from .server import app

app.include_router(auth_router)
app.include_router(files_router, dependencies=[Depends(require_jwt)])
app.include_router(chat_router, dependencies=[Depends(require_jwt)])
app.include_router(statistics_router, dependencies=[Depends(require_jwt)])
app.include_router(preview_router)
