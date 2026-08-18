from .routes.chat import router as chat_router
from .routes.files import router as files_router
from .server import app

app.include_router(files_router)
app.include_router(chat_router)
