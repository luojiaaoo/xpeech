from .server import app

import uvicorn


def run(host="0.0.0.0", port=7878):
    uvicorn.run(
        app=app,
        host=host,
        port=port,
        reload=False,
        reload_includes=None,
        reload_excludes=None,
        workers=1,
        access_log=True,
        lifespan="on",
    )
