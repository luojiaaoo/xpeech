from fastapi import FastAPI
from .middleware import ContextASGIMiddleware
from ...utils.logging import configure_logging

configure_logging()

app = FastAPI()

app.add_middleware(ContextASGIMiddleware)
