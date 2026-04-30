from contextlib import asynccontextmanager
from datetime import datetime

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from loguru import logger
from rich.console import Console
from rich.panel import Panel
from rich.text import Text


@asynccontextmanager
async def span(app: FastAPI):
    label = app.title
    start_ts = datetime.now()
    console = Console()
    console.print(
        Panel(
            Text.from_markup(f'\n[bold cyan] ⏱ Start[/]  [dim]{start_ts:%Y-%m-%d %H:%M:%S}[/]\n'),
            title=f'[bold cyan]{label}[/]',
            subtitle='[dim]Powered by pydantic-ai[/]',
            border_style='cyan',
            expand=False,
        )
    )
    yield


app = FastAPI(title='Xpeech Agent', lifespan=span)
