import asyncio
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from loguru import logger
app = FastAPI()
@app.websocket("/ws")
async def ws_cmd(websocket: WebSocket):
    await websocket.accept()
    

    async def stream_output():
        try:
            while True:
                line = await process.stdout.readline()
                if not line:
                    break
                await websocket.send_text(line.decode().rstrip())
            await websocket.send_text("[process done]")
        except WebSocketDisconnect:
            logger.info("websocket disconnected")
    async def receive_input():
        try:
            while True:
                msg = await websocket.receive_text()
                print("客户端发来:", msg)
        except WebSocketDisconnect:
            pass
    await asyncio.gather(stream_output(), receive_input())

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
