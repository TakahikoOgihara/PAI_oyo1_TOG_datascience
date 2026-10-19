# -*- coding: utf-8 -*-
"""ARM_BCRL006 operator web app -- runs LOCALLY on the operator PC.

Browser (http://localhost:8000)  <-ws->  this app  <-ws->  Jetson runtime
                                                          (policy_runtime.py :8765)

The app is a thin relay: it serves the single-page UI and pipes the browser's
WebSocket to the Jetson's telemetry/command endpoint, so the browser shows
the live wrist camera + joint state and the buttons (Start / Pause / Jog /
Gripper / Home / E-STOP) reach the runtime. Keeping the relay on the
operator PC means the browser needs no direct route to the robot LAN.

Run:
  pip install -r requirements.txt
  JETSON_WS=ws://192.168.10.2:8765 uvicorn app:app --port 8000
  # then open http://localhost:8000
"""

from __future__ import annotations
import asyncio
import os

import websockets
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

JETSON_WS = os.environ.get("JETSON_WS", "ws://192.168.10.2:8765")
STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")

app = FastAPI(title="ARM_BCRL006 operator console")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
async def index():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


@app.websocket("/ws")
async def ws_relay(browser: WebSocket):
    await browser.accept()
    try:
        async with websockets.connect(JETSON_WS, max_size=None) as jetson:
            async def to_jetson():
                while True:
                    msg = await browser.receive_text()
                    await jetson.send(msg)

            async def to_browser():
                async for msg in jetson:
                    await browser.send_text(msg)

            done, pending = await asyncio.wait(
                [asyncio.create_task(to_jetson()), asyncio.create_task(to_browser())],
                return_when=asyncio.FIRST_EXCEPTION)
            for t in pending:
                t.cancel()
    except (WebSocketDisconnect, OSError, websockets.WebSocketException) as e:
        try:
            await browser.send_json({"type": "error",
                                     "error": f"jetson link down ({e}); check JETSON_WS={JETSON_WS}"})
        except Exception:
            pass
    finally:
        try:
            await browser.close()
        except Exception:
            pass
