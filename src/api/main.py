"""!
@file main.py
@brief FastAPI app with WebSocket broadcasting and lifespan management.
"""
import asyncio
import json
from contextlib import asynccontextmanager
from typing import Set
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pathlib import Path

from src.brain import orchestrator
from src.execution.portfolio import Portfolio
from src.data.market_data import PAIR_MAP
from src.config.lm_config import trading_cfg, server_cfg

# ── WebSocket Hub ─────────────────────────────────────────────────────────────
_clients: Set[WebSocket] = set()
_queue:   asyncio.Queue  = asyncio.Queue(maxsize=300)


def _enqueue(event: str, data):
    try: _queue.put_nowait({"event": event, "data": data})
    except asyncio.QueueFull: pass


async def _broadcaster():
    global _clients
    while True:
        msg  = await _queue.get()
        dead = set()
        for ws in list(_clients):
            try: await ws.send_text(json.dumps(msg, default=str))
            except Exception: dead.add(ws)
        _clients -= dead


@asynccontextmanager
async def lifespan(app: FastAPI):
    orchestrator.set_broadcast(_enqueue)
    orchestrator.start()
    asyncio.create_task(_broadcaster())
    yield
    orchestrator.stop()


app = FastAPI(title="CryptoAgent", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])

# ── Include route modules ─────────────────────────────────────────────────────
from src.api.routes import portfolio as portfolio_routes
from src.api.routes import trades as trade_routes
from src.api.routes import signals as signal_routes
from src.api.routes import control as control_routes

app.include_router(portfolio_routes.router, prefix="/api")
app.include_router(trade_routes.router,     prefix="/api")
app.include_router(signal_routes.router,    prefix="/api")
app.include_router(control_routes.router,   prefix="/api")


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    global _clients
    await ws.accept()
    _clients.add(ws)
    try:
        p = Portfolio()
        prices = orchestrator.get_last_prices() or {}
        pm = {pair: d.get("price", 0) for pair, d in prices.items()}
        await ws.send_text(json.dumps({
            "event": "snapshot",
            "data": {
                "portfolio": p.summary(pm),
                "prices": pm,
                "trades": p.trade_history(20),
                "mode": trading_cfg().get("mode", "paper"),
            }
        }, default=str))
        while True: await ws.receive_text()
    except WebSocketDisconnect:
        _clients.discard(ws)


# ── Serve UI ──────────────────────────────────────────────────────────────────
_ui = Path(__file__).parent.parent.parent / "ui"
if _ui.exists():
    app.mount("/", StaticFiles(directory=str(_ui), html=True), name="ui")
