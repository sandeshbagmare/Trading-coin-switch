"""!@file signals.py @brief Agent signal log and prices routes."""
from fastapi import APIRouter
from src.brain import orchestrator
from src.data.market_data import PAIR_MAP

router = APIRouter()

@router.get("/prices")
async def get_prices():
    return orchestrator.get_last_prices()

@router.get("/pairs")
async def get_pairs():
    return list(PAIR_MAP.keys())

@router.get("/agent-log")
async def get_agent_log(limit: int = 20):
    return orchestrator.get_agent_log()[:limit]

@router.get("/status")
async def get_status():
    from src.config.lm_config import trading_cfg
    return {"status": "running", "mode": trading_cfg().get("mode", "paper")}
