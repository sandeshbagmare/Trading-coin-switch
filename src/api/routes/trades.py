"""!@file trades.py @brief Trade history REST routes."""
from fastapi import APIRouter
from src.execution.portfolio import Portfolio

router = APIRouter()

@router.get("/trades")
async def get_trades(limit: int = 50):
    return Portfolio().trade_history(limit=limit)
