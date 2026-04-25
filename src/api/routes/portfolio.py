"""!@file portfolio.py @brief Portfolio REST routes."""
from fastapi import APIRouter
from src.execution.portfolio import Portfolio
from src.brain import orchestrator

router = APIRouter()

@router.get("/portfolio")
async def get_portfolio():
    p = Portfolio()
    prices = orchestrator.get_last_prices() or {}
    pm = {pair: d.get("price", 0) for pair, d in prices.items()}
    return p.summary(pm)

@router.get("/positions")
async def get_positions():
    p = Portfolio()
    prices = orchestrator.get_last_prices() or {}
    result = []
    for pos in p.open_positions():
        cur = prices.get(pos.pair, {}).get("price", pos.entry_price)
        result.append({
            "id": pos.id, "pair": pos.pair, "side": pos.side,
            "instrument": pos.instrument, "leverage": pos.leverage,
            "size": pos.quantity_inr, "entry_price": pos.entry_price,
            "current_price": cur,
            "unrealized_pnl": round(pos.unrealized_pnl(cur), 2),
            "pnl_pct": round(pos.pnl_pct(cur), 2),
            "sl": pos.stop_loss_price, "tp": pos.take_profit_price,
            "strategy_name": pos.strategy_name,
            "reasoning": pos.reasoning, "opened_at": pos.opened_at,
        })
    return result
