"""!@file control.py @brief Trade control routes: close position, pause/resume."""
from fastapi import APIRouter
from src.execution.portfolio import Portfolio
from src.brain import orchestrator

router = APIRouter()

@router.post("/close/{position_id}")
async def close_position(position_id: str):
    p = Portfolio()
    prices = orchestrator.get_last_prices() or {}
    pos_map = {pos.id: pos for pos in p.open_positions()}
    pos = pos_map.get(position_id)
    if not pos:
        return {"error": "Position not found"}
    price = prices.get(pos.pair, {}).get("price", pos.entry_price)
    result = p.close_position(position_id, price, "MANUAL")
    if result:
        orchestrator._push("trade_closed", {"trade": result})
    return {"success": bool(result), "pnl": result["pnl"] if result else 0}

@router.post("/reset")
async def reset_portfolio():
    """!@brief Reset portfolio (paper mode only)."""
    import sqlite3
    from src.config import settings
    from src.config.lm_config import trading_cfg
    tcfg = trading_cfg()
    if tcfg.get("mode") != "paper":
        return {"error": "Can only reset in paper mode"}
    with sqlite3.connect(str(settings.TRADES_DB)) as c:
        c.execute("DELETE FROM positions")
        c.execute("DELETE FROM trades")
        c.execute("UPDATE state SET value=? WHERE key='balance'",
                  (str(tcfg.get("starting_balance", 2000.0)),))
    return {"success": True, "message": "Portfolio reset to paper defaults"}
