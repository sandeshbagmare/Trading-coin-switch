"""!
@file risk_manager.py
@brief Risk manager agent — validates trades, computes exact SL/TP INR prices.
"""
from src.brain.prompts import RISK_SYSTEM, RISK_USER
from src.brain.llm_caller import call_ensemble
from src.config.lm_config import get_role_models, trading_cfg

_REJECT = {"approved": False, "final_size_inr": 0, "adjusted_leverage": 1,
           "stop_loss_price": 0, "take_profit_price": 0,
           "risk_amount_inr": 0, "reward_amount_inr": 0, "actual_rr": 0,
           "reason": "Risk manager unavailable"}


def _compute_sl_tp(action: str, price: float, sl_pct: float, tp_pct: float, leverage: int):
    """!@brief Compute SL/TP prices from percentage distances."""
    sl = sl_pct / max(leverage, 1)
    tp = tp_pct / max(leverage, 1)
    if action == "BUY":
        return price * (1 - sl), price * (1 + tp)
    else:
        return price * (1 + sl), price * (1 - tp)


def run(pair: str, strategy: dict, price: float, balance: float, n_open: int) -> dict:
    """!
    @brief Run the risk manager to validate and size the proposed trade.
    @param pair Trading pair.
    @param strategy Output from strategist.run().
    @param price Current market price.
    @param balance Available balance in INR.
    @param n_open Count of currently open positions.
    @return Risk assessment dict with final SL/TP prices in INR.
    """
    tcfg = trading_cfg()
    max_pos  = tcfg.get("max_open_positions", 3)
    min_trade = tcfg.get("min_trade_inr", 100.0)
    max_pct  = tcfg.get("max_position_pct", 0.05)

    action    = strategy.get("action", "HOLD")
    leverage  = strategy.get("leverage", 4)
    size_pct  = strategy.get("position_size_pct", 0.03)
    sl_pct    = strategy.get("stop_loss_pct", 0.05)
    tp_pct    = strategy.get("take_profit_pct", 0.10)
    proposed  = round(balance * size_pct, 2)

    # Hard guards before calling LLM
    if action == "HOLD":
        return {**_REJECT, "reason": "HOLD — no trade needed"}
    if n_open >= max_pos:
        return {**_REJECT, "reason": f"Max {max_pos} positions open"}
    if balance < min_trade:
        return {**_REJECT, "reason": f"Balance {balance:.2f} < min {min_trade}"}

    user = RISK_USER.format(
        pair=pair, action=action, instrument=strategy.get("instrument", "SPOT"),
        leverage=leverage, price=price,
        proposed_size=proposed, size_pct=size_pct * 100,
        sl_pct=sl_pct * 100, tp_pct=tp_pct * 100,
        balance=balance, n_open=n_open, max_positions=max_pos,
    )
    models = get_role_models("risk")
    result = call_ensemble(models, RISK_SYSTEM, user, vote_key="approved", max_tokens=300)

    if not result:
        return _REJECT

    # Clamp final size
    result["final_size_inr"] = max(min_trade, min(
        float(result.get("final_size_inr", proposed)),
        balance * max_pct
    ))
    result["adjusted_leverage"] = max(1, min(8, int(result.get("adjusted_leverage", leverage))))

    # Ensure SL/TP are present (compute from pct if LLM missed them)
    if not result.get("stop_loss_price") or not result.get("take_profit_price"):
        sl_p, tp_p = _compute_sl_tp(action, price, sl_pct, tp_pct,
                                    result["adjusted_leverage"])
        result.setdefault("stop_loss_price", round(sl_p, 4))
        result.setdefault("take_profit_price", round(tp_p, 4))

    return result
