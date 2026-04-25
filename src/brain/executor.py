"""!
@file executor.py
@brief Execution agent — produces final trade order from upstream agents.
"""
from src.brain.prompts import EXECUTION_SYSTEM, EXECUTION_USER
from src.brain.llm_caller import call_ensemble
from src.config.lm_config import get_role_models


def run(pair: str, analysis: dict, strategy: dict, risk: dict, price: float) -> dict:
    """!
    @brief Build final execution order combining all upstream analysis.
    @param pair Trading pair string.
    @param analysis Analyst output.
    @param strategy Strategist output.
    @param risk Risk manager output.
    @param price Current market price.
    @return Final executable trade dict, or {execute: False} if rejected.
    """
    if not risk.get("approved", False):
        return {
            "execute": False,
            "pair": pair,
            "reason": risk.get("reason", "Rejected by risk manager"),
        }

    action   = strategy.get("action", "BUY")
    size     = risk["final_size_inr"]
    leverage = risk["adjusted_leverage"]
    sl_price = risk.get("stop_loss_price", 0)
    tp_price = risk.get("take_profit_price", 0)

    user = EXECUTION_USER.format(
        pair=pair,
        action=action,
        price=price,
        size=size,
        leverage=leverage,
        instrument=strategy.get("instrument", "SPOT"),
        sl_price=sl_price,
        tp_price=tp_price,
        risk_inr=risk.get("risk_amount_inr", 0),
        reward_inr=risk.get("reward_amount_inr", 0),
        rr=risk.get("actual_rr", 0),
        strategy_name=strategy.get("strategy_name", ""),
        rationale=strategy.get("rationale", ""),
    )
    models = get_role_models("execution")
    result = call_ensemble(models, EXECUTION_SYSTEM, user, vote_key="action", max_tokens=350)

    if not result or not result.get("execute"):
        # Safe fallback using validated risk values
        return {
            "execute": True,
            "pair": pair,
            "action": action,
            "instrument": strategy.get("instrument", "SPOT"),
            "size_inr": size,
            "leverage": leverage,
            "entry_price": price,
            "stop_loss_price": sl_price,
            "take_profit_price": tp_price,
            "strategy_name": strategy.get("strategy_name", "Auto"),
            "reasoning": strategy.get("rationale", "Auto-execution"),
        }
    # Ensure critical fields are not overridden with bad values
    result["stop_loss_price"]  = result.get("stop_loss_price", sl_price) or sl_price
    result["take_profit_price"] = result.get("take_profit_price", tp_price) or tp_price
    result["size_inr"]   = float(result.get("size_inr", size))
    result["leverage"]   = int(result.get("leverage", leverage))
    return result
