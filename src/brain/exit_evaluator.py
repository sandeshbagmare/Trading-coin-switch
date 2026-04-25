"""!
@file exit_evaluator.py
@brief Early Exit Evaluator — monitors open trades and decides if early market exit is needed based on new pattern logic.
"""
from src.brain.prompts import EXIT_SYSTEM, EXIT_USER
from src.brain.llm_caller import call_ensemble
from src.config.lm_config import get_role_models
from src.data.market_data import get_klines

_FALLBACK = {"decision": "HOLD", "reason": "Evaluator unavailable, falling back to static SL/TP."}

def run(position, current_price: float) -> dict:
    """!
    @brief Evaluates if an active position should be closed immediately.
    @param position Portfolio Position object.
    @param current_price Live current price in INR.
    @return Decision dict containing 'EXIT' or 'HOLD'.
    """
    try:
        klines = get_klines(position.pair)
        pnl_inr = position.unrealized_pnl(current_price)
        pnl_pct = position.pnl_pct(current_price)
        
        user = EXIT_USER.format(
            pair=position.pair,
            side=position.side,
            leverage=position.leverage,
            entry_price=position.entry_price,
            current_price=current_price,
            pnl_pct=pnl_pct,
            pnl_inr=pnl_inr,
            sl=position.stop_loss_price,
            tp=position.take_profit_price,
            strategy_name=position.strategy_name,
            reasoning=position.reasoning,
            klines=klines,
        )
        
        models = get_role_models("strategist")  # Reuse the strategist ensemble for exit management
        result = call_ensemble(models, EXIT_SYSTEM, user, vote_key="decision", max_tokens=200)
        
        return result if result else _FALLBACK
    except Exception as e:
        print(f"  [Exit] Error evaluating {position.pair}: {e}")
        return _FALLBACK
