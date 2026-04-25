"""!
@file analyst.py
@brief Market Analyst agent — generates directional signal using DeepSeek ensemble.
"""
from typing import Dict
from src.brain.prompts import ANALYST_SYSTEM, ANALYST_USER
from src.brain.llm_caller import call_ensemble
from src.config.lm_config import get_role_models

_FALLBACK = {"signal": "HOLD", "confidence": 0.5, "trend": "SIDEWAYS",
             "volatility": "MEDIUM", "market_condition": "RANGING",
             "key_factors": ["LLM unavailable"], "reasoning": "Fallback HOLD"}


def run(pair: str, prices: Dict[str, dict]) -> dict:
    """!
    @brief Run the analyst agent on a given pair.
    @param pair Trading pair e.g. 'BTC-INR'.
    @param prices Full price dict from market_data.
    @return Analyst signal dict.
    """
    data = prices.get(pair, {})
    if not data:
        return _FALLBACK

    btc_chg = prices.get("BTC-INR", {}).get("change_24h", 0)
    all_changes = {p: d.get("change_24h", 0) for p, d in prices.items()}
    bullish_pairs = sum(1 for c in all_changes.values() if c > 0)
    context = f"{bullish_pairs}/{len(all_changes)} pairs bullish today"

    from src.data.market_data import get_klines
    klines = get_klines(pair)
    
    user = ANALYST_USER.format(
        pair=pair,
        price=data.get("price", 0),
        change_24h=data.get("change_24h", 0),
        volume=data.get("volume_24h", 0),
        btc_change=btc_chg,
        context=context,
        klines=klines,
    )
    models = get_role_models("analyst")
    result = call_ensemble(models, ANALYST_SYSTEM, user, vote_key="signal", max_tokens=300)
    return result if result else _FALLBACK
