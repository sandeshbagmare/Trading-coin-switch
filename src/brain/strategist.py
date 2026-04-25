"""!
@file strategist.py
@brief Strategist agent — selects strategy and computes SL/TP plan using RAG context.
"""
from typing import Dict
from src.brain.prompts import STRATEGIST_SYSTEM, STRATEGIST_USER
from src.brain.llm_caller import call_ensemble
from src.config.lm_config import get_role_models
from src.config import settings

_kb_cache = {"embedder": None, "vstore": None}

_FALLBACK = {"action": "HOLD", "strategy_name": "No Strategy", "instrument": "SPOT",
             "leverage": 1, "entry_type": "MARKET", "position_size_pct": 0.03,
             "stop_loss_pct": 0.05, "take_profit_pct": 0.10, "risk_reward_ratio": 2.0,
             "rationale": "LLM unavailable — holding"}


def _get_kb_context(query: str) -> str:
    """!@brief Lazy-load kb and retrieve relevant context for strategy selection."""
    try:
        if _kb_cache["embedder"] is None:
            from src.knowledge.knowledge_base import ChunkEmbedder, VectorStore
            _kb_cache["embedder"] = ChunkEmbedder(settings.EMBEDDING_MODEL)
            _kb_cache["vstore"]   = VectorStore(settings.CHROMA_DB_DIR)
        emb = _kb_cache["embedder"].model.encode([query], normalize_embeddings=True)[0].tolist()
        res = _kb_cache["vstore"].search(emb, n_results=2)
        chunks = res.get("documents", [[]])[0]
        return " | ".join(c[:200] for c in chunks) if chunks else "No context."
    except Exception as e:
        return f"KB error: {e}"


def run(pair: str, analysis: dict, price: float) -> dict:
    """!
    @brief Run the strategist on analyst output to build a full trade plan.
    @param pair Trading pair string.
    @param analysis Output from analyst.run().
    @param price Current market price.
    @return Strategy dict with full SL/TP plan.
    """
    signal = analysis.get("signal", "HOLD")
    trend  = analysis.get("trend", "SIDEWAYS")
    vol    = analysis.get("volatility", "MEDIUM")

    kb_context = _get_kb_context(f"{trend} {vol} crypto {signal} strategy")
    from src.config.lm_config import trading_cfg
    tcfg = trading_cfg()
    allow_lev = str(tcfg.get("allow_leverage", True)).lower()
    max_lev = int(tcfg.get("max_leverage", 4))

    user = STRATEGIST_USER.format(
        pair=pair,
        signal=signal,
        confidence=analysis.get("confidence", 0.5),
        trend=trend,
        volatility=vol,
        price=price,
        reasoning=analysis.get("reasoning", ""),
        kb_context=kb_context,
        allow_leverage=allow_lev,
        max_leverage=max_lev,
    )
    models = get_role_models("strategist")
    result = call_ensemble(models, STRATEGIST_SYSTEM, user, vote_key="action", max_tokens=350)

    if not result:
        return _FALLBACK

    # Clamp leverage and position
    result["leverage"] = max(1, min(max_lev if allow_lev == "true" else 1, int(result.get("leverage", 1))))
    result["position_size_pct"] = max(0.05, min(0.10, float(result.get("position_size_pct", 0.05))))
    return result
