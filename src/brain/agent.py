"""!
@file agent.py
@brief Orchestrates the 4-agent pipeline with ML scoring and live step broadcasting.
"""
import time
from datetime import datetime, timezone
from typing import Dict, List, Callable, Optional
from src.brain import analyst, strategist, risk_manager, executor
from src.risk.signal_scorer import predict_score, get_features_summary, record_outcome
from src.risk import rl_agent
from src.data.technical_indicators import compute_features


def run_cycle(pair: str, prices: Dict[str, dict],
              balance: float, open_positions: List,
              step_cb: Optional[Callable] = None) -> dict:
    """!
    @brief Runs Analyst→ML Score→Strategist→Risk→Execution for one pair.
    @param step_cb Callback(pair, step_text) for live UI step broadcasting.
    """
    t0    = time.time()
    price = prices.get(pair, {}).get("price", 0)
    n_open = len(open_positions)
    _step = lambda msg: step_cb(pair, msg) if step_cb else None

    _step("📡 Fetching live market data...")
    print(f"  [Agent] {pair} @ ₹{price:,.0f}")

    # Step 1: ML & RL Signal Score
    _step("📊 Computing technical indicators & running ML/RL models...")
    actual_features = compute_features(pair)
    ml_score = predict_score(pair)
    rl_score = rl_agent.get_rl_action_score(actual_features) if actual_features else 0.5
    indicators = get_features_summary(pair)
    _step(f"🧮 ML: {ml_score:.0%} | RL: {rl_score:.0%} | {indicators}" if ml_score else "🧮 ML indicators computed")

    # Step 2: LLM Analyst
    _step("🤖 Analyst agent analyzing market conditions...")
    a = analyst.run(pair, prices)
    conf = a.get('confidence', 0)
    print(f"    Analyst: {a.get('signal')} conf={conf:.0%} trend={a.get('trend')}")
    _step(f"📋 Analyst: {a.get('signal')} {conf:.0%} — {a.get('trend')}")

    # Step 3: Confidence gate
    from src.config.lm_config import trading_cfg
    min_conf = float(trading_cfg().get("min_confidence_threshold", 0.80))

    # Combine LLM Analyst, ML Score, and RL Score
    combined_conf = conf
    if ml_score is not None:
        combined_conf = conf * 0.5 + ml_score * 0.3 + rl_score * 0.2  # Tri-blend architecture

    if a.get('signal') == "HOLD" or combined_conf < min_conf:
        _step(f"⏸ Confidence {combined_conf:.0%} < {min_conf:.0%} threshold — skipping")
        print(f"    -> Skipping: combined conf {combined_conf:.0%} < {min_conf:.0%}")
        return {
            "pair": pair, "timestamp": datetime.now(timezone.utc).isoformat(),
            "price": price, "analysis": a,
            "ml_score": ml_score, "indicators": indicators,
            "strategy": {}, "risk": {},
            "decision": {"execute": False, "action": "HOLD"},
            "elapsed_s": round(time.time() - t0, 1),
        }

    # Step 4: Strategist
    _step("🎯 Strategist designing trade setup...")
    s = strategist.run(pair, a, price)
    print(f"    Strategist: {s.get('action')} {s.get('strategy_name')} lev={s.get('leverage')}x")
    _step(f"📐 Strategy: {s.get('strategy_name')} — {s.get('action')} {s.get('leverage')}x")

    # Step 5: Risk Manager
    _step("🛡️ Risk manager validating R:R ratio...")
    r = risk_manager.run(pair, s, price, balance, n_open)
    approved = r.get("approved", False)
    print(f"    Risk: {'APPROVED' if approved else 'REJECTED'} — {r.get('reason','')[:50]}")
    _step(f"{'✅ Risk APPROVED' if approved else '❌ Risk REJECTED'} — R:R {r.get('actual_rr', 0):.1f}")

    # Step 6: Executor
    _step("⚡ Executor finalizing trade order...")
    d = executor.run(pair, a, s, r, price)
    elapsed = round(time.time() - t0, 1)
    print(f"    Execute: {d.get('execute')} — {elapsed}s")
    _step(f"{'🚀 TRADE EXECUTED' if d.get('execute') else '⏹ No execution'} — {elapsed}s total")

    return {
        "pair":       pair,
        "timestamp":  datetime.now(timezone.utc).isoformat(),
        "price":      price,
        "analysis":   a,
        "ml_score":   ml_score,
        "rl_score":   rl_score,
        "indicators": indicators,
        "strategy":   s,
        "risk":       r,
        "decision":   d,
        "elapsed_s":  elapsed,
    }
