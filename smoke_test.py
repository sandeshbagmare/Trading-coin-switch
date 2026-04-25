"""Quick smoke test for all V2 modules."""
from src.brain import analyst, strategist, risk_manager, executor, exit_evaluator
from src.brain import llm_caller, orchestrator, prompts, agent
from src.data import market_data
from src.execution import portfolio
from src.config import lm_config

print("All modules import OK")
print(f"Rate limiter lock: {llm_caller._limiter_lock}")
print(f"Klines func: {market_data.get_klines}")
print(f"Exit evaluator func: {exit_evaluator.run}")

tcfg = lm_config.trading_cfg()
print(f"Min confidence: {tcfg.get('min_confidence_threshold')}")
print(f"Allow leverage: {tcfg.get('allow_leverage')}")
print(f"Max leverage: {tcfg.get('max_leverage')}")

# Quick klines test
kl = market_data.get_klines("BTC-INR", limit=3)
print(f"Klines sample: {kl[:120]}...")

print("\nSMOKE TEST PASSED")
