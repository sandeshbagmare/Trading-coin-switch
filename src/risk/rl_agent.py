"""!
@file rl_agent.py
@brief Reinforcement Learning (Q-Learning proxy) agent for trade weighting.
This module simulates dynamic Q-value adaptation based on P&L rewards,
complementing the static ML gradient boosting model.
"""
import json
import os
from pathlib import Path
from typing import Dict

MODEL_DIR = Path(__file__).parent.parent.parent / "data" / "models"
Q_TABLE_PATH = MODEL_DIR / "rl_q_table.json"

# State buckets for RL
def _get_state_key(features: Dict[str, float]) -> str:
    """!@brief Discretize continuous ML features into a simple RL state space."""
    rsi = features.get("rsi", 50)
    rsi_state = "OVERSOLD" if rsi < 35 else "OVERBOUGHT" if rsi > 70 else "NEUTRAL"
    trend = "UP" if features.get("ema_cross", 0) else "DOWN"
    vol = "HIGH" if features.get("vol_ratio", 1) > 1.2 else "NORMAL"
    return f"{rsi_state}_{trend}_{vol}"

def _load_q_table() -> Dict[str, Dict[str, float]]:
    if Q_TABLE_PATH.exists():
        try:
            with open(Q_TABLE_PATH, "r") as f:
                return json.load(f)
        except Exception:
            pass
    return {}

def _save_q_table(q_table: Dict[str, Dict[str, float]]):
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    with open(Q_TABLE_PATH, "w") as f:
        json.dump(q_table, f)

def get_rl_action_score(features: Dict[str, float]) -> float:
    """!
    @brief Returns the RL agent's confidence for a LONG (BUY) action in the current state.
    Value between 0.0 and 1.0. 0.5 is neutral.
    """
    if not features:
        return 0.5
    state = _get_state_key(features)
    q_table = _load_q_table()
    
    # Init state if not present
    if state not in q_table:
        q_table[state] = {"BUY": 0.0, "HOLD": 0.0}
        _save_q_table(q_table)

    q_buy = q_table[state]["BUY"]
    q_hold = q_table[state]["HOLD"]
    
    # Normalize to 0-1 confidence
    total = abs(q_buy) + abs(q_hold)
    if total == 0:
        return 0.5
    
    # Softmax-style approximation
    import math
    try:
        exp_buy = math.exp(q_buy)
        exp_hold = math.exp(q_hold)
        return exp_buy / (exp_buy + exp_hold)
    except OverflowError:
        return 1.0 if q_buy > q_hold else 0.0

def update_q_value(features: Dict[str, float], action: str, reward_pnl: float):
    """!
    @brief Reward function. Updates the Q-table based on Trade PnL.
    Alpha (learning rate) = 0.1, Gamma (discount) = 0.9 (simplified 1-step).
    """
    if not features or action not in ("BUY", "SELL", "HOLD"):
        return
        
    # Map SELL to negative BUY for simplicity in this 1D action space
    rl_action = "BUY" if action == "BUY" else "HOLD" if action == "HOLD" else "BUY"
    
    # Normalize reward
    reward = min(max(reward_pnl / 100.0, -1.0), 1.0) 
    if action == "SELL":
        reward = -reward # A successful short means BUY was bad
        
    state = _get_state_key(features)
    q_table = _load_q_table()
    
    if state not in q_table:
        q_table[state] = {"BUY": 0.0, "HOLD": 0.0}
        
    current_q = q_table[state][rl_action]
    
    # Bellman equation update (simplified, no next state transition logic since we are evaluating pointwise trades)
    alpha = 0.1
    new_q = current_q + alpha * (reward - current_q)
    
    q_table[state][rl_action] = round(new_q, 4)
    _save_q_table(q_table)
    print(f"  [RL Agent] Updated Q-Table ({state}) -> {rl_action}: {new_q:+.2f} (Reward: {reward:+.2f})")
