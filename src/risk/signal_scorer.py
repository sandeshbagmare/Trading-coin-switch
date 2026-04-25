"""!
@file signal_scorer.py
@brief ML-based signal scoring using gradient boosting on technical indicators.
Trains on historical signal outcomes. Persists model to data/models/.
"""
import json
import os
import pickle
from pathlib import Path
from typing import Dict, Optional
from src.data.technical_indicators import compute_features
from src.risk import rl_agent

MODEL_DIR = Path(__file__).parent.parent.parent / "data" / "models"
MODEL_PATH = MODEL_DIR / "signal_scorer.pkl"
HISTORY_PATH = MODEL_DIR / "training_data.json"


def _ensure_dirs():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)


def _load_history() -> list:
    if HISTORY_PATH.exists():
        with open(HISTORY_PATH) as f:
            return json.load(f)
    return []


def _save_history(data: list):
    _ensure_dirs()
    with open(HISTORY_PATH, "w") as f:
        json.dump(data, f)


def record_outcome(features: dict, signal: str, confidence: float, pnl: float):
    """!@brief Record a trade outcome for future model training."""
    history = _load_history()
    history.append({
        "features": features, "signal": signal,
        "confidence": confidence, "pnl": pnl,
    })
    if len(history) > 5000:
        history = history[-5000:]
    _save_history(history)
    
    # Also update the RL Q-table logic simultaneously!
    rl_agent.update_q_value(features, signal, pnl)


def train_model() -> bool:
    """!@brief Train gradient boosting classifier on accumulated trade data."""
    history = _load_history()
    if len(history) < 10:
        return False
    try:
        from sklearn.ensemble import GradientBoostingClassifier
        feature_keys = ["rsi", "ema_cross", "macd", "bb_position", "bb_width",
                        "atr_pct", "vol_ratio", "pct_1h", "pct_4h", "pct_24h"]
        X, y = [], []
        for entry in history:
            feat = entry["features"]
            if not all(k in feat for k in feature_keys):
                continue
            X.append([feat[k] for k in feature_keys])
            y.append(1 if entry["pnl"] > 0 else 0)
        if len(X) < 10:
            return False
        model = GradientBoostingClassifier(
            n_estimators=50, max_depth=3, learning_rate=0.1,
        )
        model.fit(X, y)
        _ensure_dirs()
        with open(MODEL_PATH, "wb") as f:
            pickle.dump({"model": model, "features": feature_keys}, f)
        print(f"[ML] Model trained on {len(X)} samples. Saved to {MODEL_PATH}")
        return True
    except ImportError:
        print("[ML] scikit-learn not installed, skipping model training.")
        return False


def predict_score(pair: str) -> Optional[float]:
    """!
    @brief Use trained ML model to score a trade signal (0.0-1.0 win probability).
    Falls back to rule-based scoring if no model exists.
    """
    features = compute_features(pair)
    if not features:
        return None

    # Try ML model first
    if MODEL_PATH.exists():
        try:
            with open(MODEL_PATH, "rb") as f:
                data = pickle.load(f)
            model = data["model"]
            keys = data["features"]
            x = [[features.get(k, 0) for k in keys]]
            proba = model.predict_proba(x)[0]
            return float(proba[1]) if len(proba) > 1 else float(proba[0])
        except Exception as e:
            print(f"[ML] Prediction failed: {e}")

    # Fallback: rule-based score from technical indicators
    score = 0.5
    rsi = features.get("rsi", 50)
    if 30 < rsi < 70:
        score += 0.05  # Neutral zone is safer
    if features.get("ema_cross", 0) == 1.0:
        score += 0.1   # Bullish EMA crossover
    if features.get("vol_ratio", 1) > 1.3:
        score += 0.05  # High volume confirms moves
    if features.get("bb_position", 0.5) < 0.2:
        score += 0.1   # Near lower BB = potential bounce
    elif features.get("bb_position", 0.5) > 0.8:
        score -= 0.05  # Near upper BB = potential reversal
    if features.get("macd", 0) > 0:
        score += 0.05
    return max(0.0, min(1.0, score))


def get_features_summary(pair: str) -> str:
    """!@brief Returns a human-readable summary of technical indicators for UI display."""
    f = compute_features(pair)
    if not f:
        return "Indicators unavailable"
    parts = [
        f"RSI:{f['rsi']:.0f}",
        f"EMA:{'↑' if f['ema_cross'] else '↓'}",
        f"MACD:{f['macd']:+.2f}",
        f"BB:{f['bb_position']:.0%}",
        f"Vol:{f['vol_ratio']:.1f}x",
        f"1h:{f['pct_1h']:+.2f}%",
    ]
    return " | ".join(parts)
