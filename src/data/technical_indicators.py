"""!
@file technical_indicators.py
@brief Compute technical indicators from Binance OHLCV data for ML feature engineering.
"""
import requests
from typing import List, Dict, Optional

BINANCE_MAP = {
    "BTC-INR": "BTCUSDT", "ETH-INR": "ETHUSDT", "SOL-INR": "SOLUSDT",
    "BNB-INR": "BNBUSDT", "DOGE-INR": "DOGEUSDT",
}


def _fetch_klines(pair: str, interval: str = "1h", limit: int = 50) -> List[dict]:
    """!@brief Fetch raw kline data from Binance."""
    symbol = BINANCE_MAP.get(pair, pair.split("-")[0] + "USDT")
    url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&limit={limit}"
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        return [{"o": float(k[1]), "h": float(k[2]), "l": float(k[3]),
                 "c": float(k[4]), "v": float(k[5])} for k in resp.json()]
    except Exception:
        return []


def compute_features(pair: str) -> Optional[Dict[str, float]]:
    """!
    @brief Compute all technical features for a pair.
    @return Dict of feature_name -> value, or None on failure.
    """
    candles = _fetch_klines(pair, "1h", 50)
    if len(candles) < 26:
        return None

    closes = [c["c"] for c in candles]
    highs  = [c["h"] for c in candles]
    lows   = [c["l"] for c in candles]
    vols   = [c["v"] for c in candles]

    # RSI (14-period)
    gains, losses = [], []
    for i in range(1, min(15, len(closes))):
        d = closes[-i] - closes[-i - 1]
        gains.append(max(d, 0))
        losses.append(max(-d, 0))
    avg_gain = sum(gains) / 14 if gains else 0.001
    avg_loss = sum(losses) / 14 if losses else 0.001
    rs = avg_gain / avg_loss if avg_loss > 0 else 100
    rsi = 100 - (100 / (1 + rs))

    # EMAs
    def ema(data, period):
        k = 2 / (period + 1)
        val = data[0]
        for p in data[1:]:
            val = p * k + val * (1 - k)
        return val

    ema_9  = ema(closes, 9)
    ema_21 = ema(closes, 21)
    ema_50 = ema(closes[-50:], 50) if len(closes) >= 50 else ema(closes, len(closes))

    # MACD
    ema_12 = ema(closes, 12)
    ema_26 = ema(closes, 26)
    macd_line = ema_12 - ema_26

    # Bollinger Bands (20-period)
    sma_20 = sum(closes[-20:]) / 20
    std_20 = (sum((c - sma_20) ** 2 for c in closes[-20:]) / 20) ** 0.5
    bb_upper = sma_20 + 2 * std_20
    bb_lower = sma_20 - 2 * std_20
    bb_width = (bb_upper - bb_lower) / sma_20 if sma_20 else 0
    bb_position = (closes[-1] - bb_lower) / (bb_upper - bb_lower) if (bb_upper - bb_lower) > 0 else 0.5

    # ATR (14-period)
    trs = []
    for i in range(-14, 0):
        tr = max(highs[i] - lows[i], abs(highs[i] - closes[i - 1]), abs(lows[i] - closes[i - 1]))
        trs.append(tr)
    atr = sum(trs) / len(trs) if trs else 0

    # Volume trend
    vol_sma = sum(vols[-10:]) / 10 if len(vols) >= 10 else sum(vols) / max(len(vols), 1)
    vol_ratio = vols[-1] / vol_sma if vol_sma > 0 else 1.0

    # Price momentum
    pct_1h = (closes[-1] - closes[-2]) / closes[-2] * 100 if len(closes) >= 2 else 0
    pct_4h = (closes[-1] - closes[-4]) / closes[-4] * 100 if len(closes) >= 4 else 0
    pct_24h = (closes[-1] - closes[-24]) / closes[-24] * 100 if len(closes) >= 24 else 0

    return {
        "rsi": round(rsi, 2),
        "ema_9": round(ema_9, 4), "ema_21": round(ema_21, 4), "ema_50": round(ema_50, 4),
        "ema_cross": 1.0 if ema_9 > ema_21 else 0.0,
        "macd": round(macd_line, 4),
        "bb_width": round(bb_width, 4), "bb_position": round(bb_position, 4),
        "atr": round(atr, 4), "atr_pct": round(atr / closes[-1] * 100, 4) if closes[-1] else 0,
        "vol_ratio": round(vol_ratio, 2),
        "pct_1h": round(pct_1h, 3), "pct_4h": round(pct_4h, 3), "pct_24h": round(pct_24h, 3),
        "price": closes[-1],
    }
