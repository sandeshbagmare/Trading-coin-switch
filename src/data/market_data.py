"""!
@file market_data.py
@brief Real-time price fetching via CoinGecko (free, no key needed).
"""
import requests
from datetime import datetime, timezone
from typing import Dict, Optional

PAIR_MAP = {
    "BTC-INR":  "bitcoin",
    "ETH-INR":  "ethereum",
    "SOL-INR":  "solana",
    "BNB-INR":  "binancecoin",
    "DOGE-INR": "dogecoin",
}


def get_live_prices() -> Dict[str, dict]:
    """!
    @brief Fetch live INR prices for all pairs from CoinGecko.
    @return Dict: pair → {price, change_24h, volume_24h, updated_at}
    """
    ids = ",".join(PAIR_MAP.values())
    url = (
        "https://api.coingecko.com/api/v3/simple/price"
        f"?ids={ids}&vs_currencies=inr"
        "&include_24hr_change=true&include_24hr_vol=true"
    )
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        result = {}
        now = datetime.now(timezone.utc).isoformat()
        for pair, cg_id in PAIR_MAP.items():
            if cg_id in data:
                item = data[cg_id]
                result[pair] = {
                    "price":      item.get("inr", 0),
                    "change_24h": item.get("inr_24h_change", 0),
                    "volume_24h": item.get("inr_24h_vol", 0),
                    "updated_at": now,
                }
        return result
    except Exception as e:
        print(f"[MarketData] Fetch failed: {e}")
        return {}


def get_price(pair: str, prices: Dict[str, dict]) -> Optional[float]:
    """!@brief Extract price for a single pair from a price dict."""
    return prices.get(pair, {}).get("price")


def get_klines(pair: str, interval: str = "1h", limit: int = 12) -> str:
    """!
    @brief Fetch recent candlestick (OHLCV) data from Binance for pattern matching.
    Converts pair like 'BTC-INR' to 'BTCUSDT'.
    @return A summarized string of the latest candlesticks and simple pattern analysis.
    """
    base = pair.split("-")[0]
    symbol = f"{base}USDT"
    url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&limit={limit}"
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        
        # [OpenTime, Open, High, Low, Close, Volume, ...]
        summary = []
        for i, k in enumerate(data):
            o, h, l, c = float(k[1]), float(k[2]), float(k[3]), float(k[4])
            color = "Bullish" if c > o else "Bearish"
            body = abs(c - o)
            total = h - l
            
            # Simple pattern logic
            pattern = ""
            if total > 0:
                if body / total < 0.1: pattern = "Doji"
                elif color == "Bullish" and (c - l) / total > 0.8: pattern = "Hammer"
                elif color == "Bearish" and (h - c) / total > 0.8: pattern = "Shooting Star"
            
            summary.append(f"T-{limit-i}: {color} (O:{o:.2f} H:{h:.2f} L:{l:.2f} C:{c:.2f}) {pattern}")
            
        return " | ".join(summary)
    except Exception as e:
        return f"OHLC data unavailable ({symbol}): {e}"
