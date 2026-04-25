"""!
@file run.py
@brief Entry point for CryptoAgent backend server.
"""
import sys
import uvicorn
from src.config.lm_config import trading_cfg, server_cfg

# Force UTF-8 encoding for Windows terminals to avoid crash on symbols
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except AttributeError:
        pass

if __name__ == "__main__":
    tc = trading_cfg()
    sc = server_cfg()
    mode = tc.get("mode", "paper")
    host = sc.get("host", "127.0.0.1")
    port = sc.get("port", 8000)
    print("=" * 60)
    print("  CryptoAgent AI Trading System")
    print(f"  Mode: {'PAPER TRADING' if mode=='paper' else 'LIVE TRADING'}")
    print(f"  Starting balance: INR {tc.get('starting_balance', 2000.0)}")
    print(f"  Dashboard: http://{host}:{port}")
    print("=" * 60)
    uvicorn.run(
        "src.api.main:app",
        host=host, port=port,
        reload=False, log_level="info",
    )
