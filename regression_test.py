"""!
@file regression_test.py  
@brief Runs 12 simulated agent pipeline scenarios to verify end-to-end system integrity.
"""
import sys
sys.path.insert(0, ".")

from src.brain import analyst, strategist, risk_manager, executor

SCENARIOS = [
    ("BTC-INR",  {"price": 8500000, "change_24h":  5.2, "volume_24h": 12e9}, "Strong Bull"),
    ("ETH-INR",  {"price": 220000,  "change_24h":  3.8, "volume_24h": 3e9},  "Moderate Bull"),
    ("SOL-INR",  {"price": 12500,   "change_24h": -4.1, "volume_24h": 1e9},  "Moderate Bear"),
    ("BNB-INR",  {"price": 48000,   "change_24h": -7.5, "volume_24h": 2e9},  "Strong Bear"),
    ("DOGE-INR", {"price": 17,      "change_24h":  0.3, "volume_24h": 5e8},  "Sideways"),
    ("BTC-INR",  {"price": 9200000, "change_24h": 12.1, "volume_24h": 20e9}, "Breakout Bull"),
    ("ETH-INR",  {"price": 195000,  "change_24h": -9.2, "volume_24h": 4e9},  "Crash Bear"),
    ("SOL-INR",  {"price": 13800,   "change_24h":  1.2, "volume_24h": 8e8},  "Weak Bull"),
    ("BNB-INR",  {"price": 52000,   "change_24h": -0.5, "volume_24h": 1.5e9},"Ranging"),
    ("DOGE-INR", {"price": 20,      "change_24h":  8.0, "volume_24h": 9e8},  "Meme Spike"),
    ("BTC-INR",  {"price": 7800000, "change_24h": -15,  "volume_24h": 25e9}, "Major Selloff"),
    ("ETH-INR",  {"price": 250000,  "change_24h":  2.0, "volume_24h": 2.5e9},"Steady Grind"),
]

BALANCE = 2000.0
BTC_REF = {"price": 8500000, "change_24h": 1.0}


def validate(result: dict, label: str) -> bool:
    """!@brief Assert all required fields are present and valid."""
    a, s, r, d = result["a"], result["s"], result["r"], result["d"]
    assert a.get("signal") in ("BUY","SELL","HOLD"), f"{label}: bad signal"
    assert 0 <= float(a.get("confidence", 0)) <= 1,   f"{label}: bad confidence"
    assert s.get("action") in ("BUY","SELL","HOLD"),   f"{label}: bad action"
    assert isinstance(r.get("approved"), bool),        f"{label}: approved not bool"
    if r["approved"]:
        assert float(r.get("stop_loss_price",  0)) > 0, f"{label}: SL=0"
        assert float(r.get("take_profit_price",0)) > 0, f"{label}: TP=0"
        assert float(r.get("final_size_inr",   0)) >= 100, f"{label}: size<100"
    assert "execute" in d, f"{label}: missing execute"
    return True


def run():
    print("\n" + "="*65)
    print("  CryptoAgent — 12-Scenario Regression Test")
    print("="*65)
    passed, results = 0, []

    for i, (pair, pdata, label) in enumerate(SCENARIOS, 1):
        prices = {pair: pdata, "BTC-INR": BTC_REF}
        price  = pdata["price"]
        print(f"\n[{i:02d}] {label} — {pair} @ {price:,.0f}", flush=True)
        try:
            a = analyst.run(pair, prices)
            print(f"  A: {a['signal']} conf={float(a.get('confidence',0)):.0%} "
                  f"trend={a.get('trend','?')}", flush=True)

            s = strategist.run(pair, a, price)
            print(f"  S: {s.get('action')} '{s.get('strategy_name','')}' "
                  f"lev={s.get('leverage')} SL={float(s.get('stop_loss_pct',0))*100:.1f}% "
                  f"TP={float(s.get('take_profit_pct',0))*100:.1f}%", flush=True)

            r = risk_manager.run(pair, s, price, BALANCE, 0)
            print(f"  R: {'APPROVED' if r['approved'] else 'REJECTED'} "
                  f"SL={r.get('stop_loss_price',0):,.2f} "
                  f"TP={r.get('take_profit_price',0):,.2f} "
                  f"R:R={r.get('actual_rr',0):.2f}", flush=True)

            d = executor.run(pair, a, s, r, price)
            print(f"  E: execute={d['execute']}", flush=True)

            validate({"a":a,"s":s,"r":r,"d":d}, label)
            passed += 1
            print(f"  [PASS]", flush=True)
            results.append({"label":label,"signal":a["signal"],"approved":r["approved"],"pass":True})
        except Exception as e:
            print(f"  [FAIL] {e}", flush=True)
            results.append({"label":label,"pass":False,"error":str(e)[:80]})

    print("\n" + "="*65)
    buys  = sum(1 for r in results if r.get("signal")=="BUY")
    sells = sum(1 for r in results if r.get("signal")=="SELL")
    holds = sum(1 for r in results if r.get("signal")=="HOLD")
    apprv = sum(1 for r in results if r.get("approved"))
    execs = sum(1 for r in results if r.get("approved"))
    print(f"  RESULT  : {passed}/12 passed")
    print(f"  Signals : BUY={buys} SELL={sells} HOLD={holds}")
    print(f"  Approved: {apprv}/12  Executed: {execs}/12")
    print("="*65)
    return passed == 12


if __name__ == "__main__":
    sys.exit(0 if run() else 1)
