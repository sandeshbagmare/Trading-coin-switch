"""!
@file orchestrator.py
@brief Main trading loop with live step broadcasting, ML outcome recording, and early exit evaluation.
"""
import time
import threading
from datetime import datetime
from typing import Dict, Callable, List, Optional

from src.config.lm_config import trading_cfg
from src.data.market_data import get_live_prices, PAIR_MAP
from src.execution.portfolio import Portfolio
from src.execution.broker import LiveBroker
from src.brain.agent import run_cycle
from src.brain import exit_evaluator
from src.risk.signal_scorer import record_outcome

# ── Shared State ──────────────────────────────────────────────────────────────
_running   = False
_broadcast: Optional[Callable] = None
_prices:    Dict[str, dict] = {}
_agent_log: List[dict] = []


def set_broadcast(fn: Callable):
    """!@brief Register WS broadcast callback."""
    global _broadcast
    _broadcast = fn


def _push(event: str, data):
    if _broadcast:
        try: _broadcast(event, data)
        except Exception: pass


def _step_callback(pair: str, step_text: str):
    """!@brief Broadcast a live analysis step to the UI."""
    _push("agent_step", {"pair": pair, "step": step_text,
                         "timestamp": datetime.now().strftime("%H:%M:%S")})


def _check_sl_tp(portfolio: Portfolio, prices: dict):
    """!@brief Check all positions for SL/TP hit."""
    for pos in portfolio.open_positions():
        cur = prices.get(pos.pair, {}).get("price")
        if not cur: continue
        reason = None
        if pos.side == "BUY":
            if cur <= pos.stop_loss_price:   reason = "STOP_LOSS"
            elif cur >= pos.take_profit_price: reason = "TAKE_PROFIT"
        else:
            if cur >= pos.stop_loss_price:   reason = "STOP_LOSS"
            elif cur <= pos.take_profit_price: reason = "TAKE_PROFIT"
        if reason:
            trade = portfolio.close_position(pos.id, cur, reason)
            if trade:
                pnl = trade["pnl"]
                print(f"[Orch] {reason}: {trade['pair']} PnL=₹{pnl:+.2f}")
                _push("trade_closed", {"trade": trade})
                # Record outcome for ML training
                record_outcome({}, pos.side, 1.0, pnl)


def _evaluate_early_exits(portfolio: Portfolio, prices: dict):
    """!@brief Uses Exit Strategist to evaluate early exits on active positions."""
    for pos in portfolio.open_positions():
        cur = prices.get(pos.pair, {}).get("price")
        if not cur: continue
        _step_callback(pos.pair, f"🔎 Exit evaluator checking {pos.pair}...")
        eval_result = exit_evaluator.run(pos, cur)
        if eval_result.get("decision") == "EXIT":
            reason = f"EARLY_EXIT: {eval_result.get('reason', 'Trend reversed')}"
            trade = portfolio.close_position(pos.id, cur, reason)
            if trade:
                print(f"[Orch] {reason}: {trade['pair']} PnL=₹{trade['pnl']:+.2f}")
                _push("trade_closed", {"trade": trade})
                record_outcome({}, pos.side, 1.0, trade["pnl"])


def _execute_decision(portfolio: Portfolio, decision: dict, price: float, broker: Optional[LiveBroker] = None):
    """!@brief Execute a single approved trade decision via portfolio & broker."""
    if not decision.get("execute"): return None
    side = decision.get("action")
    if side not in ("BUY", "SELL"): return None
    
    if broker:
        live_id = broker.place_real_order(decision["pair"], side, float(decision.get("size_inr", 0)), price)
        if live_id:
            print(f"  [Orch] Placed LIVE physical order! ID: {live_id}")
        else:
            print(f"  [Orch] LIVE physical order failed, reverting to paper execution track.")

    pos = portfolio.open_position(
        pair=decision["pair"], side=side,
        instrument=decision.get("instrument", "SPOT"),
        leverage=int(decision.get("leverage", 1)),
        quantity_inr=float(decision.get("size_inr", 0)),
        entry_price=price,
        sl_price=float(decision.get("stop_loss_price", price * 0.95)),
        tp_price=float(decision.get("take_profit_price", price * 1.05)),
        reasoning=decision.get("reasoning", ""),
        strategy_name=decision.get("strategy_name", ""),
    )
    if pos:
        print(f"[Orch] OPENED {pos.side} {pos.pair} @ ₹{pos.entry_price:,.0f} "
              f"₹{pos.quantity_inr:.0f}×{pos.leverage}x")
        _push("trade_opened", {"position": {
            "id": pos.id, "pair": pos.pair, "side": pos.side,
            "instrument": pos.instrument, "leverage": pos.leverage,
            "size": pos.quantity_inr, "entry_price": pos.entry_price,
            "sl": pos.stop_loss_price, "tp": pos.take_profit_price,
            "reasoning": pos.reasoning, "opened_at": pos.opened_at,
        }})
    return pos


def _main_loop():
    global _prices, _running
    portfolio = Portfolio()
    tcfg      = trading_cfg()
    pairs     = tcfg.get("pairs", list(PAIR_MAP.keys()))
    is_live   = tcfg.get("mode", "paper").lower() == "live"
    broker    = LiveBroker() if is_live else None
    cycle     = 0

    while _running:
        cycle += 1
        print(f"\n[Orch] === Cycle {cycle} @ {datetime.now().strftime('%H:%M:%S')} ===")
        _push("cycle_start", {"cycle": cycle, "time": datetime.now().strftime("%H:%M:%S")})

        prices = get_live_prices()
        if prices:
            _prices = prices
            _push("prices", {p: d.get("price", 0) for p, d in prices.items()})

        _check_sl_tp(portfolio, prices)

        # Early exit evaluation on open positions
        if portfolio.open_positions():
            _evaluate_early_exits(portfolio, prices)

        price_map = {p: d.get("price", 0) for p, d in prices.items()}
        _push("portfolio", portfolio.summary(price_map))

        # Real Live Balance or Simulated Paper Balance
        balance  = broker.fetch_real_balance() if broker else portfolio.balance
        if broker:
            print(f"  [Orch] True Live Balance from Vault: ₹{balance:,.2f}")
        
        open_pos = portfolio.open_positions()

        for pair in pairs:
            if not prices.get(pair): continue
            _push("scanning", {"pair": pair})

            result = run_cycle(pair, prices, balance, open_pos,
                              step_cb=_step_callback)
            _agent_log.append(result)
            if len(_agent_log) > 50: _agent_log.pop(0)

            decision = result.get("decision", {})
            _push("agent_signal", {
                "pair":       pair,
                "signal":     result["analysis"].get("signal", "HOLD"),
                "confidence": result["analysis"].get("confidence", 0),
                "strategy":   result.get("strategy", {}).get("strategy_name", ""),
                "action":     decision.get("action", "HOLD"),
                "reasoning":  decision.get("reasoning", ""),
                "timestamp":  result["timestamp"],
                "approved":   result.get("risk", {}).get("approved", False),
                "sl":         result.get("risk", {}).get("stop_loss_price", 0),
                "tp":         result.get("risk", {}).get("take_profit_price", 0),
                "rr":         result.get("risk", {}).get("actual_rr", 0),
                "ml_score":   result.get("ml_score"),
                "indicators": result.get("indicators", ""),
            })

            if decision.get("execute"):
                pos = _execute_decision(portfolio, decision, result["price"], broker)
                if pos:
                    balance   -= pos.quantity_inr
                    open_pos.append(pos)

        # Push live position P&L update
        pos_data = []
        for p in portfolio.open_positions():
            cur = prices.get(p.pair, {}).get("price", p.entry_price)
            pos_data.append({
                "id": p.id, "pair": p.pair, "side": p.side,
                "instrument": p.instrument, "leverage": p.leverage,
                "size": p.quantity_inr, "entry_price": p.entry_price,
                "current_price": cur,
                "unrealized_pnl": round(p.unrealized_pnl(cur), 2),
                "pnl_pct": round(p.pnl_pct(cur), 2),
                "sl": p.stop_loss_price, "tp": p.take_profit_price,
                "strategy_name": p.strategy_name,
                "reasoning": p.reasoning, "opened_at": p.opened_at,
            })
        _push("positions", pos_data)
        time.sleep(1)


def start():
    """!@brief Start the orchestrator background thread."""
    global _running
    _running = True
    t = threading.Thread(target=_main_loop, daemon=True, name="Orchestrator")
    t.start()
    print("[Orch] Started.")


def stop():
    """!@brief Stop the orchestrator."""
    global _running
    _running = False


def get_agent_log()  -> List[dict]: return list(reversed(_agent_log))
def get_last_prices() -> Dict:      return _prices
