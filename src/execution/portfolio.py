"""!
@file portfolio.py
@brief Paper/Live portfolio manager with SQLite persistence.
"""
import sqlite3
import uuid
from datetime import datetime, timezone
from typing import List, Optional
from dataclasses import dataclass
from src.config import settings
from src.config.lm_config import trading_cfg


def _tcfg():
    return trading_cfg()


@dataclass
class Position:
    """!@brief Represents an open trading position."""
    id: str; pair: str; side: str; instrument: str; leverage: int
    quantity_inr: float; entry_price: float
    stop_loss_price: float; take_profit_price: float
    opened_at: str; status: str; reasoning: str; strategy_name: str

    def unrealized_pnl(self, cur: float) -> float:
        if self.entry_price == 0: return 0.0
        chg = (cur - self.entry_price) / self.entry_price
        if self.side == "SELL": chg = -chg
        return self.quantity_inr * self.leverage * chg

    def pnl_pct(self, cur: float) -> float:
        if not self.quantity_inr: return 0.0
        return self.unrealized_pnl(cur) / self.quantity_inr * 100


class Portfolio:
    """!@brief Paper trading portfolio backed by SQLite."""

    _DB = str(settings.TRADES_DB)

    def __init__(self):
        self._init_db()
        self._ensure_balance()

    def _con(self):
        return sqlite3.connect(self._DB)

    def _init_db(self):
        with self._con() as c:
            c.executescript("""
                CREATE TABLE IF NOT EXISTS state (key TEXT PRIMARY KEY, value TEXT);
                CREATE TABLE IF NOT EXISTS positions (
                    id TEXT PRIMARY KEY, pair TEXT, side TEXT, instrument TEXT,
                    leverage INT, quantity_inr REAL, entry_price REAL,
                    stop_loss_price REAL, take_profit_price REAL,
                    opened_at TEXT, status TEXT, reasoning TEXT, strategy_name TEXT);
                CREATE TABLE IF NOT EXISTS trades (
                    id TEXT PRIMARY KEY, pair TEXT, side TEXT, instrument TEXT,
                    leverage INT, quantity_inr REAL, entry_price REAL,
                    exit_price REAL, pnl REAL, pnl_pct REAL, fees REAL,
                    opened_at TEXT, closed_at TEXT, close_reason TEXT,
                    reasoning TEXT, strategy_name TEXT);
            """)

    def _ensure_balance(self):
        with self._con() as c:
            if not c.execute("SELECT 1 FROM state WHERE key='balance'").fetchone():
                c.execute("INSERT INTO state VALUES ('balance',?)",
                          (str(_tcfg().get("starting_balance", 2000.0)),))

    @property
    def balance(self) -> float:
        with self._con() as c:
            row = c.execute("SELECT value FROM state WHERE key='balance'").fetchone()
            return float(row[0]) if row else 2000.0

    @balance.setter
    def balance(self, v: float):
        with self._con() as c:
            c.execute("INSERT OR REPLACE INTO state VALUES ('balance',?)", (str(round(v, 4)),))

    def open_positions(self) -> List[Position]:
        with self._con() as c:
            rows = c.execute("SELECT * FROM positions WHERE status='OPEN'").fetchall()
        return [Position(*r) for r in rows]

    def open_position(self, pair, side, instrument, leverage,
                      quantity_inr, entry_price, sl_price, tp_price,
                      reasoning="", strategy_name="") -> Optional[Position]:
        """!@brief Open a new position, deducting capital from balance."""
        if quantity_inr > self.balance:
            return None
        pos = Position(
            id=str(uuid.uuid4())[:8], pair=pair, side=side, instrument=instrument,
            leverage=leverage, quantity_inr=quantity_inr, entry_price=entry_price,
            stop_loss_price=sl_price, take_profit_price=tp_price,
            opened_at=datetime.now(timezone.utc).isoformat(), status="OPEN",
            reasoning=reasoning, strategy_name=strategy_name,
        )
        self.balance = self.balance - quantity_inr
        with self._con() as c:
            c.execute("INSERT INTO positions VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                      (pos.id, pos.pair, pos.side, pos.instrument, pos.leverage,
                       pos.quantity_inr, pos.entry_price, pos.stop_loss_price,
                       pos.take_profit_price, pos.opened_at, pos.status,
                       pos.reasoning, pos.strategy_name))
        return pos

    def close_position(self, pos_id: str, exit_price: float, reason: str = "MANUAL"):
        """!@brief Close an open position by ID, record trade, update balance."""
        positions = {p.id: p for p in self.open_positions()}
        pos = positions.get(pos_id)
        if not pos: return None

        pnl = pos.unrealized_pnl(exit_price)
        pct = pos.pnl_pct(exit_price)
        fees = pos.quantity_inr * pos.leverage * 0.0005 * 2  # 0.05% each side

        self.balance = self.balance + pos.quantity_inr + pnl - fees
        with self._con() as c:
            c.execute("UPDATE positions SET status='CLOSED' WHERE id=?", (pos_id,))
            c.execute("""INSERT INTO trades VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                      (pos.id, pos.pair, pos.side, pos.instrument, pos.leverage,
                       pos.quantity_inr, pos.entry_price, exit_price,
                       round(pnl - fees, 4), round(pct, 4), round(fees, 4),
                       pos.opened_at, datetime.now(timezone.utc).isoformat(),
                       reason, pos.reasoning, pos.strategy_name))
        return {"id": pos.id, "pair": pos.pair, "pnl": round(pnl - fees, 4), "reason": reason}

    def trade_history(self, limit=50) -> list:
        cols = ["id","pair","side","instrument","leverage","quantity_inr",
                "entry_price","exit_price","pnl","pnl_pct","fees",
                "opened_at","closed_at","close_reason","reasoning","strategy_name"]
        with self._con() as c:
            rows = c.execute("SELECT * FROM trades ORDER BY closed_at DESC LIMIT ?", (limit,)).fetchall()
        return [dict(zip(cols, r)) for r in rows]

    def summary(self, price_map: dict) -> dict:
        positions = self.open_positions()
        upnl = sum(p.unrealized_pnl(price_map.get(p.pair, p.entry_price)) for p in positions)
        trades = self.trade_history(500)
        wins = [t for t in trades if t["pnl"] > 0]
        total_pnl = sum(t["pnl"] for t in trades)
        start_bal = _tcfg().get("starting_balance", 2000.0)
        equity = self.balance + sum(p.quantity_inr for p in positions) + upnl
        return {
            "balance": round(self.balance, 2),
            "equity": round(equity, 2),
            "starting_balance": start_bal,
            "total_pnl": round(total_pnl, 2),
            "total_pnl_pct": round((equity - start_bal) / start_bal * 100, 2),
            "unrealized_pnl": round(upnl, 2),
            "open_positions": len(positions),
            "total_trades": len(trades),
            "win_rate": round(len(wins) / len(trades) * 100, 1) if trades else 0,
            "mode": _tcfg().get("mode", "paper"),
        }
