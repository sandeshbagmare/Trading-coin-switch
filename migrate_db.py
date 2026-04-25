"""Migrate existing trades.db to add strategy_name column and reset paper portfolio."""
import sqlite3
from src.config.lm_config import trading_cfg

con = sqlite3.connect("data/trades.db")
cur = con.cursor()

try:
    cur.execute("ALTER TABLE positions ADD COLUMN strategy_name TEXT DEFAULT ''")
    print("Added strategy_name to positions")
except Exception as e:
    print(f"positions: {e}")

try:
    cur.execute("ALTER TABLE trades ADD COLUMN strategy_name TEXT DEFAULT ''")
    print("Added strategy_name to trades")
except Exception as e:
    print(f"trades: {e}")

tcfg = trading_cfg()
balance = tcfg.get("starting_balance", 2000.0)
cur.execute("DELETE FROM positions")
cur.execute("DELETE FROM trades")
cur.execute("INSERT OR REPLACE INTO state VALUES ('balance', ?)", (str(balance),))
con.commit()
con.close()
print(f"DB migrated and reset. Balance: INR {balance}")
