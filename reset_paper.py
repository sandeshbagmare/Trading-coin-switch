import sqlite3
from src.config import settings
from src.config.lm_config import trading_cfg

tcfg = trading_cfg()
balance = tcfg.get("starting_balance", 2000.0)
with sqlite3.connect(str(settings.TRADES_DB)) as c:
    c.execute("DELETE FROM positions")
    c.execute("DELETE FROM trades")
    c.execute("INSERT OR REPLACE INTO state VALUES ('balance',?)", (str(balance),))
print(f"Paper portfolio reset. Balance: INR {balance}")
