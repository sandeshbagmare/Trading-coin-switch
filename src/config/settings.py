"""!
@file settings.py
@brief Centralized settings loaded from lm.yaml + environment fallbacks.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).parent.parent.parent

# ── Lazy import to avoid circular ────────────────────────────────────────────
def _tcfg():
    from src.config.lm_config import trading_cfg
    return trading_cfg()

def _scfg():
    from src.config.lm_config import server_cfg
    return server_cfg()

@property
def TRADING_MODE() -> str: return _tcfg().get("mode", "paper")

# ── Static constants used before lm.yaml is loaded ───────────────────────────
CHROMA_DB_DIR  = str(BASE_DIR / "data" / "knowledge" / "chromadb")
KNOWLEDGE_DIR  = str(BASE_DIR / "data" / "knowledge")
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
COINSWITCH_API_KEY    = os.getenv("COINSWITCH_API_KEY", "PLACEHOLDER")
COINSWITCH_SECRET_KEY = os.getenv("COINSWITCH_SECRET_KEY", "PLACEHOLDER")

MODELS_DIR = BASE_DIR / "data" / "models"
LOGS_DIR   = BASE_DIR / "data" / "logs"
TRADES_DB  = BASE_DIR / "data" / "trades.db"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)
(BASE_DIR / "data" / "knowledge" / "chromadb").mkdir(parents=True, exist_ok=True)


def get_trading_cfg():
    """!@brief Returns full trading config dict from lm.yaml."""
    return _tcfg()

def get_server_cfg():
    """!@brief Returns full server config dict from lm.yaml."""
    return _scfg()
