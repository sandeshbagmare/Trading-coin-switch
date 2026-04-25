"""!
@file lm_config.py
@brief Loads lm.yaml and builds LLM clients + role→model routing with per-provider RPM.
"""
import yaml
from pathlib import Path
from openai import OpenAI
from typing import List, Tuple, Dict, Any
from src.brain.llm_caller import _init_provider

_cfg: Dict[str, Any] = {}
_clients: Dict[str, OpenAI] = {}

BASE_DIR = Path(__file__).parent.parent.parent


def load() -> Dict[str, Any]:
    """!@brief Load and cache lm.yaml. Returns parsed dict."""
    global _cfg
    if not _cfg:
        path = BASE_DIR / "lm.yaml"
        with open(path, "r") as f:
            _cfg = yaml.safe_load(f)
    return _cfg


def _client(provider_name: str) -> OpenAI:
    """!@brief Returns a cached OpenAI-compatible client for a provider."""
    if provider_name not in _clients:
        cfg = load()
        p = cfg["providers"][provider_name]
        client = OpenAI(base_url=p["base_url"], api_key=p["api_key"])
        client._provider_name = provider_name  # Tag for rate limiter
        _clients[provider_name] = client
        # Register RPM with rate limiter
        rpm = p.get("rpm", 10)
        _init_provider(provider_name, rpm)
    return _clients[provider_name]


def get_role_models(role: str) -> List[Tuple[OpenAI, str]]:
    """!@brief Returns list of (client, model_name) for a given role."""
    cfg = load()
    entries = cfg.get("roles", {}).get(role, [])
    result = []
    for entry in entries:
        client = _client(entry["provider"])
        result.append((client, entry["model"]))
    return result


def get_total_rpm() -> int:
    """!@brief Sum of RPM across all active providers."""
    cfg = load()
    return sum(p.get("rpm", 10) for p in cfg.get("providers", {}).values())


def trading_cfg() -> Dict[str, Any]:
    """!@brief Returns the 'trading' section from lm.yaml."""
    return load().get("trading", {})


def server_cfg() -> Dict[str, Any]:
    """!@brief Returns the 'server' section from lm.yaml."""
    return load().get("server", {})
