"""!
@file llm_caller.py
@brief Resilient LLM caller with per-provider rate limiting and ensemble voting.
"""
import json
import time
import threading
from typing import List, Tuple, Dict, Any, Optional
from openai import OpenAI

# Per-provider rate limiting: provider_name -> last_call_time
_provider_locks: Dict[str, threading.Lock] = {}
_provider_times: Dict[str, float] = {}
_provider_rpms: Dict[str, int] = {}


def _init_provider(provider: str, rpm: int):
    """!@brief Register a provider with its RPM limit."""
    if provider not in _provider_locks:
        _provider_locks[provider] = threading.Lock()
        _provider_times[provider] = 0.0
        _provider_rpms[provider] = rpm


def _rate_limit(provider: str):
    """!@brief Enforce per-provider rate limit."""
    if provider not in _provider_locks:
        _init_provider(provider, 10)
    rpm = _provider_rpms.get(provider, 10)
    gap = 60.0 / rpm
    with _provider_locks[provider]:
        now = time.time()
        elapsed = now - _provider_times.get(provider, 0)
        if elapsed < gap:
            time.sleep(gap - elapsed)
        _provider_times[provider] = time.time()


def _extract_json(text: str) -> Optional[Dict]:
    """!@brief Extract the first JSON object from any LLM response string."""
    try:
        start = text.find('{')
        end   = text.rfind('}') + 1
        if start >= 0 and end > start:
            return json.loads(text[start:end])
    except Exception:
        pass
    return None


def call_one(client: OpenAI, model: str, system: str, user: str,
             max_tokens: int = 500, provider: str = "lm1",
             step_callback=None) -> Optional[Dict]:
    """!
    @brief Call a single LLM model and return parsed JSON dict.
    @param step_callback Optional callback(step_text) to broadcast live status.
    """
    for attempt in range(3):
        try:
            _rate_limit(provider)
            if step_callback:
                step_callback(f"Calling {model.split('/')[-1]}...")

            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user",   "content": user},
                ],
                temperature=0.15,
                max_tokens=max_tokens,
                timeout=120,
            )
            text = resp.choices[0].message.content.strip()
            result = _extract_json(text)
            if result is not None:
                return result
            print(f"  [LLM] Non-JSON from {model}: {text[:80]}")
        except Exception as e:
            print(f"  [LLM] {model} attempt {attempt+1} failed: {str(e)[:80]}")
            if attempt < 2:
                time.sleep(2)
    return None


def call_ensemble(models: List[Tuple[OpenAI, str]], system: str, user: str,
                  vote_key: str, max_tokens: int = 500,
                  step_callback=None) -> Dict:
    """!@brief Calls all models and merges by majority vote."""
    results = []
    for client, model in models:
        # Determine provider from client base URL
        provider = getattr(client, '_provider_name', 'lm1')
        r = call_one(client, model, system, user, max_tokens,
                     provider=provider, step_callback=step_callback)
        if r:
            results.append(r)

    if not results:
        return {}
    if len(results) == 1:
        return results[0]

    from collections import Counter
    votes = Counter(r.get(vote_key) for r in results if r.get(vote_key))
    winner = votes.most_common(1)[0][0]
    for r in results:
        if r.get(vote_key) == winner:
            return r
    return results[0]
