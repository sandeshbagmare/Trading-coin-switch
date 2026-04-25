"""Test raw API latency."""
import time
from openai import OpenAI
import yaml

with open("lm.yaml") as f:
    cfg = yaml.safe_load(f)

p1 = cfg["providers"]["lm1"]
c = OpenAI(base_url=p1["base_url"], api_key=p1["api_key"])

for model in ["deepseek-ai/deepseek-v3.2", "deepseek-ai/deepseek-v3.1"]:
    t = time.time()
    try:
        r = c.chat.completions.create(
            model=model,
            messages=[{"role":"user","content":'Return: {"ok":true}'}],
            max_tokens=20, timeout=60
        )
        out = r.choices[0].message.content[:60]
        print(f"[{model}] {time.time()-t:.1f}s: {out}")
    except Exception as e:
        print(f"[{model}] FAILED {time.time()-t:.1f}s: {str(e)[:80]}")
