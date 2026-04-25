import os
import requests
from pathlib import Path

# Important trading theory open-access papers and crypto strategy guides
urls = [
    ("https://arxiv.org/pdf/2005.13221.pdf", "crypto_pairs_trading.pdf"),
    ("https://arxiv.org/pdf/1908.08703.pdf", "rl_financial_trading.pdf"),
    ("https://arxiv.org/pdf/1805.05256.pdf", "crypto_price_prediction_ta.pdf"),
    ("https://arxiv.org/pdf/1904.05322.pdf", "crypto_volatility_forecasting.pdf"),
]

output_dir = Path("data/knowledge")
output_dir.mkdir(parents=True, exist_ok=True)

print("Starting to download trading research documents...")
for url, filename in urls:
    filepath = output_dir / filename
    if filepath.exists():
        print(f"[OK] {filename} already exists. Skipping.")
        continue
        
    print(f"[*] Downloading {filename}...")
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
    try:
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        with open(filepath, 'wb') as f:
            f.write(response.content)
        print(f"[SUCCESS] Downloaded {filename} successfully.")
    except Exception as e:
        print(f"[ERROR] Failed to download {filename}: {e}")

print("Download complete.")
