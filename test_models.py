"""Test which LLM models work with the BluesMinds API."""
import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
    base_url=os.getenv("LLM_BASE_URL"),
    api_key=os.getenv("LLM_API_KEY")
)

models_to_test = [
    "gpt-4o",
    "gpt-4o-mini",
    "deepseek-ai/deepseek-v3.1",
    "deepseek-ai/deepseek-v3.2",
]

print("Testing models...")
working = []
for model in models_to_test:
    try:
        r = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "Say OK"}],
            max_tokens=5
        )
        text = r.choices[0].message.content.strip()
        print(f"  [PASS] {model}: '{text}'")
        working.append(model)
    except Exception as e:
        print(f"  [FAIL] {model}: {str(e)[:80]}")

print(f"\nWorking models: {working}")
