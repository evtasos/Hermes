import httpx
import time

OLLAMA_HOST = "http://192.168.1.90:11434"
MODELS = ["qwen3.5:0.8b", "qwen3.5:2b", "qwen3:1.7b"]
PROMPTS = [
    "What's 12 + 7?",
    "Turn on the kitchen light.",
    "What's the weather usually like in Athens in July?"
]

def call_ollama(model, prompt):
    r = httpx.post(
        f"{OLLAMA_HOST}/api/generate",
        json={"model": model, "prompt": prompt, "stream": False, "keep_alive": 0},
        timeout=120
    )
    r.raise_for_status()
    return r.json()

for model in MODELS:
    print(f"\n{'='*50}\nMODEL: {model}\n{'='*50}")
    for prompt in PROMPTS:
        start = time.time()
        result = call_ollama(model, prompt)
        elapsed = time.time() - start
        tokens = result.get('eval_count', 0)
        eval_dur = result.get('eval_duration', 1) / 1e9
        tps = tokens / eval_dur if eval_dur > 0 else 0
        print(f"\nPrompt: {prompt}")
        print(f"  Total time: {elapsed:.2f}s | Tokens: {tokens} | Tok/s: {tps:.2f}")
        print(f"  Response: {result['response'][:150]}{'...' if len(result['response']) > 150 else ''}")
