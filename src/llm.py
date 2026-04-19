"""Shared Ollama/Mistral wrapper."""

import json
import sys

import requests

OLLAMA_URL = "http://localhost:11434/api/generate"
LLM_MODEL  = "mistral"


def call_llm(prompt: str, fmt: str | None = None, num_predict: int = -1, num_ctx: int = 8192) -> str:
    payload = {
        "model": LLM_MODEL,
        "prompt": prompt,
        "stream": False,
        "keep_alive": -1,
        "options": {"temperature": 0, "top_k": 1, "num_predict": num_predict, "num_ctx": num_ctx},
    }
    if fmt:
        payload["format"] = fmt
    try:
        resp = requests.post(OLLAMA_URL, json=payload, timeout=300)
        resp.raise_for_status()
        raw = resp.json().get("response", "")
        print(f"    [llm] {len(raw)} chars", flush=True)
        return raw
    except Exception as e:
        print(f"    [llm] error: {e}", file=sys.stderr, flush=True)
        return ""


def call_llm_json(prompt: str, num_predict: int = 512, num_ctx: int = 8192) -> dict:
    raw = call_llm(prompt, fmt="json", num_predict=num_predict, num_ctx=num_ctx)
    try:
        data = json.loads(raw)
        return data
    except Exception:
        return {}
