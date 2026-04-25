"""Shared Ollama/Mistral wrapper."""

import json
import os
import sys

import requests

OLLAMA_URL = "http://localhost:11434/api/generate"
LLM_MODEL  = os.getenv("RP_MODEL", "mistral")  # override: RP_MODEL=mistral-rp python src/pipeline.py


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
    if not raw:
        return {}
    try:
        return json.loads(raw)
    except Exception:
        pass
    # Truncated JSON — try to salvage a partial object by closing open braces
    salvaged = raw.rstrip()
    open_braces  = salvaged.count("{") - salvaged.count("}")
    open_brackets = salvaged.count("[") - salvaged.count("]")
    # Close any unterminated string first
    if salvaged and salvaged[-1] not in ('"', '}', ']'):
        # Drop the last incomplete token (could be a partial string or value)
        for i in range(len(salvaged) - 1, -1, -1):
            if salvaged[i] in (',', '{', '[', ':'):
                salvaged = salvaged[:i]
                break
    salvaged += "]" * max(0, open_brackets) + "}" * max(0, open_braces)
    try:
        data = json.loads(salvaged)
        print("    [llm] recovered partial JSON", flush=True)
        return data
    except Exception:
        return {}
