"""Shared Ollama/Mistral wrapper with per-task model routing."""

import json
import os
import sys
from pathlib import Path

import requests

# Auto-load config/models.env without overriding explicit env vars
_CONFIG_ENV = Path(__file__).resolve().parent.parent / "config" / "models.env"
if _CONFIG_ENV.exists():
    for _line in _CONFIG_ENV.read_text(encoding="utf-8").splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _k, _, _v = _line.partition("=")
            _k = _k.strip()
            if _k.startswith("RP_MODEL") and _k not in os.environ:
                os.environ[_k] = _v.strip().strip('"').strip("'")

OLLAMA_URL  = "http://localhost:11434/api/generate"
LLM_MODEL   = os.getenv("RP_MODEL", "mistral")

# Per-task model overrides via environment variables.
# All models must be ≤ 7B. Defaults to LLM_MODEL if not set.
#
# Example — use a dedicated translation model:
#   RP_MODEL_TRANSLATE=mistral python src/pipeline.py
#
# Recommended split:
#   translate, clean, rp_filter, prefix  → fast/light model (translation-optimized or small)
#   split, analyze, general_lore         → main model (better reasoning)
_TASK_MODELS: dict[str, str] = {
    "translate": os.getenv("RP_MODEL_TRANSLATE", LLM_MODEL),
    "clean":     os.getenv("RP_MODEL_CLEAN",     LLM_MODEL),
    "rp_filter": os.getenv("RP_MODEL_FILTER",    LLM_MODEL),
    "prefix":    os.getenv("RP_MODEL_PREFIX",     LLM_MODEL),
    "split":     os.getenv("RP_MODEL_SPLIT",      LLM_MODEL),
    "analyze":   os.getenv("RP_MODEL_ANALYZE",    LLM_MODEL),
}


def model_for(task: str) -> str:
    """Return the configured model for a given task."""
    return _TASK_MODELS.get(task, LLM_MODEL)


def call_llm(prompt: str, fmt: str | None = None, num_predict: int = -1,
             num_ctx: int = 8192, model: str | None = None) -> str:
    payload = {
        "model":      model or LLM_MODEL,
        "prompt":     prompt,
        "stream":     False,
        "keep_alive": -1,
        "options": {"temperature": 0, "top_k": 1, "num_predict": num_predict, "num_ctx": num_ctx},
    }
    if fmt:
        payload["format"] = fmt
    try:
        resp = requests.post(OLLAMA_URL, json=payload, timeout=300)
        resp.raise_for_status()
        raw = resp.json().get("response", "")
        print(f"    [llm/{payload['model']}] {len(raw)} chars", flush=True)
        return raw
    except Exception as e:
        print(f"    [llm] error: {e}", file=sys.stderr, flush=True)
        return ""


def call_llm_json(prompt: str, num_predict: int = 512, num_ctx: int = 8192,
                  model: str | None = None) -> dict:
    raw = call_llm(prompt, fmt="json", num_predict=num_predict, num_ctx=num_ctx, model=model)
    if not raw:
        return {}
    try:
        return json.loads(raw)
    except Exception:
        pass
    # Truncated JSON — try to salvage a partial object by closing open braces
    salvaged = raw.rstrip()
    open_braces   = salvaged.count("{") - salvaged.count("}")
    open_brackets = salvaged.count("[") - salvaged.count("]")
    if salvaged and salvaged[-1] not in ('"', '}', ']'):
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
