"""
Scene-level synthesis via Mistral.
One LLM call per scene → summary, characters, location, themes, lore hints.
Replaces analyzer.py + tagger.py for the indexing pipeline.
"""

import json
import sys

import requests

from analyzer import compress_scene_text, preprocess_for_llm


OLLAMA_URL = "http://localhost:11434/api/generate"
LLM_MODEL  = "mistral"

SCENE_TAG_VOCAB: list[str] = [
    "combat", "violence", "fuite", "poursuite",
    "romance", "intimité", "nsfw",
    "intrigue", "trahison", "manipulation", "alliance",
    "confrontation", "négociation", "menace",
    "révélation", "secret", "enquête",
    "deuil", "perte", "rédemption",
    "humour", "légèreté",
    "rituel", "magie", "surnaturel",
    "politique", "pouvoir",
    "exploration", "voyage",
    "philosophie", "tension",
]

_SYSTEM = (
    "Tu es un analyste de scènes de roleplay narratif écrit. "
    "Tu retournes UNIQUEMENT du JSON valide, sans texte supplémentaire. "
    "Tu n'inventes rien : tu te bases uniquement sur ce qui est explicite ou fortement implicite dans la scène."
)

_PROMPT = """\
Alias connus : {aliases}

Analyse cette scène de roleplay et retourne :
- summary    : résumé narratif dense en 3-6 phrases (prose, ce qui se passe, ce qui se dit, ce qui se fait)
- characters : personnages présents et actifs (noms canoniques)
- referenced : personnages mentionnés mais non présents
- location   : lieu de la scène (null si non précisé)
- themes     : 2-4 tags parmi {vocab}

Retour JSON strict :
{{"summary":"","characters":[],"referenced":[],"location":null,"themes":[]}}

Scène :
---
{scene_text}
---"""


def _empty() -> dict:
    return {
        "summary":    "",
        "characters": [],
        "referenced": [],
        "location":   None,
        "themes":     [],
    }


def synthesize_scene(
    scene_text: str,
    alias_map: dict | None = None,
) -> dict:
    """
    One Mistral call per scene.
    Returns summary, characters, referenced, location, themes, lore.
    """
    if not scene_text.strip():
        return _empty()

    aliases_str = ", ".join(f"{k}→{v}" for k, v in (alias_map or {}).items()) or "aucun"
    prompt = _SYSTEM + "\n\n" + _PROMPT.format(
        aliases=aliases_str,
        vocab=", ".join(SCENE_TAG_VOCAB),
        scene_text=compress_scene_text(scene_text),
    )

    try:
        resp = requests.post(
            OLLAMA_URL,
            json={
                "model": LLM_MODEL,
                "prompt": prompt,
                "format": "json",
                "stream": False,
                "keep_alive": -1,
                "options": {"temperature": 0, "top_k": 1, "num_predict": 1024, "num_ctx": 4096},
            },
            timeout=180,
        )
        resp.raise_for_status()
        raw = resp.json().get("response", "{}")
        print(f"    [synthesizer] raw response ({len(raw)} chars): {raw[:200]}")
        data = json.loads(raw)
    except Exception as e:
        print(f"  [synthesizer] error: {e}", file=sys.stderr)
        return _empty()

    return {
        "summary":    str(data.get("summary") or ""),
        "characters": [str(c) for c in (data.get("characters") or [])],
        "referenced": [str(c) for c in (data.get("referenced") or [])],
        "location":   data.get("location"),
        "themes":     [t for t in (data.get("themes") or []) if t in SCENE_TAG_VOCAB],
    }
