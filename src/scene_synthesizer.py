"""
Scene-level synthesis via Mistral.
One LLM call per scene → summary, characters, location, themes, lore hints.
Replaces analyzer.py + tagger.py for the indexing pipeline.
"""

import re
import sys

import requests

from analyzer import compress_scene_text


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
    "Tu n'inventes rien : tu te bases uniquement sur ce qui est explicite ou fortement implicite dans la scène."
)

_PROMPT = """\
Alias connus : {aliases}

Analyse cette scène de roleplay. Réponds en respectant EXACTEMENT ce format (une valeur par ligne) :

SUMMARY: <résumé narratif exhaustif — tout ce qui se passe, se dit, se fait. Autant de phrases que nécessaire.>
CHARACTERS: <noms canoniques séparés par des virgules, ou vide>
REFERENCED: <personnages mentionnés mais absents, séparés par des virgules, ou vide>
LOCATION: <lieu de la scène, ou vide>
THEMES: <2-4 tags parmi {vocab}, séparés par des virgules>

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

    print(f"    [synthesizer] prompt {len(prompt)} chars, envoi à Ollama...")
    try:
        resp = requests.post(
            OLLAMA_URL,
            json={
                "model": LLM_MODEL,
                "prompt": prompt,
                "stream": False,
                "keep_alive": -1,
                "options": {"temperature": 0, "top_k": 1, "num_predict": -1, "num_ctx": 4096},
            },
            timeout=300,
        )
        resp.raise_for_status()
        resp_json = resp.json()
        raw = resp_json.get("response", "")
        print(f"    [synthesizer] réponse {len(raw)} chars | done={resp_json.get('done')} | done_reason={resp_json.get('done_reason')}")
        print(f"    [synthesizer] raw[:300]: {repr(raw[:300])}")
    except Exception as e:
        print(f"  [synthesizer] ERREUR: {type(e).__name__}: {e}", file=sys.stderr)
        return _empty()

    result = _parse_text_response(raw)
    print(f"    [synthesizer] parsed: summary={len(result['summary'])} chars, characters={result['characters']}")
    if not result["summary"]:
        print(f"    [synthesizer] RAW COMPLET:\n{raw}")
    return result


def _parse_text_response(raw: str) -> dict:
    def _field(key: str) -> str:
        m = re.search(rf'^{key}:\s*(.+?)(?=\n[A-Z]+:|$)', raw, re.MULTILINE | re.DOTALL)
        return m.group(1).strip() if m else ""

    def _list(key: str) -> list[str]:
        val = _field(key)
        return [v.strip() for v in val.split(",") if v.strip()] if val else []

    themes = [t for t in _list("THEMES") if t in SCENE_TAG_VOCAB]
    location = _field("LOCATION") or None

    return {
        "summary":    _field("SUMMARY"),
        "characters": _list("CHARACTERS"),
        "referenced": _list("REFERENCED"),
        "location":   location,
        "themes":     themes,
    }
