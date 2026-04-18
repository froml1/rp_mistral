"""
Scene-level synthesis via Mistral.
One LLM call per scene → summary, characters, location, themes, lore hints.
Replaces analyzer.py + tagger.py for the indexing pipeline.
"""

import json
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

_SUMMARY_SYSTEM = (
    "Tu es un analyste de scènes de roleplay narratif écrit. "
    "Tu n'inventes rien : tu te bases uniquement sur ce qui est explicite dans la scène."
)

_SUMMARY_PROMPT = """\
Résume exhaustivement cette scène de roleplay en prose. Tu dois couvrir :
- chaque personnage présent, ce qu'il fait, ce qu'il dit, comment il réagit
- les échanges et interactions entre personnages
- les événements, révélations, tensions dramatiques
- l'évolution de la situation du début à la fin de la scène

Ne laisse aucun personnage de côté. Sois complet et précis.

Alias : {aliases}

Scène :
---
{scene_text}
---"""

_META_SYSTEM = (
    "Tu es un extracteur de métadonnées pour scènes de roleplay. "
    "Tu retournes UNIQUEMENT du JSON valide, sans texte supplémentaire."
)

_META_PROMPT = """\
À partir de ce résumé de scène, extrais :
- characters : personnages présents et actifs (noms canoniques)
- referenced : personnages mentionnés mais absents
- location   : lieu de la scène (null si non précisé)
- themes     : 2-4 tags parmi {vocab}

JSON strict : {{"characters":[],"referenced":[],"location":null,"themes":[]}}

Résumé :
{summary}"""


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
    compressed  = compress_scene_text(scene_text)

    # ── Appel 1 : résumé prose libre ─────────────────────────────────────────
    summary = _call(
        system=_SUMMARY_SYSTEM,
        prompt=_SUMMARY_PROMPT.format(aliases=aliases_str, scene_text=compressed),
        fmt=None,
        num_predict=-1,
    )
    print(f"    [synthesizer] summary={len(summary)} chars")
    if not summary:
        return _empty()

    # ── Appel 2 : métadonnées JSON compactes ──────────────────────────────────
    meta_raw = _call(
        system=_META_SYSTEM,
        prompt=_META_PROMPT.format(vocab=", ".join(SCENE_TAG_VOCAB), summary=summary[:1000]),
        fmt="json",
        num_predict=200,
    )
    try:
        meta = json.loads(meta_raw)
    except Exception:
        meta = {}

    return {
        "summary":    summary,
        "characters": [str(c) for c in (meta.get("characters") or [])],
        "referenced": [str(c) for c in (meta.get("referenced") or [])],
        "location":   meta.get("location"),
        "themes":     [t for t in (meta.get("themes") or []) if t in SCENE_TAG_VOCAB],
    }


def _call(system: str, prompt: str, fmt: str | None, num_predict: int) -> str:
    payload = {
        "model": LLM_MODEL,
        "prompt": system + "\n\n" + prompt,
        "stream": False,
        "keep_alive": -1,
        "options": {"temperature": 0, "top_k": 1, "num_predict": num_predict, "num_ctx": 4096},
    }
    if fmt:
        payload["format"] = fmt
    try:
        resp = requests.post(OLLAMA_URL, json=payload, timeout=300)
        resp.raise_for_status()
        return resp.json().get("response", "")
    except Exception as e:
        print(f"  [synthesizer] ERREUR: {e}", file=sys.stderr)
        return ""
