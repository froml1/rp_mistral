"""
Two-level tagging for RP content.

Message level  — fast heuristics, structural analysis (no LLM call).
Scene level    — Mistral with a constrained vocabulary prompt (lightweight).

Message tags : action, dialogue, descriptif, pensée, nsfw_hint, ooc
Scene tags   : selected from SCENE_TAG_VOCAB (2–5 per scene)
"""

import json
import re
import sys
from pathlib import Path

import requests


OLLAMA_URL = "http://localhost:11434/api/generate"
LLM_MODEL = "mistral"

# ── Message-level patterns ────────────────────────────────────────────────────

ACTION_RE = re.compile(r'\*[^*]+\*')
DIALOGUE_RE = re.compile(r'[—«»"""]|(?:^|\s)-\s', re.MULTILINE)
THOUGHT_RE = re.compile(
    r'\b(pensa|songea|réfléchit|se demanda|imagina|se souvint|se rappela|'
    r'pensait|songeait|se demandait)\b',
    re.IGNORECASE,
)

# Unambiguous NSFW indicators in French RP — intentionally conservative.
# Scene-level Mistral tagging catches the subtler cases.
NSFW_RE = re.compile(
    r'\b(déshabill\w+|dénud\w+|nu\b|nue\b|nus\b|nues\b|'
    r'caress\w+|étreint\w+|enlac\w+|embras\w+|'
    r'gémit|gémiss\w+|haleta|haletait|frémit|frémiss\w+|'
    r'plaisir\b|jouissance|désir\b|désirs\b)\b',
    re.IGNORECASE,
)

# ── Scene-level vocabulary ────────────────────────────────────────────────────

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

TAGGING_SYSTEM = (
    "Tu es un classificateur de scènes narratives. "
    "Tu retournes UNIQUEMENT du JSON valide, sans texte supplémentaire."
)

TAGGING_PROMPT = """\
Lis cette scène de roleplay et attribue-lui 2 à 5 tags thématiques.

Retourne exactement ce JSON :
{{"tags": ["tag1", "tag2"]}}

Choisis UNIQUEMENT parmi cette liste :
{vocab}

Si la scène contient du contenu sexuellement explicite, inclus "nsfw".
Si aucun tag ne s'applique vraiment, retourne {{"tags": ["tension"]}}.

Scène :
---
{text}
---"""


# ── Message-level tagger (heuristic) ─────────────────────────────────────────

def tag_message(raw_content: str, is_ooc: bool) -> list[str]:
    """Fast structural tagging — no LLM call."""
    if is_ooc:
        return ["ooc"]

    tags: list[str] = []

    has_action = bool(ACTION_RE.search(raw_content))
    has_dialogue = bool(DIALOGUE_RE.search(raw_content))
    has_thought = bool(THOUGHT_RE.search(raw_content))
    has_nsfw_hint = bool(NSFW_RE.search(raw_content))

    if has_action:
        tags.append("action")
    if has_dialogue:
        tags.append("dialogue")
    if not has_action and not has_dialogue:
        tags.append("descriptif")
    if has_thought:
        tags.append("pensée")
    if has_nsfw_hint:
        tags.append("nsfw_hint")

    return tags


# ── Scene-level tagger (Mistral) ──────────────────────────────────────────────

def tag_scene(scene_text: str) -> list[str]:
    """
    Calls Mistral to assign thematic tags to a scene.
    Returns a list of tags from SCENE_TAG_VOCAB.
    Falls back to [] on error.
    """
    vocab_str = ", ".join(SCENE_TAG_VOCAB)
    prompt = TAGGING_SYSTEM + "\n\n" + TAGGING_PROMPT.format(
        vocab=vocab_str, text=scene_text
    )
    try:
        resp = requests.post(
            OLLAMA_URL,
            json={"model": LLM_MODEL, "prompt": prompt, "format": "json", "stream": False},
            timeout=60,
        )
        resp.raise_for_status()
        raw = resp.json().get("response", "{}")
        data = json.loads(raw)
        tags = data.get("tags", [])
        # Sanitise: keep only known vocab tags
        return [t for t in tags if t in SCENE_TAG_VOCAB]
    except (requests.RequestException, json.JSONDecodeError, KeyError) as e:
        print(f"  [tagger] scene tagging error: {e}", file=sys.stderr)
        return []
