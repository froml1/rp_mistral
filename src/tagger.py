"""
Scene-level thematic tagging via Mistral.
Message-level structural tagging is handled by analyzer.py.
"""

import json
import sys

import requests

from analyzer import compress_scene_text

OLLAMA_URL = "http://localhost:11434/api/generate"
LLM_MODEL = "mistral"

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
    "Tu es un classificateur de scènes narratives. "
    "Tu retournes UNIQUEMENT du JSON valide, sans texte supplémentaire."
)

_PROMPT = """\
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


def tag_scene(scene_text: str) -> list[str]:
    """Calls Mistral to assign thematic tags to a scene. Falls back to [] on error."""
    prompt = _SYSTEM + "\n\n" + _PROMPT.format(
        vocab=", ".join(SCENE_TAG_VOCAB),
        text=compress_scene_text(scene_text),
    )
    try:
        resp = requests.post(
            OLLAMA_URL,
            json={"model": LLM_MODEL, "prompt": prompt, "format": "json", "stream": False, "options": {
            "temperature": 0,
            "top_k": 1,
            "num_predict": 250,
            "num_ctx": 4096
        }},
            timeout=60,
        )
        resp.raise_for_status()
        data = json.loads(resp.json().get("response", "{}"))
        return [t for t in data.get("tags", []) if t in SCENE_TAG_VOCAB]
    except (requests.RequestException, json.JSONDecodeError, KeyError) as e:
        print(f"  [tagger] scene tagging error: {e}", file=sys.stderr)
        return []
