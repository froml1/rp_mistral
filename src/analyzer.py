"""
LLM-based message analyzer — replaces all regex-based content analysis.

Sends batches of Discord messages to Mistral and returns per-message
structured analysis: RP/HRP classification, character attribution,
speaker segmentation, and structural tags.
"""

import json
import sys

import requests


OLLAMA_URL = "http://localhost:11434/api/generate"
LLM_MODEL = "mistral"
BATCH_SIZE = 10

MSG_TAG_VOCAB = ["action", "dialogue", "descriptif", "pensée", "nsfw_hint"]

_SYSTEM = (
    "Tu es un analyseur de messages pour jeu de rôle narratif sur Discord. "
    "Tu retournes UNIQUEMENT du JSON valide, sans texte supplémentaire."
)

_PROMPT = """\
Alias connus : {aliases}

Conventions :
- Les actions sont entre *astérisques* ; le sujet grammatical du premier bloc = personnage actif
- Les dialogues directs sont précédés de — ou entre guillemets " "
- Un tiret « - » après * . ! ? » " = changement de locuteur dans le même message
- Les commentaires méta sont entre ((...)) ou [OOC...]
- Résoudre les alias vers leur nom canonique

Pour chaque message, retourne :
- is_rp : true si roleplay narratif (actions, dialogues de scène, descriptions), false si hors-rp (bavardage informel, liens, emojis excessifs)
- is_ooc : true si le message entier est un commentaire méta
- characters : liste des noms canoniques des personnages actifs (sujets d'actions ou locuteurs directs)
- referenced : noms canoniques des personnages évoqués/absents
- speaker_segments : découpe en segments [{{"text":"...","character":"Nom ou null","type":"action|dialogue|narration"}}]
- tags : 0 à 2 tags parmi : {vocab}

Retourne uniquement ce JSON (les tableaux peuvent être vides) :
{{"messages":[{{"index":0,"is_rp":true,"is_ooc":false,"characters":[],"referenced":[],"speaker_segments":[{{"text":"","character":null,"type":"narration"}}],"tags":[]}}]}}

Messages :
{messages}"""


def _get_author(msg: dict) -> str:
    author = msg.get("author", {})
    if isinstance(author, dict):
        return author.get("name", "?")
    return str(author) or "?"


def _default() -> dict:
    return {
        "is_rp": False,
        "is_ooc": False,
        "characters": [],
        "referenced": [],
        "speaker_segments": [],
        "tags": [],
    }


def analyze_batch(
    messages: list[dict],
    alias_map: dict[str, str] | None = None,
) -> list[dict]:
    """
    Sends a batch of raw Discord messages to Mistral.
    Returns one analysis dict per message (same order, same length).
    """
    if not messages:
        return []

    aliases_str = ", ".join(f"{k}→{v}" for k, v in (alias_map or {}).items()) or "aucun"
    vocab_str = ", ".join(MSG_TAG_VOCAB)
    formatted = "\n".join(
        f"[{i}] {_get_author(m)}: {m.get('content', '')}"
        for i, m in enumerate(messages)
    )

    prompt = _SYSTEM + "\n\n" + _PROMPT.format(
        aliases=aliases_str,
        vocab=vocab_str,
        messages=formatted,
    )

    try:
        resp = requests.post(
            OLLAMA_URL,
            json={"model": LLM_MODEL, "prompt": prompt, "format": "json", "stream": False},
            timeout=120,
        )
        resp.raise_for_status()
        data = json.loads(resp.json().get("response", "{}"))
        by_index = {item["index"]: item for item in data.get("messages", [])}
    except Exception as e:
        print(f"  [analyzer] batch error: {e}", file=sys.stderr)
        by_index = {}

    output = []
    for i in range(len(messages)):
        r = by_index.get(i, {})
        output.append({
            "is_rp":    bool(r.get("is_rp", False)),
            "is_ooc":   bool(r.get("is_ooc", False)),
            "characters": [str(c) for c in (r.get("characters") or [])],
            "referenced": [str(c) for c in (r.get("referenced") or [])],
            "speaker_segments": [
                {
                    "text":      str(s.get("text", "")),
                    "character": s.get("character"),
                    "type":      s.get("type", "narration"),
                }
                for s in (r.get("speaker_segments") or [])
            ],
            "tags": [t for t in (r.get("tags") or []) if t in MSG_TAG_VOCAB],
        })
    return output


_CLASSIFY_SYSTEM = (
    "Tu es un classificateur de messages Discord pour un jeu de rôle narratif. "
    "Tu retournes UNIQUEMENT du JSON valide, sans texte supplémentaire."
)

_CLASSIFY_PROMPT = """\
Pour chaque message, indique s'il fait partie d'une scène de roleplay (is_rp: true) ou non (is_rp: false).
Un commentaire méta entre ((...)) ou [OOC...] dans une scène est is_ooc: true (et is_rp: true).

Critères RP : actions entre *astérisques*, dialogues narratifs, descriptions de scène, style littéraire.
Critères HRP : bavardage informel, liens, emojis excessifs, réactions courtes.

Retourne exactement ce JSON :
{{"classifications": [{{"index": 0, "is_rp": true, "is_ooc": false}}]}}

Messages :
{messages}"""


def classify_messages(
    messages: list[dict],
    batch_size: int = BATCH_SIZE,
    context_overlap: int = 3,
) -> list[dict]:
    """
    Lightweight RP/HRP classification — only returns {is_rp, is_ooc} per message.
    Used by the purger; cheaper and more reliable than full analysis.
    """
    if not messages:
        return []

    results: list[dict | None] = [None] * len(messages)

    for start in range(0, len(messages), batch_size):
        end = min(start + batch_size, len(messages))
        ctx_start = max(0, start - context_overlap)
        batch = messages[ctx_start:end]

        formatted = "\n".join(
            f"[{i}] {_get_author(m)}: {m.get('content', '')}"
            for i, m in enumerate(batch)
        )
        prompt = _CLASSIFY_SYSTEM + "\n\n" + _CLASSIFY_PROMPT.format(messages=formatted)

        try:
            resp = requests.post(
                OLLAMA_URL,
                json={"model": LLM_MODEL, "prompt": prompt, "format": "json", "stream": False},
                timeout=120,
            )
            resp.raise_for_status()
            data = json.loads(resp.json().get("response", "{}"))
            by_index = {item["index"]: item for item in data.get("classifications", [])}
        except Exception as e:
            print(f"  [analyzer] classify error: {e}", file=sys.stderr)
            by_index = {}

        offset = start - ctx_start
        for j, local_j in enumerate(range(offset, len(batch))):
            global_i = start + j
            if global_i < len(messages):
                r = by_index.get(local_j, {})
                results[global_i] = {
                    "is_rp":  bool(r.get("is_rp", False)),
                    "is_ooc": bool(r.get("is_ooc", False)),
                }

    return [r if r is not None else {"is_rp": False, "is_ooc": False} for r in results]


def analyze_messages(
    messages: list[dict],
    alias_map: dict[str, str] | None = None,
    batch_size: int = BATCH_SIZE,
    context_overlap: int = 3,
) -> list[dict]:
    """
    Analyzes all messages, processing in overlapping batches to preserve
    narrative context at batch boundaries.

    Returns one analysis dict per input message.
    """
    if not messages:
        return []

    results: list[dict | None] = [None] * len(messages)

    for start in range(0, len(messages), batch_size):
        end = min(start + batch_size, len(messages))
        ctx_start = max(0, start - context_overlap)
        batch = messages[ctx_start:end]

        batch_results = analyze_batch(batch, alias_map=alias_map)

        offset = start - ctx_start
        for j, result in enumerate(batch_results[offset:], start=start):
            if j < len(messages):
                results[j] = result

    return [r if r is not None else _default() for r in results]
