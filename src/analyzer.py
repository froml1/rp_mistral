"""
LLM-based message analyzer — replaces all regex-based content analysis.

Sends batches of Discord messages to Mistral and returns per-message
structured analysis: RP/HRP classification, character attribution,
speaker segmentation, and structural tags.
"""

import json
import re
import sys
from datetime import datetime

import requests


OLLAMA_URL   = "http://localhost:11434/api/generate"
_SCENE_BREAK = re.compile(r'^[\-_*=~]{3,}\s*$')
_PARENS      = re.compile(r'[()]')
_RP_GAP_SECS = 300
_NARRATIVE_WORD = re.compile(r'[a-zA-ZÀ-ÿ]{3,}')


def is_preflight_rp(content: str) -> bool:
    """Étoile hors parenthèses + au moins 4 mots → ouverture de bloc RP garantie."""
    s = content.strip()
    if '*' not in s or _PARENS.search(s):
        return False
    return len(s.split()) > 3


def is_preflight_hrp(content: str) -> bool:
    """
    Preflight HRP — éliminé sans passer par Mistral.
    Priorité : parens > séparateur > absence de mot narratif.
    (L'étoile est testée avant dans purge_export.)
    """
    s = content.strip()
    if not s or _SCENE_BREAK.match(s):
        return True
    if _PARENS.search(s):
        return True
    if not _NARRATIVE_WORD.search(s):
        return True
    return False


def _parse_ts(ts: str) -> datetime | None:
    if not ts:
        return None
    try:
        return datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except ValueError:
        return None


def _can_inherit_rp(msg: dict, prev_msg: dict) -> bool:
    content = msg.get("content", "").strip()
    if _SCENE_BREAK.match(content) or _PARENS.search(content):
        return False
    if _SCENE_BREAK.match(prev_msg.get("content", "").strip()):
        return False
    ts_cur  = _parse_ts(msg.get("timestamp", ""))
    ts_prev = _parse_ts(prev_msg.get("timestamp", ""))
    if not (ts_cur and ts_prev):
        return False
    return abs((ts_cur - ts_prev).total_seconds()) <= _RP_GAP_SECS


def pre_classify_messages(messages: list[dict], seed_msg: dict | None = None) -> list[str]:
    """
    Fast heuristic pass — no LLM.
    Returns per-message: 'rp', 'non_rp', or 'uncertain'.
    seed_msg: last confirmed-RP message from a previous chunk.
    """
    statuses: list[str] = []
    prev_msg: dict | None = seed_msg
    prev_is_rp: bool = seed_msg is not None

    for msg in messages:
        content = msg.get("content", "").strip()
        if is_preflight_hrp(content):
            statuses.append("non_rp")
            prev_is_rp = False
        elif prev_is_rp and prev_msg is not None and _can_inherit_rp(msg, prev_msg):
            statuses.append("rp")
            prev_is_rp = True
        else:
            statuses.append("uncertain")
            prev_is_rp = False
        prev_msg = msg

    return statuses


def extend_rp_chain(messages: list[dict], statuses: list[str]) -> list[str]:
    """
    After LLM results are merged, cascade the RP chain:
    any 'uncertain' following a confirmed 'rp' that meets continuity → 'rp'.
    Repeats until stable.
    """
    statuses = list(statuses)
    changed = True
    while changed:
        changed = False
        for i in range(1, len(messages)):
            if statuses[i] == "uncertain" and statuses[i - 1] == "rp":
                if _can_inherit_rp(messages[i], messages[i - 1]):
                    statuses[i] = "rp"
                    changed = True
    return statuses


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

def _parse_classify_json(raw: str) -> dict:
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        items = []
        for m in re.finditer(
            r'\{[^}]*"index"\s*:\s*(\d+)[^}]*"is_rp"\s*:\s*(true|false)[^}]*"is_ooc"\s*:\s*(true|false)[^}]*\}',
            raw,
        ):
            items.append({"index": int(m.group(1)), "is_rp": m.group(2) == "true", "is_ooc": m.group(3) == "true"})
        if not items:
            for m in re.finditer(
                r'\{[^}]*"index"\s*:\s*(\d+)[^}]*"is_ooc"\s*:\s*(true|false)[^}]*"is_rp"\s*:\s*(true|false)[^}]*\}',
                raw,
            ):
                items.append({"index": int(m.group(1)), "is_rp": m.group(3) == "true", "is_ooc": m.group(2) == "true"})
        return {"classifications": items}


_CLASSIFY_PROMPT = """\
Pour chaque message, indique s'il fait partie d'une scène de roleplay (is_rp: true) ou non (is_rp: false).
Un commentaire méta entre ((...)) ou [OOC...] dans une scène est is_ooc: true (et is_rp: true).

Critères RP : actions entre *astérisques*, dialogues narratifs, descriptions de scène, style littéraire.
Critères HRP : bavardage informel, liens, emojis excessifs, réactions courtes.

Retourne exactement ce JSON :
{{"classifications": [{{"index": 0, "is_rp": true, "is_ooc": false}}]}}

Messages :
{messages}"""


def classify_messages_batched(
    messages: list[dict],
    batch_size: int = BATCH_SIZE,
    context_overlap: int = 3,
):
    """
    Generator — yields (start, end, results) for each batch processed.
    results is a list of {is_rp, is_ooc} for messages[start:end].
    """
    if not messages:
        return

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
            raw = resp.json().get("response", "{}")
            data = _parse_classify_json(raw)
            by_index = {item["index"]: item for item in data.get("classifications", [])}
        except Exception as e:
            print(f"  [analyzer] classify error: {e}", file=sys.stderr)
            by_index = {}

        offset = start - ctx_start
        batch_results = []
        for j in range(end - start):
            r = by_index.get(offset + j, {})
            batch_results.append({
                "is_rp":  bool(r.get("is_rp", False)),
                "is_ooc": bool(r.get("is_ooc", False)),
            })
        yield start, end, batch_results


def classify_messages(
    messages: list[dict],
    batch_size: int = BATCH_SIZE,
    context_overlap: int = 3,
) -> list[dict]:
    """
    Lightweight RP/HRP classification — only returns {is_rp, is_ooc} per message.
    Used by the purger; cheaper and more reliable than full analysis.
    """
    results: list[dict | None] = [None] * len(messages)
    for start, end, batch_results in classify_messages_batched(messages, batch_size, context_overlap):
        for j, r in enumerate(batch_results):
            results[start + j] = r
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
