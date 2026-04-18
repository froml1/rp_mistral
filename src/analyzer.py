"""
LLM-based message analyzer — replaces all regex-based content analysis.

Sends batches of Discord messages to Mistral and returns per-message
structured analysis: RP/HRP classification, character attribution,
speaker segmentation, and structural tags.
"""

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
import os, time
import re
import sys
from datetime import datetime

import requests
OLLAMA_URL   = "http://localhost:11434/api/generate"
_SCENE_BREAK = re.compile(r'^[\-_*=~]{3,}\s*$')
_PARENS      = re.compile(r'[()]')
_RP_GAP_SECS = 300
_NARRATIVE_WORD = re.compile(r'[a-zA-ZÀ-ÿ]{3,}')
_STAR_CONTENT   = re.compile(r'\*([^*]+)\*')

# ── Preprocessing for LLM input ───────────────────────────────────────────────
_MENTION_RE      = re.compile(r'<@!?\d+>')
_CUSTOM_EMOJI_RE = re.compile(r'<:\w+:\d+>')
_URL_RE          = re.compile(r'https?://\S+')
_UNICODE_EMOJI_RE = re.compile(
    "[\U0001F300-\U0001F9FF\U00002700-\U000027BF\U0001FA00-\U0001FA6F\U0001F004\U0001F0CF]+",
    flags=re.UNICODE,
)
_PUNCT_REPEAT_RE = re.compile(r'([!?])\1{2,}')
_SPEAKER_LINE_RE = re.compile(r'^\[([^\]]+)\]\s*(.*)')


def preprocess_for_llm(text: str) -> str:
    """Strip Discord artifacts, emojis, repeated punctuation, extra whitespace."""
    text = _MENTION_RE.sub("", text)
    text = _CUSTOM_EMOJI_RE.sub("", text)
    text = _URL_RE.sub("", text)
    text = _UNICODE_EMOJI_RE.sub("", text)
    text = _PUNCT_REPEAT_RE.sub(r'\1', text)
    return re.sub(r'\s+', ' ', text).strip()


def compress_scene_text(text: str) -> str:
    """Preprocess scene text: clean each line + merge consecutive same-speaker lines."""
    merged: list[str] = []
    for raw_line in text.splitlines():
        line = preprocess_for_llm(raw_line)
        if not line:
            continue
        m = _SPEAKER_LINE_RE.match(line)
        if m and merged:
            prev_m = _SPEAKER_LINE_RE.match(merged[-1])
            if prev_m and prev_m.group(1) == m.group(1):
                merged[-1] = f"[{m.group(1)}] {prev_m.group(2)} {m.group(2)}"
                continue
        merged.append(line)
    return "\n".join(merged)

def is_preflight_rp(content: str) -> bool:
    """
    Candidat à l'ouverture de bloc RP :
    - pas de parenthèses
    - au moins une paire *...* dont le contenu interne dépasse 3 mots
    """
    s = content.strip()
    if _PARENS.search(s):
        return False
    matches = _STAR_CONTENT.findall(s)
    if not matches:
        return False
    return any(len(m.split()) > 3 for m in matches)


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
BATCH_SIZE = 12

_OPENER_SYSTEM = (
    "Tu es un classificateur de messages pour jeu de rôle narratif sur Discord. "
    "Tu retournes UNIQUEMENT du JSON valide, sans texte supplémentaire."
)

_OPENER_PROMPT = """\
{context_block}Message candidat à l'ouverture :
{content}

Réponds à deux questions :
1. is_opener : ce message est-il une ouverture RP de style littéraire ?
   (action narrative entre *astérisques*, style descriptif, peut ouvrir une scène de façon autonome)
2. is_new_scene : si is_opener est true, s'agit-il d'une NOUVELLE scène ou d'une REPRISE de la scène précédente ?
   - Nouvelle scène : personnages différents, lieu différent, ou rupture thématique claire
   - Reprise : mêmes personnages, même lieu, continuité narrative directe

Retourne uniquement : {{"is_opener": true/false, "is_new_scene": true/false}}"""


def classify_opener(content: str, prev_context: list[str] | None = None) -> tuple[bool, bool]:
    """
    Demande à Mistral si le message est une ouverture RP littéraire.
    Retourne (is_opener, is_new_scene).
    prev_context : derniers messages de la scène précédente (pour détecter les reprises).
    """
    content = preprocess_for_llm(content)
    if prev_context:
        ctx = "Contexte de la scène précédente :\n"
        ctx += "\n".join(f"  {preprocess_for_llm(m)}" for m in prev_context[-5:])
        ctx += "\n\n"
    else:
        ctx = ""

    prompt = _OPENER_SYSTEM + "\n\n" + _OPENER_PROMPT.format(
        context_block=ctx, content=content
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
        raw  = resp.json().get("response", "{}")
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            is_op  = re.search(r'"is_opener"\s*:\s*(true|false)', raw)
            is_ns  = re.search(r'"is_new_scene"\s*:\s*(true|false)', raw)
            data   = {
                "is_opener":    is_op.group(1) == "true" if is_op else False,
                "is_new_scene": is_ns.group(1) == "true" if is_ns else True,
            }
        is_opener    = bool(data.get("is_opener", False))
        is_new_scene = bool(data.get("is_new_scene", True))
        return is_opener, is_new_scene
    except Exception as e:
        print(f"  [analyzer] opener error: {e}", file=sys.stderr)
        return False, True

MSG_TAG_VOCAB = ["action", "dialogue", "descriptif", "pensée", "nsfw_hint"]

_SYSTEM = (
    "Tu es un analyseur de messages pour jeu de rôle narratif sur Discord. "
    "Tu retournes UNIQUEMENT du JSON valide, sans texte supplémentaire."
)

_PROMPT = """\
Alias : {aliases}

Classifie chaque message Discord.

Retourne pour chaque message :
- characters : personnages actifs
- referenced : personnages mentionnés
- speaker_segments : découpe simple texte → action / dialogue / narration
- tags : max 2 parmi {vocab}

Règles :
- alias → nom canonique
- speaker_segments = découpage léger (pas fin)

Retour JSON :
{{"messages":[{{"index":0,"characters":[],"referenced":[],"speaker_segments":[{{"text":"","character":null,"type":"narration"}}],"tags":[]}}]}}

Messages :
{messages}
"""


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
        f"[{i}] {_get_author(m)}: {preprocess_for_llm(m.get('content', ''))}"
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
            json={"model": LLM_MODEL, "prompt": prompt, "format": "json", "stream": False, "options": {
            "temperature": 0,
            "top_k": 1,
            "num_predict": 250,
            "num_ctx": 4096
        }},
            timeout=120,
        )
        resp.raise_for_status()
        raw = resp.json().get("response", "{}")
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            data = _parse_analyze_json(raw)
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
                if isinstance(s, dict)
            ],
            "tags": [t for t in (r.get("tags") or []) if t in MSG_TAG_VOCAB],
        })
    return output


def _parse_analyze_json(raw: str) -> dict:
    """Fallback regex quand Mistral retourne du JSON malformé pour analyze_batch."""
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        items = []
        for m in re.finditer(r'"index"\s*:\s*(\d+)', raw):
            idx   = int(m.group(1))
            chunk = raw[m.start(): m.start() + 400]
            is_rp  = re.search(r'"is_rp"\s*:\s*(true|false)', chunk)
            is_ooc = re.search(r'"is_ooc"\s*:\s*(true|false)', chunk)
            chars  = re.findall(r'"characters"\s*:\s*\[([^\]]*)\]', chunk)
            item = {
                "index":      idx,
                "is_rp":      is_rp.group(1) == "true"  if is_rp  else False,
                "is_ooc":     is_ooc.group(1) == "true" if is_ooc else False,
                "characters": re.findall(r'"([^"]+)"', chars[0]) if chars else [],
            }
            items.append(item)
        return {"messages": items}


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
                json={"model": LLM_MODEL, "prompt": prompt, "format": "json", "stream": False, "options": {
            "temperature": 0,
            "top_k": 1,
            "num_predict": 250,
            "num_ctx": 4096
        }},
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

def worker_analyze(messages, start, ctx_start, batch, alias_map):
    """
    Process one batch in a thread.
    Returns list of tuples: (global_index, result)
    """
    print(f"Analyzing... {start}-{start + BATCH_SIZE}/{len(messages)}")
    start_time = time.perf_counter()
    batch_results = analyze_batch(batch, alias_map=alias_map)
    end_time = time.perf_counter()
    execution_time = end_time - start_time
    offset = start - ctx_start
    out = []
    print(f"End Analyzing... {start}-{start + BATCH_SIZE}/{len(messages)}. Programme exécuté en : {execution_time: .5f} secondes")
    for j, result in enumerate(batch_results[offset:], start=start):
        if j < len(messages):
            out.append((j, result))

    return out


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
    count = 1
    batches = []
    jobs = []
    for start in range(0, len(messages), batch_size):
        count += 1
        end = min(start + batch_size, len(messages))
        ctx_start = max(0, start - context_overlap)
        batch = messages[ctx_start:end]
        batches.append(batch)
        jobs.append((messages, start, ctx_start, batch, alias_map))

    max_workers = min(1, os.cpu_count() or 1)
    futures = []
    all_outputs  = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(worker_analyze, *job) for job in jobs]
        all_outputs = [future.result() for future in futures]
    
    for output in all_outputs:
        for idx, value in output:
            results[idx] = value

    return [r if r is not None else _default() for r in results]
