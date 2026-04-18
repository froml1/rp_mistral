"""
RP/HRP separator — pre-processing pass before indexing.

Reads raw Discord exports from data/exports/
Writes filtered exports (RP only) to data/exports_filtered/

Detection strategy (in confidence order):
  1. Structural  — separator lines (---/___/***) mark scene boundaries
  2. Heuristic   — emoji density, URLs, informal markers score each message
  3. Temporal    — long gap after a scene signals possible end
  4. LLM         — Mistral classifies ambiguous windows (optional, --with-llm)

Usage:
  python src/purger.py [data/exports/] [--with-llm]
"""

import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import requests
import yaml


# ── Config ────────────────────────────────────────────────────────────────────

OLLAMA_URL = "http://localhost:11434/api/generate"
LLM_MODEL = "mistral"
EXPORTS_DIR = Path("data/exports")
OUTPUT_DIR = Path("data/exports_filtered")
SCENE_GAP_MINUTES = 45       # gap beyond which we consider a scene may have ended
AMBIGUOUS_BATCH = 15         # messages per LLM classification batch


# ── Structural patterns ───────────────────────────────────────────────────────

# A message that is purely a separator line
SEPARATOR_RE = re.compile(r'^\s*[-_*=~—]{3,}\s*$')

# RP signals
ACTION_RE = re.compile(r'\*[^*]+\*')
DIALOGUE_RE = re.compile(r'(?:^|\s)[—«»]|\*{3}|_{3}')
OOC_INLINE_RE = re.compile(r'^\s*\(.*\)\s*$', re.DOTALL)

# Non-RP signals
URL_RE = re.compile(r'https?://\S+')
EMOJI_RE = re.compile(
    "[\U0001F300-\U0001F9FF\U00002500-\U00002BEF\U00002702-\U000027B0"
    "\U0000FE00-\U0000FE0F\U0001F000-\U0001F02F]+",
    flags=re.UNICODE,
)
INFORMAL_RE = re.compile(
    r'\b(lol|mdr|xd|ptdr|ahah|haha|hehe|héhé|omg|wtf|ouf|wsh|bref|nan|'
    r'ouais|ouaip|okay|ok|yep|nope|gg|bg|np|rip|smh|nvm|fyi|irl|afk)\b',
    re.IGNORECASE,
)


# ── Heuristic scorer ──────────────────────────────────────────────────────────

def rp_score(content: str) -> float:
    """
    Returns a float in [-1, +1].
    Positive = likely RP, negative = likely non-RP, ~0 = ambiguous.
    """
    score = 0.0

    if not content or not content.strip():
        return -0.5

    # Strong non-RP
    if URL_RE.search(content):
        score -= 0.6
    emoji_count = len(EMOJI_RE.findall(content))
    if emoji_count >= 3:
        score -= 0.6
    elif emoji_count >= 1:
        score -= 0.2
    if INFORMAL_RE.search(content):
        score -= 0.3
    if OOC_INLINE_RE.match(content):
        score -= 0.4
    if len(content.strip()) < 20 and not ACTION_RE.search(content):
        score -= 0.2

    # Strong RP
    if ACTION_RE.search(content):
        score += 0.6
    if DIALOGUE_RE.search(content):
        score += 0.3
    if len(content) > 120:
        score += 0.2

    return max(-1.0, min(1.0, score))


def is_separator(content: str) -> bool:
    return bool(SEPARATOR_RE.match(content.strip()))


def parse_ts(timestamp: str) -> datetime | None:
    try:
        return datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
    except (ValueError, AttributeError):
        return None


def gap_minutes(ts_a: str, ts_b: str) -> float:
    a, b = parse_ts(ts_a), parse_ts(ts_b)
    if a and b:
        return abs((b - a).total_seconds()) / 60
    return 0.0


# ── LLM classifier ────────────────────────────────────────────────────────────

LLM_SYSTEM = (
    "Tu es un classificateur de messages Discord pour un jeu de rôle narratif. "
    "Tu retournes UNIQUEMENT du JSON valide, sans texte supplémentaire."
)

LLM_PROMPT = """\
Voici une séquence de messages Discord. Pour chaque message (identifié par son index),
détermine s'il fait partie d'une scène de roleplay (rp) ou s'il est hors-rp (hrp).

Critères RP : actions entre *astérisques*, dialogues narratifs, descriptions de scène, style littéraire.
Critères HRP : conversation informelle, liens, emojis excessifs, commentaires méta, réactions courtes.

Retourne exactement ce JSON :
{{"classifications": [{{"index": 0, "label": "rp"}}, {{"index": 1, "label": "hrp"}}, ...]}}

Messages :
{messages}"""


def llm_classify_batch(messages: list[dict]) -> dict[int, str]:
    """
    Sends a batch of messages to Mistral for RP/HRP classification.
    Returns {original_index: "rp"|"hrp"}.
    """
    formatted = "\n".join(
        f'[{i}] {m["author"]}: {m["content"][:200]}'
        for i, m in enumerate(messages)
    )
    prompt = LLM_SYSTEM + "\n\n" + LLM_PROMPT.format(messages=formatted)
    try:
        resp = requests.post(
            OLLAMA_URL,
            json={"model": LLM_MODEL, "prompt": prompt, "format": "json", "stream": False},
            timeout=90,
        )
        resp.raise_for_status()
        data = json.loads(resp.json().get("response", "{}"))
        return {item["index"]: item["label"] for item in data.get("classifications", [])}
    except Exception as e:
        print(f"  [purger] LLM error: {e}", file=sys.stderr)
        return {}


# ── Main classification pass ──────────────────────────────────────────────────

RP = "rp"
HRP = "hrp"
AMBIGUOUS = "ambiguous"


def classify_messages(messages: list[dict]) -> list[str]:
    """
    Returns a label list (RP | HRP | AMBIGUOUS) for each message, same length as input.

    Pass 1 — structural: separators set scene_active flag.
    Pass 2 — heuristic scoring within/outside scene blocks.
    Ambiguous messages are those with score in (-0.3, +0.3) that don't benefit
    from structural context.
    """
    n = len(messages)
    labels = [AMBIGUOUS] * n
    scene_active = False

    for i, msg in enumerate(messages):
        content = msg.get("content", "").strip()

        # Separator line → toggle scene mode, label as HRP (it's a marker, not content)
        if is_separator(content):
            scene_active = not scene_active
            labels[i] = HRP
            continue

        # Check temporal gap from previous message
        if scene_active and i > 0:
            gap = gap_minutes(
                messages[i - 1].get("timestamp", ""),
                msg.get("timestamp", ""),
            )
            if gap > SCENE_GAP_MINUTES:
                scene_active = False

        score = rp_score(content)

        if scene_active:
            # Inside a scene: trust structure, only override on strong non-RP signal
            if score < -0.5:
                labels[i] = HRP
            else:
                labels[i] = RP
        else:
            # Outside a scene: trust heuristics
            if score >= 0.3:
                labels[i] = RP
                scene_active = True   # unannounced scene start
            elif score <= -0.3:
                labels[i] = HRP
            else:
                labels[i] = AMBIGUOUS

    return labels


def resolve_ambiguous_with_llm(
    messages: list[dict], labels: list[str]
) -> list[str]:
    """Sends ambiguous windows to Mistral and updates labels in-place."""
    ambiguous_indices = [i for i, l in enumerate(labels) if l == AMBIGUOUS]
    if not ambiguous_indices:
        return labels

    updated = list(labels)

    # Process in batches with a small context window
    for batch_start in range(0, len(ambiguous_indices), AMBIGUOUS_BATCH):
        batch_idx = ambiguous_indices[batch_start: batch_start + AMBIGUOUS_BATCH]
        # Include a few surrounding messages for context
        context_start = max(0, batch_idx[0] - 3)
        context_end = min(len(messages), batch_idx[-1] + 4)
        context_msgs = messages[context_start:context_end]

        result = llm_classify_batch(context_msgs)

        for local_i, label in result.items():
            global_i = context_start + local_i
            if global_i in batch_idx:   # only update the truly ambiguous ones
                updated[global_i] = RP if label == "rp" else HRP

        # Remaining still-ambiguous → default to HRP (conservative)
        for i in batch_idx:
            if updated[i] == AMBIGUOUS:
                updated[i] = HRP

    return updated


# ── File processing ───────────────────────────────────────────────────────────

def purge_export(filepath: Path, with_llm: bool = False, verbose: bool = True) -> dict:
    with open(filepath, encoding="utf-8") as f:
        data = json.load(f)

    raw_messages = data.get("messages", data) if isinstance(data, dict) else data
    is_wrapped = isinstance(data, dict) and "messages" in data

    labels = classify_messages(raw_messages)

    if with_llm:
        ambiguous_count = labels.count(AMBIGUOUS)
        if ambiguous_count and verbose:
            print(f"  [purger] {ambiguous_count} ambiguous → sending to Mistral…")
        labels = resolve_ambiguous_with_llm(raw_messages, labels)
    else:
        # Without LLM, default ambiguous to HRP
        labels = [HRP if l == AMBIGUOUS else l for l in labels]

    rp_messages = [msg for msg, label in zip(raw_messages, labels) if label == RP]

    if verbose:
        total = len(raw_messages)
        kept = len(rp_messages)
        dropped = total - kept
        print(f"  {filepath.name}: {total} messages → {kept} RP kept, {dropped} HRP dropped")

    return {"messages": rp_messages} if is_wrapped else rp_messages


def purge_all(exports_dir: str = "data/exports", with_llm: bool = False):
    input_path = Path(exports_dir)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    json_files = list(input_path.glob("**/*.json"))
    if not json_files:
        print(f"No JSON files found in {exports_dir}")
        return

    print(f"Purging {len(json_files)} file(s) → {OUTPUT_DIR}/")
    for filepath in json_files:
        filtered = purge_export(filepath, with_llm=with_llm)
        out_path = OUTPUT_DIR / filepath.name
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(filtered, f, ensure_ascii=False, indent=2)

    print(f"Done. Filtered exports in {OUTPUT_DIR}/")
    print("→ Run indexer on filtered exports:")
    print(f"  python src/indexer.py {OUTPUT_DIR}/")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    exports_dir = next((a for a in sys.argv[1:] if not a.startswith("--")), "data/exports")
    with_llm = "--with-llm" in sys.argv
    purge_all(exports_dir, with_llm=with_llm)
