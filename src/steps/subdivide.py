"""
Step 3 - Subdivide: LLM coherence split + heuristic RP pre-filter.

For each scene file from translate:
  1. LLM detects all scene boundaries in one pass → N sub-scenes
  2. Drop duplicate opening messages vs previous scene
  3. Trim OOC prefix (heuristic + LLM for borderline cases)
  4. Hard-reject only extreme non-RP (score < 0.15 AND no_narrative flag)
  5. Write surviving sub-scenes with rp_confidence metadata
  6. rp_filter.py (LLM + lore context) is the authoritative RP gate downstream
  7. Append hard-rejected scenes to rp_report.json for manual review
"""

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from llm import call_llm_json

_SPLIT_PROMPT = """\
You are analyzing a sequence of RP messages. Find ALL scene boundaries.

A NEW scene starts when ANY of the following occurs:
- The active character group changes (different characters now on stage)
- The location changes (they move elsewhere, or a new group appears elsewhere)
- A clear narrative break: timeskip, topic shift, end of one arc and start of another
- Parallel scenes: group A in location X while group B is simultaneously in location Y

Return the index of the FIRST message of each new scene.
If this is already ONE continuous scene, return an empty list.

JSON: {{"boundaries": [5, 12, 23], "reasons": ["char group split", "location change", "timeskip"]}}
  or: {{"boundaries": []}} if single scene

Messages (first 150 chars each):
{messages}"""

_MANIFEST        = "_manifest.json"
MIN_MESSAGES     = 20
_HARD_REJECT_SCORE = 0.15   # below this AND no_narrative → hard reject without LLM review

# ── OOC / meta-RP patterns ────────────────────────────────────────────────────

_OOC_PATTERNS = re.compile(
    r'\b(lol|lmao|mdr|xd|haha|hihi|ahah|irl|ooc|brb|afk|ok\s|okay|'
    r'ce\s+soir|demain|ce\s+week|disponible|dispo|pas\s+là|absent|'
    r'on\s+joue|on\s+reporte|on\s+reprend|prêt\b|ready\b|'
    r'désolé\b|sorry\b|connexion|internet|reboot|bug\s+technique)\b',
    re.IGNORECASE,
)

# Players commenting on the fiction from outside — not being their character (C)
_META_RP_PATTERNS = re.compile(
    r'\b(comme personnage|en tant que|la scène (où|d[eu])|'
    r'vous avez (joué|fait)|cette session|le RP (d[eu]|hier)|'
    r"incroyable comme|j'ai adoré|c'était (bien|super|génial|trop)|"
    r'trop (bien|bon|fort) (ce|cette|le|la)|cette scène (était|a été))\b',
    re.IGNORECASE,
)

# Prefix decision thresholds (D)
_KEEP_PREFIX_MIN_LEN  = 45    # avg char length above which prefix may be in-character dialogue
_KEEP_PREFIX_MAX_OOC  = 0.25  # ooc ratio below which prefix may be in-character dialogue
_CLEARLY_OOC_AVG_LEN  = 25    # avg len below which prefix is certainly coordination/OOC
_CLEARLY_OOC_RATIO    = 0.35  # ooc ratio above which prefix is certainly OOC


# ── Heuristic RP check ────────────────────────────────────────────────────────

def _is_narrative_asterisk(inner: str) -> bool:
    """True if text between asterisks looks like narrative action, not casual emphasis."""
    words = re.findall(r'[a-zA-ZÀ-ÿ]{3,}', inner)
    return len(inner) >= 25 and len(set(words)) >= 2


def _heuristic_rp_check(messages: list[dict]) -> tuple[bool, float, list[str]]:
    """
    Score RP quality from observable text signals. No LLM.

      + narrative *asterisk actions*  → primary signal, casual emphasis ignored (B)
      + avg message length            → narrative depth
      + quoted dialogue               → in-character speech
      - OOC keyword density           → meta/casual talk
      - meta-RP commentary density    → players commenting on fiction, not playing it (C)
    """
    contents = [
        (m.get("content_en") or m.get("content") or "").strip()
        for m in messages
    ]
    contents = [c for c in contents if c]
    if not contents:
        return False, 0.0, ["no_narrative"]

    n         = len(contents)
    full_text = "\n".join(contents)

    # Count only narrative asterisks — casual *emphasis* does not count (B)
    narrative_msgs = sum(
        1 for c in contents
        if any(_is_narrative_asterisk(m.group(1))
               for m in re.finditer(r'\*([^*]+)\*', c))
    )
    narrative_ratio = narrative_msgs / n
    avg_len         = sum(len(c) for c in contents) / n
    has_quotes      = bool(re.search(r'[«»""][^"«»""]{5,}[«»""]', full_text))
    ooc_msgs        = sum(1 for c in contents if _OOC_PATTERNS.search(c))
    ooc_ratio       = ooc_msgs / n
    meta_msgs       = sum(1 for c in contents if _META_RP_PATTERNS.search(c))
    meta_ratio      = meta_msgs / n

    score = 0.0

    if narrative_ratio >= 0.4:    score += 0.55
    elif narrative_ratio >= 0.15: score += 0.35
    elif narrative_ratio >= 0.05: score += 0.15

    if avg_len >= 120:   score += 0.25
    elif avg_len >= 60:  score += 0.15
    elif avg_len >= 30:  score += 0.05

    if has_quotes: score += 0.10

    if ooc_ratio >= 0.5:    score -= 0.40
    elif ooc_ratio >= 0.25: score -= 0.20
    elif ooc_ratio >= 0.10: score -= 0.05

    # Meta-RP: observers of fiction score as non-RP even with some asterisks (C)
    if meta_ratio >= 0.3 and ooc_ratio >= 0.15:
        score -= 0.35
    elif meta_ratio >= 0.2:
        score -= 0.15
    elif meta_ratio >= 0.1:
        score -= 0.05

    score  = round(max(0.0, min(1.0, score)), 3)
    is_rp  = score >= 0.5

    flags: list[str] = []
    if ooc_ratio >= 0.3:                        flags.append("ooc")
    if meta_ratio >= 0.2:                       flags.append("meta_rp")
    if narrative_ratio == 0 and avg_len < 80:   flags.append("no_narrative")
    if avg_len < 40 and not flags:              flags.append("too_casual")

    return is_rp, score, flags


# ── Duplicate-opening remover ────────────────────────────────────────────────

def _normalize(text: str) -> str:
    return re.sub(r'\s+', ' ', text.strip().lower())


def _drop_duplicate_opening(messages: list[dict], prev_messages: list[dict],
                            check_start: int = 10) -> list[dict]:
    """
    Remove messages at the start of `messages` that duplicate any message
    from the entire previous scene.
    Stops at the first non-duplicate so only a contiguous opening prefix is removed.
    """
    if not prev_messages or not messages:
        return messages

    prev_set = {
        _normalize(m.get("content_en") or m.get("content", ""))
        for m in prev_messages
    }
    prev_set.discard("")

    drop = 0
    for m in messages[:check_start]:
        t = _normalize(m.get("content_en") or m.get("content", ""))
        if t and t in prev_set:
            drop += 1
        else:
            break

    if drop:
        print(f"    [dedup] {drop} repeated opening message(s) removed")
        return messages[drop:]
    return messages


# ── Scene text formatter (shared by split + prefix LLM) ──────────────────────

def _scene_text(messages: list[dict]) -> str:
    lines = []
    for i, msg in enumerate(messages):
        author  = msg.get("author", {})
        name    = author.get("name", "?") if isinstance(author, dict) else str(author)
        content = (msg.get("content_en") or msg.get("content", ""))[:150]
        lines.append(f"[{i}] {name}: {content}")
    return "\n".join(lines)


# ── OOC prefix trimmer ───────────────────────────────────────────────────────

_CONTEXT_CHECK_PROMPT = """\
Two groups of messages from a Discord RP channel.

GROUP A — before the first narrative action:
{prefix}

GROUP B — starts with a narrative RP action (*...*):
{rp_open}

Are these two groups part of the SAME continuous scene (same characters, same narrative flow, \
GROUP A is in-universe dialogue leading naturally into GROUP B)?
Or is GROUP A out-of-character player discussion that happened BEFORE GROUP B's scene started?

JSON: {{"same_scene": true}} or {{"same_scene": false}}"""


def _prefix_verdict(prefix: list[dict]) -> str:
    """
    Return 'keep', 'trim', or 'ask_llm' for the OOC prefix decision (D).

    'keep'     → long messages, low OOC → likely in-character dialogue
    'trim'     → short (coordination) OR high OOC → clearly not in-character
    'ask_llm'  → borderline length + OOC mix → delegate to LLM
    """
    if not prefix:
        return "trim"
    contents = [(m.get("content_en") or m.get("content", "")).strip() for m in prefix]
    contents = [c for c in contents if c]
    if not contents:
        return "trim"
    avg_len   = sum(len(c) for c in contents) / len(contents)
    ooc_ratio = sum(1 for c in contents if _OOC_PATTERNS.search(c)) / len(contents)

    if avg_len >= _KEEP_PREFIX_MIN_LEN and ooc_ratio < _KEEP_PREFIX_MAX_OOC:
        return "keep"
    if avg_len < _CLEARLY_OOC_AVG_LEN or ooc_ratio >= _CLEARLY_OOC_RATIO:
        return "trim"
    if len(prefix) >= 2:
        return "ask_llm"
    return "trim"


def _trim_rp_prefix(messages: list[dict]) -> list[dict]:
    """
    Cut OOC prefix before the first genuine *action* (15+ chars).
    Uses heuristic verdict; calls LLM only for borderline cases (D+E).
    """
    if not messages:
        return messages
    anchor = next(
        (i for i, m in enumerate(messages)
         if re.search(r'\*[^*]{15,}\*', m.get("content_en") or m.get("content", ""))),
        None,
    )
    if not anchor:
        return messages

    prefix  = messages[:anchor]
    verdict = _prefix_verdict(prefix)

    if verdict == "keep":
        return messages

    if verdict == "ask_llm":
        rp_open = messages[anchor: anchor + 5]
        result  = call_llm_json(
            _CONTEXT_CHECK_PROMPT.format(
                prefix=_scene_text(prefix),
                rp_open=_scene_text(rp_open),
            ),
            num_predict=50,
        )
        if result.get("same_scene", False):
            print(f"    [trim] prefix kept — LLM: same scene")
            return messages
        print(f"    [trim] {anchor} prefix message(s) cut — LLM: OOC")
        return messages[anchor:]

    # verdict == "trim"
    print(f"    [trim] {anchor} prefix message(s) cut")
    return messages[anchor:]


# ── LLM split ─────────────────────────────────────────────────────────────────

def _is_valid_json(path: Path) -> bool:
    try:
        json.loads(path.read_text(encoding="utf-8"))
        return True
    except Exception:
        return False


def _split(messages: list[dict]) -> list[list[dict]]:
    if len(messages) < 4:
        return [messages]

    data    = call_llm_json(
        _SPLIT_PROMPT.format(messages=_scene_text(messages)),
        num_predict=200,
    )
    raw     = data.get("boundaries") or []
    reasons = data.get("reasons") or []
    bounds  = sorted(set(b for b in raw if isinstance(b, int) and 1 <= b < len(messages)))

    if not bounds:
        return [messages]

    for i, b in enumerate(bounds):
        reason = reasons[i] if i < len(reasons) else ""
        print(f"    split [{b}]" + (f" — {reason}" if reason else ""))

    segments, prev = [], 0
    for b in bounds:
        segments.append(messages[prev:b])
        prev = b
    segments.append(messages[prev:])
    return [s for s in segments if s]


# ── Manifest helpers ──────────────────────────────────────────────────────────

def _load_manifest(out_dir: Path) -> dict:
    mf = out_dir / _MANIFEST
    if mf.exists():
        try:
            return json.loads(mf.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}


def _save_manifest(out_dir: Path, manifest: dict):
    (out_dir / _MANIFEST).write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )


# ── Main ──────────────────────────────────────────────────────────────────────

def run_subdivide(translated_dir: Path, out_dir: Path,
                  purged_dir: Path | None = None,
                  report_path: Path | None = None) -> list[Path]:
    """
    Split each translated scene file by narrative coherence.

    Hard-reject only extreme non-RP (score < 0.15 AND no_narrative flag).
    Everything else passes through with rp_confidence metadata so that
    rp_filter.py (LLM + lore context) can make the authoritative call.
    """
    scene_files = sorted(translated_dir.glob("**/*.json"))
    if not scene_files:
        print(f"  No translated scene files found in {translated_dir}")
        return []

    if report_path is None:
        report_path = out_dir.parent / "rp_report.json"

    rp_report: dict = {}
    if report_path.exists() and _is_valid_json(report_path):
        rp_report = json.loads(report_path.read_text(encoding="utf-8"))

    produced:      list[Path] = []
    prev_messages: list[dict] = []

    for fp in scene_files:
        stem       = fp.parent.name
        out_subdir = out_dir / stem
        out_subdir.mkdir(parents=True, exist_ok=True)
        manifest   = _load_manifest(out_subdir)

        if fp.stem in manifest:
            for sid in manifest[fp.stem]:
                sp = out_subdir / f"{sid}.json"
                if sp.exists():
                    produced.append(sp)
                    try:
                        data = json.loads(sp.read_text(encoding="utf-8"))
                        prev_messages = data.get("messages", [])
                    except Exception:
                        pass
            continue

        if not _is_valid_json(fp):
            print(f"  [error] {fp.name} is malformed, skipping")
            continue

        with open(fp, encoding="utf-8") as f:
            scene = json.load(f)

        messages = scene.get("messages", [])
        source   = scene.get("source", fp.name)
        print(f"  {fp.name} ({len(messages)} msgs)", end="")

        sub_scenes = _split(messages)

        kept_ids: list[str] = []
        next_idx = max(
            (int(p.stem.rsplit("_", 1)[-1])
             for p in out_subdir.glob("*.json")
             if p.name != _MANIFEST and p.stem.rsplit("_", 1)[-1].isdigit()),
            default=-1,
        ) + 1

        accepted = rejected_rp = low_conf = 0
        local_prev_messages = prev_messages

        for sub in sub_scenes:
            sub = _drop_duplicate_opening(sub, local_prev_messages)
            sub = _trim_rp_prefix(sub)

            is_rp, rp_score, rp_flags = _heuristic_rp_check(sub)

            # Hard-reject only when there is absolutely no narrative signal (A)
            if rp_score < _HARD_REJECT_SCORE and "no_narrative" in rp_flags:
                rejected_rp += 1
                tmp_id = f"{stem}_{next_idx:03d}"
                rp_report[tmp_id] = {
                    "source":   source,
                    "rp_score": rp_score,
                    "rp_flags": rp_flags,
                    "n_msgs":   len(sub),
                    "preview":  ((sub[0].get("content_en") or sub[0].get("content", ""))[:120]) if sub else "",
                }
                next_idx += 1
                continue

            if not is_rp:
                low_conf += 1

            scene_id   = f"{stem}_{next_idx:03d}"
            scene_path = out_subdir / f"{scene_id}.json"
            scene_path.write_text(
                json.dumps({
                    "scene_id":      scene_id,
                    "source":        source,
                    "rp_confidence": rp_score,
                    "rp_flags":      rp_flags,
                    "messages":      sub,
                }, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            produced.append(scene_path)
            kept_ids.append(scene_id)
            next_idx += 1
            accepted += 1
            local_prev_messages = sub

        prev_messages = local_prev_messages

        parts = [f" → {accepted} kept"]
        if low_conf:    parts.append(f"{low_conf} low-confidence")
        if rejected_rp: parts.append(f"{rejected_rp} hard-rejected")
        print(",  ".join(parts))

        manifest[fp.stem] = kept_ids
        _save_manifest(out_subdir, manifest)

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(rp_report, ensure_ascii=False, indent=2), encoding="utf-8")
    if rp_report:
        print(f"  [rp report] {len(rp_report)} hard-rejected scene(s) → {report_path}")

    return produced
