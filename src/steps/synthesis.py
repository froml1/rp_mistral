"""
Step 4 — Synthesis: detailed narrative extraction + heuristic RP quality check.
Output: data/lore/lore_how.yaml

For each scene:
  - LLM extracts: narrative (3-4 sentences), characters (name+action), tensions
  - Heuristic (no LLM): is_rp / rp_score / rp_flags based on observable text signals

Incremental: already-processed scenes are skipped on re-run.
synthesis_context_block() formats the collection for prompt injection in step 5 (analyze).
is_scene_rp() is the single source of truth for RP filtering.
"""

import json
import re
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from llm import call_llm_json

LORE_HOW_FILE = "lore_how.yaml"

# OOC patterns — player meta-talk, scheduling, reactions
_OOC_PATTERNS = re.compile(
    r'\b(lol|lmao|mdr|xd|haha|hihi|ahah|irl|ooc|brb|afk|ok\s|okay|'
    r'ce\s+soir|demain|ce\s+week|disponible|dispo|pas\s+là|absent|'
    r'on\s+joue|on\s+reporte|on\s+reprend|prêt\b|ready\b|'
    r'désolé\b|sorry\b|connexion|internet|reboot|bug\s+technique)\b',
    re.IGNORECASE,
)

_PROMPT = """\
Read this scene from a text-based roleplay and produce a detailed narrative summary.

Characters speak in dialogue and perform *actions written between asterisks*.
Be specific — name characters, places, objects. Capture the emotional atmosphere and stakes.

PRIOR SCENES (use to recognize recurring characters and ongoing tensions — do not invent):
{prior_context}

Extract:
- narrative: 3-4 sentences covering what happens (dialogue topics, *physical actions*, key decisions, emotional tone)
- characters: for each active character, what they do, say, or feel in this scene (one sentence each)
- tensions: unresolved conflicts, revelations, emotional stakes, questions left open at scene's end

JSON:
{{
  "narrative": "",
  "characters": [{{"name": "", "action": ""}}],
  "tensions": []
}}

Scene:
---
{text}
---"""


def _heuristic_rp_check(messages: list[dict]) -> tuple[bool, float, list[str]]:
    """
    Determine RP quality from observable text signals — no LLM needed.

    Signals used:
      + *asterisk actions* present              → strongest RP marker
      + average message length > 80 chars       → narrative depth
      + presence of quoted dialogue "..."       → in-character speech
      - high OOC keyword density                → meta/casual talk
      - very short messages (avg < 30 chars)    → chat reactions
      - tiny scene (< 3 messages)               → likely test/meta
    """
    if not messages:
        return False, 0.0, ["no_narrative"]

    contents = [
        (m.get("content_en") or m.get("content") or "").strip()
        for m in messages
    ]
    contents = [c for c in contents if c]
    if not contents:
        return False, 0.0, ["no_narrative"]

    n = len(contents)
    full_text = "\n".join(contents)

    # Asterisk actions: *text* or standalone * on a line
    asterisk_msgs = sum(1 for c in contents if re.search(r'\*[^*]+\*', c))
    asterisk_ratio = asterisk_msgs / n

    # Average message length
    avg_len = sum(len(c) for c in contents) / n

    # Quoted dialogue
    has_quotes = bool(re.search(r'[«»""][^"«»""]{5,}[«»""]', full_text))

    # OOC density: ratio of messages with OOC patterns
    ooc_msgs = sum(1 for c in contents if _OOC_PATTERNS.search(c))
    ooc_ratio = ooc_msgs / n

    # Score assembly
    score = 0.0

    # Asterisks are the dominant signal
    if asterisk_ratio >= 0.4:
        score += 0.55
    elif asterisk_ratio >= 0.15:
        score += 0.35
    elif asterisk_ratio >= 0.05:
        score += 0.15

    # Narrative depth
    if avg_len >= 120:
        score += 0.25
    elif avg_len >= 60:
        score += 0.15
    elif avg_len >= 30:
        score += 0.05

    # Dialogue
    if has_quotes:
        score += 0.10

    # OOC penalty
    if ooc_ratio >= 0.5:
        score -= 0.40
    elif ooc_ratio >= 0.25:
        score -= 0.20
    elif ooc_ratio >= 0.10:
        score -= 0.05

    # Very small scene penalty
    if n < 3:
        score -= 0.20

    score = round(max(0.0, min(1.0, score)), 3)
    is_rp = score >= 0.5

    flags: list[str] = []
    if ooc_ratio >= 0.3:
        flags.append("ooc")
    if asterisk_ratio == 0 and avg_len < 80:
        flags.append("no_narrative")
    if avg_len < 40 and not flags:
        flags.append("too_casual")

    return is_rp, score, flags


def _scene_text(messages: list[dict], max_msgs: int = 30) -> str:
    lines = []
    for m in list(messages)[:max_msgs]:
        author = (m.get("author") or {}).get("name", "?") if isinstance(m.get("author"), dict) else str(m.get("author", "?"))
        content = m.get("content_en") or m.get("content", "")
        lines.append(f"[{author}]: {content}")
    return "\n".join(lines)


def _load_lore_how(lore_how_path: Path) -> dict:
    if lore_how_path.exists():
        return yaml.safe_load(lore_how_path.read_text(encoding="utf-8")) or {}
    return {}


def run_synthesis(scenes_dir: Path, lore_dir: Path, report_path: Path | None = None) -> Path:
    """
    Extract detailed narrative (LLM) + RP quality check (heuristic) per scene.
    Saves incrementally to lore_how.yaml.
    Writes rp_report.json listing all non-RP scenes for manual review.
    """
    lore_how_path = lore_dir / LORE_HOW_FILE
    lore_dir.mkdir(parents=True, exist_ok=True)

    lore_how = _load_lore_how(lore_how_path)
    scenes_data: dict = lore_how.get("scenes") or {}

    scene_files = sorted(scenes_dir.glob("**/*.json"))
    if not scene_files:
        print("  [lore_how] no scene files found")
        return lore_how_path

    new_count = n_rp = n_non_rp = 0
    for scene_file in scene_files:
        try:
            scene = json.loads(scene_file.read_text(encoding="utf-8"))
        except Exception:
            continue

        scene_id = scene.get("scene_id", scene_file.stem)
        if scene_id in scenes_data:
            print(f"    [skip] {scene_id}")
            existing = scenes_data[scene_id]
            if existing.get("is_rp", True):
                n_rp += 1
            else:
                n_non_rp += 1
            continue

        messages = scene.get("messages", [])
        text = _scene_text(messages)
        if not text.strip():
            continue

        # Heuristic RP check — fast, no LLM
        is_rp, rp_score, rp_flags = _heuristic_rp_check(messages)

        # LLM narrative synthesis (only for RP scenes — skip wasted calls on OOC)
        narrative = ""
        characters: list[dict] = []
        tensions: list[str] = []
        if is_rp:
            prior_context = synthesis_context_block(lore_dir, current_scene_id=scene_id, window=5)
            result = call_llm_json(
                _PROMPT.format(prior_context=prior_context, text=text),
                num_predict=1024,
                num_ctx=6144,
            )
            narrative   = str(result.get("narrative") or "")
            characters  = [c for c in (result.get("characters") or []) if isinstance(c, dict) and c.get("name")]
            tensions    = [str(t) for t in (result.get("tensions") or [])]

        entry = {
            "narrative":  narrative,
            "characters": characters,
            "tensions":   tensions,
            "is_rp":      is_rp,
            "rp_score":   rp_score,
            "rp_flags":   rp_flags,
        }

        scenes_data[scene_id] = entry
        lore_how["scenes"] = scenes_data
        lore_how_path.write_text(yaml.dump(lore_how, allow_unicode=True, sort_keys=False), encoding="utf-8")

        new_count += 1
        rp_marker = "✓" if is_rp else "✗"
        detail = f"{len(characters)} chars, {len(tensions)} tensions" if is_rp else ", ".join(rp_flags) or "filtered"
        print(f"    [lore_how] {scene_id}: rp {rp_score:.2f} {rp_marker}  {detail}")
        if is_rp:
            n_rp += 1
        else:
            n_non_rp += 1

    # Write rp_report.json — non-RP scenes for manual review
    if report_path is None:
        report_path = lore_dir.parent / "rp_report.json"
    non_rp = {
        sid: {"rp_score": d.get("rp_score"), "rp_flags": d.get("rp_flags"), "narrative": d.get("narrative")}
        for sid, d in scenes_data.items()
        if not d.get("is_rp", True)
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(non_rp, ensure_ascii=False, indent=2), encoding="utf-8")

    total = n_rp + n_non_rp
    print(f"  [lore_how] done — {new_count} new, {total} total: {n_rp} RP / {n_non_rp} non-RP")
    if n_non_rp:
        print(f"  [rp report] → {report_path}")
    return lore_how_path


def is_scene_rp(lore_dir: Path, scene_id: str) -> bool:
    """Return True if the scene passed the RP filter (or was never synthesized → assume RP)."""
    lore_how = load_lore_how(lore_dir)
    entry = (lore_how.get("scenes") or {}).get(scene_id)
    if entry is None:
        return True
    return bool(entry.get("is_rp", True))


def load_lore_how(lore_dir: Path) -> dict:
    path = lore_dir / LORE_HOW_FILE
    if not path.exists():
        return {}
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def synthesis_context_block(lore_dir: Path, current_scene_id: str | None = None, window: int = 12) -> str:
    """
    Format lore_how.yaml as a prompt injection block for step-5 analyze functions.
    Injects narrative summaries of the N scenes preceding current_scene_id.
    Only includes scenes that passed the RP filter.
    """
    lore_how = load_lore_how(lore_dir)
    scenes: dict = lore_how.get("scenes") or {}
    if not scenes:
        return "none"

    sorted_ids = sorted(scenes.keys())

    if current_scene_id is not None:
        try:
            pos = sorted_ids.index(current_scene_id)
        except ValueError:
            pos = len(sorted_ids)
        ids_to_show = sorted_ids[max(0, pos - window): pos]
    else:
        ids_to_show = sorted_ids[-window:]

    # Filter to RP-only scenes for cleaner context
    ids_to_show = [sid for sid in ids_to_show if scenes[sid].get("is_rp", True)]

    if not ids_to_show:
        return "none"

    lines = ["Prior scene context (use to anchor characters and events — do not merge distinct entities):"]
    for sid in ids_to_show:
        entry = scenes[sid]
        narrative = entry.get("narrative") or ""
        char_parts = [
            f"{c.get('name', '')} ({c.get('action', '')})"
            for c in (entry.get("characters") or [])[:4]
            if c.get("name")
        ]
        tensions = [str(t) for t in (entry.get("tensions") or [])[:2]]
        line = f"- {sid}: {narrative}"
        if char_parts:
            line += f"  [chars: {', '.join(char_parts)}]"
        if tensions:
            line += f"  [tensions: {'; '.join(tensions)}]"
        lines.append(line)

    return "\n".join(lines)
