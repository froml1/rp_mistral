"""
Step 4 — Synthesis: detailed narrative extraction + RP quality check, one LLM call per scene.
Output: data/lore/lore_how.yaml

For each scene, extracts:
  - narrative: 3-4 sentence summary of what happens
  - characters: who is active and what they do
  - tensions: conflicts, revelations, emotional stakes
  - is_rp / rp_score / rp_flags: RP quality gate (replaces step 5 rp_filter)

Incremental: already-processed scenes are skipped on re-run.
synthesis_context_block() formats the collection for prompt injection in step 5 (analyze).
is_scene_rp() is the single source of truth for RP filtering.
"""

import json
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from llm import call_llm_json

LORE_HOW_FILE = "lore_how.yaml"

_PROMPT = """\
Read this scene from a text-based roleplay and produce a detailed narrative summary.

In genuine RP, characters speak in dialogue and perform *actions written between asterisks*.
Be specific — name characters, places, objects. Capture the emotional atmosphere and stakes.

PRIOR SCENES (use to recognize recurring characters and ongoing tensions — do not invent):
{prior_context}

Extract:
- narrative: 3-4 sentences covering what happens (dialogue topics, *physical actions*, key decisions, emotional tone)
- characters: for each active character, what they do, say, or feel in this scene (one sentence each)
- tensions: unresolved conflicts, revelations, emotional stakes, questions left open at scene's end

Then evaluate whether this scene is genuine narrative roleplay:
- is_rp: true if the scene is genuine RP, false if it is mainly out-of-character
- rp_score: float 0.0 (not RP) to 1.0 (definitely RP). Threshold for is_rp=true is 0.5
- rp_flags: list only flags that apply — "ooc" | "no_narrative" | "too_casual" | "meta" | "technical"

RP signals (raise score):
  • *actions in asterisks* present anywhere in the scene (strong signal — nearly all genuine RP has them)
  • character voices in narrative dialogue (not player chat)
  • story framing: descriptions of place, atmosphere, time
  • in-universe references to characters, places, or events from the story

Non-RP signals (lower score):
  • zero *asterisk actions* throughout the entire scene
  • players talking as themselves (scheduling, jokes, reactions to the game)
  • pure casual/familiar conversation with no story framing and no narrative structure
  • rule discussion, technical/meta talk, session setup

JSON:
{{
  "narrative": "",
  "characters": [{{"name": "", "action": ""}}],
  "tensions": [],
  "is_rp": true,
  "rp_score": 0.9,
  "rp_flags": []
}}

Scene:
---
{text}
---"""


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
    Extract detailed narrative + RP quality check per scene.
    Saves incrementally to lore_how.yaml.
    Writes a rp_report.json of all non-RP scenes for manual review.
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

        text = _scene_text(scene.get("messages", []))
        if not text.strip():
            continue

        prior_context = synthesis_context_block(lore_dir, current_scene_id=scene_id, window=5)
        result = call_llm_json(
            _PROMPT.format(prior_context=prior_context, text=text),
            num_predict=640,
            num_ctx=6144,
        )

        is_rp   = bool(result.get("is_rp", True))
        rp_score = float(result.get("rp_score", 1.0))
        rp_flags = [str(f) for f in (result.get("rp_flags") or [])]

        entry = {
            "narrative":  str(result.get("narrative") or ""),
            "characters": [c for c in (result.get("characters") or []) if isinstance(c, dict) and c.get("name")],
            "tensions":   [str(t) for t in (result.get("tensions") or [])],
            "is_rp":      is_rp,
            "rp_score":   round(rp_score, 3),
            "rp_flags":   rp_flags,
        }

        scenes_data[scene_id] = entry
        lore_how["scenes"] = scenes_data
        lore_how_path.write_text(yaml.dump(lore_how, allow_unicode=True, sort_keys=False), encoding="utf-8")

        new_count += 1
        rp_marker = "✓" if is_rp else "✗"
        print(f"    [lore_how] {scene_id}: {len(entry['characters'])} chars, {len(entry['tensions'])} tensions — rp {rp_score:.2f} {rp_marker}")
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
    Format lore_how.yaml as a prompt injection block for step-6 analyze functions.
    Injects narrative summaries of the N scenes preceding current_scene_id.
    If current_scene_id is None, injects the last N scenes overall.
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
