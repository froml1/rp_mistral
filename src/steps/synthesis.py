"""
Step 4 — Synthesis: detailed narrative extraction, one LLM call per scene.
Output: data/lore/lore_how.yaml

RP filtering is handled upstream (subdivide step) — all scenes here are assumed RP.

For each scene:
  - narrative: 3-4 sentence summary
  - characters: name + action per active character
  - tensions: unresolved conflicts, emotional stakes

Incremental: already-processed scenes are skipped on re-run.
synthesis_context_block() injects a sliding window of prior scene context into analyze prompts.
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


def run_synthesis(scenes_dir: Path, lore_dir: Path) -> Path:
    """Extract detailed narrative per scene. Saves incrementally to lore_how.yaml."""
    lore_how_path = lore_dir / LORE_HOW_FILE
    lore_dir.mkdir(parents=True, exist_ok=True)

    lore_how    = _load_lore_how(lore_how_path)
    scenes_data = lore_how.get("scenes") or {}

    scene_files = sorted(p for p in scenes_dir.glob("**/*.json") if not p.name.startswith("_"))
    if not scene_files:
        print("  [lore_how] no scene files found")
        return lore_how_path

    new_count = 0
    for scene_file in scene_files:
        try:
            scene = json.loads(scene_file.read_text(encoding="utf-8"))
        except Exception:
            continue

        scene_id = scene.get("scene_id", scene_file.stem)
        if scene_id in scenes_data:
            print(f"    [skip] {scene_id}")
            continue

        messages = scene.get("messages", [])
        text = _scene_text(messages)
        if not text.strip():
            continue

        prior_context = synthesis_context_block(lore_dir, current_scene_id=scene_id, window=5)
        result = call_llm_json(
            _PROMPT.format(prior_context=prior_context, text=text),
            num_predict=1024,
            num_ctx=6144,
        )

        entry = {
            "narrative":  str(result.get("narrative") or ""),
            "characters": [c for c in (result.get("characters") or []) if isinstance(c, dict) and c.get("name")],
            "tensions":   [str(t) for t in (result.get("tensions") or [])],
        }

        scenes_data[scene_id] = entry
        lore_how["scenes"] = scenes_data
        lore_how_path.write_text(yaml.dump(lore_how, allow_unicode=True, sort_keys=False), encoding="utf-8")

        new_count += 1
        print(f"    [lore_how] {scene_id}: {len(entry['characters'])} chars, {len(entry['tensions'])} tensions")

    print(f"  [lore_how] done — {new_count} new, {len(scenes_data)} total")
    return lore_how_path


def load_lore_how(lore_dir: Path) -> dict:
    path = lore_dir / LORE_HOW_FILE
    if not path.exists():
        return {}
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def synthesis_context_block(lore_dir: Path, current_scene_id: str | None = None, window: int = 12) -> str:
    """
    Format lore_how.yaml as a prompt injection block for analyze functions.
    Injects narrative summaries of the N scenes preceding current_scene_id.
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
        narrative  = entry.get("narrative") or ""
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
