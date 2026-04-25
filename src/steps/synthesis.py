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
from steps.scene_patch import chunk_messages

LORE_HOW_FILE = "lore_how.yaml"

_PROMPT = """\
Read this scene from a text-based roleplay and extract the three fields below IN ORDER.

Characters speak in dialogue and perform *actions written between asterisks*.

PRIOR SCENES (use to recognize recurring characters — do not invent):
{prior_context}

1. characters — every character ACTIVE in this scene (speaking or performing *actions*).
   For each: their name as it appears in the text, and what they specifically do/say/feel here.
   MUST NOT be empty. MUST use names from the scene text only.

2. tensions — unresolved conflicts, revelations, emotional stakes, open questions at scene end.
   MUST NOT be empty if anything significant is left unresolved.

3. narrative — a detailed, precise summary: dialogue topics, *physical actions*, decisions made,
   emotional atmosphere, locations mentioned. Be specific — name characters, places, objects.
   Do NOT summarise what you already listed; instead write a full scene account.

JSON:
{{
  "characters": [
    {{"name": "name as in scene", "action": "what this character does/says/feels in this scene"}},
    {{"name": "other name", "action": "their role"}}
  ],
  "tensions": [
    "first unresolved element or revelation",
    "second stake or open question"
  ],
  "narrative": "Full detailed account of the scene."
}}

Scene:
---
{text}
---"""


def _scene_text(messages: list[dict]) -> str:
    lines = []
    for m in messages:
        author = (m.get("author") or {}).get("name", "?") if isinstance(m.get("author"), dict) else str(m.get("author", "?"))
        content = m.get("content_en") or m.get("content", "")
        lines.append(f"[{author}]: {content}")
    return "\n".join(lines)


def _merge_synthesis(results: list[dict]) -> dict:
    if len(results) == 1:
        return results[0]

    # Characters: merge by name, keep longest action description
    char_map: dict[str, dict] = {}
    for r in results:
        for c in (r.get("characters") or []):
            if not isinstance(c, dict) or not c.get("name"):
                continue
            name = c["name"].lower()
            if name not in char_map or len(c.get("action", "")) > len(char_map[name].get("action", "")):
                char_map[name] = c

    # Tensions: union, deduplicate by lowercased text
    seen: set[str] = set()
    tensions: list[str] = []
    for r in results:
        for t in (r.get("tensions") or []):
            t = str(t)
            if t.lower() not in seen:
                tensions.append(t); seen.add(t.lower())

    # Narrative: join all chunk narratives in order
    narrative = " ".join(str(r.get("narrative") or "") for r in results if r.get("narrative"))

    return {"characters": list(char_map.values()), "tensions": tensions, "narrative": narrative}


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
        if not messages:
            continue

        prior_context = synthesis_context_block(lore_dir, current_scene_id=scene_id, window=5)
        chunks = chunk_messages(messages)
        if len(chunks) > 1:
            print(f"    [lore_how] {scene_id}: {len(chunks)} chunks")
        raw_results = [
            call_llm_json(
                _PROMPT.format(prior_context=prior_context, text=_scene_text(chunk)),
                num_predict=2048,
                num_ctx=8192,
            )
            for chunk in chunks
        ]
        result = _merge_synthesis(raw_results)

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


def update_scene_entry(lore_dir: Path, scene_id: str, who: dict, what: dict, how: dict) -> None:
    """
    Overwrite the lore_how.yaml entry for scene_id with richer analysis results.
    Called after step-6 analysis completes so subsequent scenes get accurate context.
    No new LLM call — uses already-computed what/how results directly.
    """
    lore_how_path = lore_dir / LORE_HOW_FILE
    lore_how   = _load_lore_how(lore_how_path)
    scenes_data = lore_how.setdefault("scenes", {})

    # Characters: names present in this scene + first relevant action from what.events
    char_names = [c for c in (who.get("characters") or []) if c]
    char_action_map: dict[str, str] = {}
    for event in (what.get("events") or []):
        desc = (event.get("description") or "")[:80]
        for char in (event.get("characters") or []):
            if char and char not in char_action_map and desc:
                char_action_map[char] = desc
    char_actions = [
        {"name": name, "action": char_action_map.get(name, "present")}
        for name in char_names
    ]

    # Tensions: only revelations and decisions from THIS scene — no cross-scene synthesis
    tensions = [
        (event.get("description") or "")[:120]
        for event in (what.get("events") or [])
        if event.get("type") in ("revelation", "decision") and event.get("description")
    ]

    narrative = str(what.get("summary") or scenes_data.get(scene_id, {}).get("narrative") or "")

    entry = {
        "narrative":  narrative,
        "characters": char_actions,
        "tensions":   tensions[:4],
        "_corrected": True,
    }

    scenes_data[scene_id] = entry
    lore_how["scenes"] = scenes_data
    lore_how_path.write_text(yaml.dump(lore_how, allow_unicode=True, sort_keys=False), encoding="utf-8")
    print(f"    [lore_how] corrected entry for {scene_id}")


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
