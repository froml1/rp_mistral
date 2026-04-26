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
Read this fragment of a roleplay scene and extract the three fields below.
Extract ONLY what is explicitly present in this fragment. Do NOT import elements from other scenes.

Characters speak in dialogue and perform *actions written between asterisks*.

CHARACTERS ALREADY IDENTIFIED IN EARLIER PARTS OF THIS SAME SCENE (use these names if the same characters continue — do not duplicate):
{prior_chunk_context}

1. characters — every character ACTIVE in this fragment (speaking or performing *actions*).
   Use PROPER NAMES only — never pronouns (he/she/il/elle/they) or vague references.
   If a character was already listed above, reuse their exact name.
   For each: their name and what they specifically do/say/feel here.

2. tensions — unresolved conflicts, revelations, emotional stakes visible at the end of this fragment.

3. narrative — detailed summary of this fragment: dialogue topics, *physical actions*, decisions made,
   emotional atmosphere, locations. Name characters, places, objects. This fragment only.

JSON:
{{
  "characters": [
    {{"name": "proper name only", "action": "what this character does/says/feels"}},
    {{"name": "other name", "action": "their role"}}
  ],
  "tensions": ["first unresolved element", "second stake"],
  "narrative": "Detailed account of this fragment."
}}

Fragment:
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


_PRONOUNS = {
    "he", "she", "it", "they", "him", "her", "his", "hers", "their", "theirs",
    "i", "me", "my", "you", "we", "us", "one", "someone", "anyone", "nobody",
    "il", "elle", "ils", "elles", "lui", "leur", "je", "tu", "nous", "vous", "on",
    "ce", "ceci", "cela", "ça", "celui", "celle", "ceux",
}


def _merge_synthesis(results: list[dict]) -> dict:
    char_map: dict[str, str] = {}
    for r in results:
        for c in (r.get("characters") or []):
            if not isinstance(c, dict) or not c.get("name"):
                continue
            name = c["name"].strip()
            name_low = name.lower()
            # skip pronouns, very short tokens, or multi-word fragments that look like sentences
            if name_low in _PRONOUNS or len(name) < 2 or " " in name and len(name) > 30:
                continue
            if name_low not in char_map or len(c.get("action") or "") > len(char_map[name_low]):
                char_map[name_low] = c.get("action") or ""
    characters = [{"name": name, "action": action} for name, action in char_map.items()]
    tensions = list(dict.fromkeys(
        str(t) for r in results for t in (r.get("tensions") or []) if t
    ))
    narrative = " ".join(str(r.get("narrative") or "") for r in results if r.get("narrative"))
    return {"characters": characters, "tensions": tensions, "narrative": narrative}


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

        chunks = chunk_messages(messages)
        if len(chunks) > 1:
            print(f"    [lore_how] {scene_id}: {len(chunks)} chunks")
        raw_results = []
        prior_chunk_context = "none"
        for chunk in chunks:
            r = call_llm_json(
                _PROMPT.format(prior_chunk_context=prior_chunk_context, text=_scene_text(chunk)),
                num_predict=2048,
                num_ctx=8192,
            )
            raw_results.append(r)
            names = [c["name"] for c in (r.get("characters") or []) if isinstance(c, dict) and c.get("name")]
            if names:
                prior_chunk_context = ", ".join(names)
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


def update_scene_entry(lore_dir: Path, scene_id: str, who: dict, what: dict, how: dict = {}) -> None:  # noqa: ARG001
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


def current_scene_synthesis(lore_dir: Path, scene_id: str) -> str:
    """Return the step-5 synthesis of the current scene as a brief context string.
    Intentionally omits the character list — it is unreliable and causes false attribution."""
    lore_how = load_lore_how(lore_dir)
    entry = (lore_how.get("scenes") or {}).get(scene_id)
    if not entry:
        return "none"
    narrative = entry.get("narrative") or ""
    tensions = [str(t) for t in (entry.get("tensions") or [])[:2]]
    out = narrative
    if tensions:
        out += f"  [tensions: {'; '.join(tensions)}]"
    return out or "none"


def synthesis_context_block(lore_dir: Path, current_scene_id: str | None = None, window: int = 12, characters: list[str] | None = None) -> str:
    """
    Format lore_how.yaml as a prompt injection block for analyze functions.
    If characters is provided, injects the last `window` preceding scenes where
    any of those characters appear. Otherwise falls back to the last `window` scenes.
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
        preceding = sorted_ids[:pos]
    else:
        preceding = sorted_ids

    if characters:
        chars_lower = {c.lower() for c in characters if c}
        preceding = [
            sid for sid in preceding
            if any(
                c.get("name", "").lower() in chars_lower
                for c in (scenes[sid].get("characters") or [])
            )
        ]

    ids_to_show = preceding[-window:]

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
