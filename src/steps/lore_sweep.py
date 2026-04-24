"""
Step 4 — Lore Sweep: light entity extraction pass over the entire scene corpus.

One LLM call per scene, extracts characters/places/concepts with 1-2 sentence
descriptions. Merges into data/lore/sweep.yaml — a stable, corpus-wide reference
used as anchor context by step 6 (analyze) and step 5 (rp_filter).
Incremental: already-swept scenes are skipped.
"""

import json
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))
from llm import call_llm_json

_PROMPT = """\
Quick entity and narrative scan of this RP scene. Only what is explicitly visible — do not invent.

For each CHARACTER present: canonical_name (lowercase full name), description (1-2 sentences on role/appearance/personality), appellations (every name form used in the text).
For each LOCATION: canonical_name (lowercase), description (1-2 sentences), appellations.
For each named CONCEPT (faction, artifact, law, system, ideology, ritual): canonical_name, type, description (1 sentence).
narrative: one sentence summarising what happens in this scene and who is involved (key events + character dynamics).

JSON:
{{
  "characters": [{{"name": "", "description": "", "appellations": []}}],
  "places":     [{{"name": "", "description": "", "appellations": []}}],
  "concepts":   [{{"name": "", "type": "", "description": ""}}],
  "narrative":  ""
}}

Scene:
---
{text}
---"""


def _scene_text(messages: list[dict]) -> str:
    return "\n".join(
        f"{(m.get('author') or {}).get('name', '?') if isinstance(m.get('author'), dict) else m.get('author', '?')}: "
        f"{m.get('content_en') or m.get('content', '')}"
        for m in messages
    )


def _merge_bucket(bucket: dict, items: list, scene_id: str, *, with_type: bool = False):
    for item in (items or []):
        # LLM sometimes returns strings instead of dicts
        if isinstance(item, str):
            item = {"name": item}
        if not isinstance(item, dict):
            continue
        name = (item.get("name") or "").strip().lower()
        if not name:
            continue
        entry = bucket.setdefault(name, {"name": name, "description": "", "appellations": [], "scenes": []})

        new_desc = (item.get("description") or "").strip()
        if new_desc and len(new_desc) > len(entry["description"]):
            entry["description"] = new_desc

        existing_apps = {a.lower() for a in entry["appellations"]}
        for app in (item.get("appellations") or []):
            if isinstance(app, str) and app.lower() not in existing_apps:
                entry["appellations"].append(app.lower())
                existing_apps.add(app.lower())

        if with_type and item.get("type"):
            entry.setdefault("type", item["type"])

        if scene_id not in entry["scenes"]:
            entry["scenes"].append(scene_id)


def _already_swept_scenes(registry: dict) -> set[str]:
    seen: set[str] = set()
    for bucket in registry.values():
        if isinstance(bucket, dict):
            for entry in bucket.values():
                if isinstance(entry, dict):
                    seen.update(entry.get("scenes") or [])
    return seen


def run_lore_sweep(scenes_dir: Path, lore_dir: Path) -> Path:
    """Light entity sweep over all scenes. Returns path to sweep.yaml."""
    sweep_path = lore_dir / "sweep.yaml"

    registry: dict = {}
    if sweep_path.exists():
        registry = yaml.safe_load(sweep_path.read_text(encoding="utf-8")) or {}

    done_scenes = _already_swept_scenes(registry)

    scene_files = sorted(scenes_dir.glob("**/*.json"))
    if not scene_files:
        print("  [sweep] no scenes found")
        return sweep_path

    new_count = 0
    for scene_file in scene_files:
        try:
            scene = json.loads(scene_file.read_text(encoding="utf-8"))
        except Exception:
            continue

        scene_id = scene.get("scene_id", scene_file.stem)
        if scene_id in done_scenes:
            print(f"    [skip] sweep {scene_id}")
            continue

        text = _scene_text(scene.get("messages", []))
        if not text.strip():
            continue

        result = call_llm_json(_PROMPT.format(text=text), num_predict=768, num_ctx=6144)

        chars_bucket    = registry.setdefault("characters", {})
        places_bucket   = registry.setdefault("places", {})
        concepts_bucket = registry.setdefault("concepts", {})

        _merge_bucket(chars_bucket,    result.get("characters") or [], scene_id)
        _merge_bucket(places_bucket,   result.get("places")     or [], scene_id)
        _merge_bucket(concepts_bucket, result.get("concepts")   or [], scene_id, with_type=True)

        narrative = (result.get("narrative") or "").strip()
        if narrative:
            registry.setdefault("narratives", {})[scene_id] = narrative

        new_count += 1
        print(
            f"    [sweep] {scene_id}: "
            f"{len(result.get('characters') or [])} chars, "
            f"{len(result.get('places') or [])} places, "
            f"{len(result.get('concepts') or [])} concepts"
        )

        # Save after each scene so interruption doesn't lose progress
        sweep_path.write_text(yaml.dump(registry, allow_unicode=True, sort_keys=False), encoding="utf-8")

    lore_dir.mkdir(parents=True, exist_ok=True)
    sweep_path.write_text(yaml.dump(registry, allow_unicode=True, sort_keys=False), encoding="utf-8")

    totals = {k: len(v) for k, v in registry.items()}
    print(f"  [sweep] done — {new_count} new scenes | totals: {totals}")
    return sweep_path


def load_sweep(lore_dir: Path) -> dict:
    path = lore_dir / "sweep.yaml"
    if not path.exists():
        return {}
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def sweep_narrative_context(lore_dir: Path, current_scene_id: str, window: int = 10) -> str:
    """
    Return narrative notes from scenes before current_scene_id (sorted).
    Used as stable pre-computed context in analyze_how and analyze_what.
    """
    sweep = load_sweep(lore_dir)
    narratives: dict[str, str] = sweep.get("narratives") or {}
    if not narratives:
        return "none yet"

    sorted_ids = sorted(narratives.keys())
    try:
        pos = sorted_ids.index(current_scene_id)
    except ValueError:
        pos = len(sorted_ids)

    nearby = sorted_ids[max(0, pos - window): pos]
    if not nearby:
        return "none yet"

    return "\n".join(f"- {sid}: {narratives[sid]}" for sid in nearby)


def sweep_context_lines(lore_dir: Path, entity_type: str, limit: int = 20) -> str:
    """Return a formatted string listing sweep entities of one type, for prompt injection."""
    sweep = load_sweep(lore_dir)
    bucket = sweep.get(entity_type) or {}
    if not bucket:
        return "none"
    lines = []
    for name, entry in list(bucket.items())[:limit]:
        desc = entry.get("description") or ""
        apps = ", ".join((entry.get("appellations") or [])[:4])
        n = len(entry.get("scenes") or [])
        line = f"- {name}: {desc}"
        if apps:
            line += f"  [aliases: {apps}]"
        if n > 1:
            line += f"  ({n} scenes)"
        lines.append(line)
    return "\n".join(lines)
