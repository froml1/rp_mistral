"""Step 4b — Where: location context analysis, updates place YAMLs."""

import json
import re
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from llm import call_llm_json
from steps.manual_lore import load_manual_place, load_all_manual_places, merge_manual_into_place


def _is_valid_json(path: Path) -> bool:
    try:
        json.loads(path.read_text(encoding="utf-8"))
        return True
    except Exception:
        return False

_PROMPT = """\
Analyze the LOCATIONS in this RP scene.

IMPORTANT: if context seams too informal ignore analyse, return struct with empty fields (maybe a casual discussion)
Known information about locations (may be incomplete or need correction):
{known_yaml}

Extract all locations present in this scene. For each location:
- canonical_name: main name in lowercase (e.g. "the silver tavern")
- appellations: all names/references used for this place in the scene, excluding pronouns (e.g. ["the tavern", "the silver", "old silver"])
- description: physical description from the scene
- attributes: list of descriptive qualities (e.g. ["dimly lit", "crowded", "underground"])
- is_primary: true if main location of the scene

Also indicate if the location changes during the scene (location_changes: true/false).

Correct any inconsistency with known information. Add new details only.

JSON: {{"locations": [{{"canonical_name": "", "appellations": [], "description": "", "attributes": [], "is_primary": true}}], "location_changes": false}}

Scene:
---
{text}
---"""


def _load_place_yaml(places_dir: Path, canonical_name: str) -> dict:
    slug = re.sub(r'\s+', '_', canonical_name.lower().strip())
    path = places_dir / f"{slug}.yaml"
    if path.exists():
        with open(path, encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return {}


def _save_place_yaml(places_dir: Path, canonical_name: str, data: dict):
    slug = re.sub(r'\s+', '_', canonical_name.lower().strip())
    places_dir.mkdir(parents=True, exist_ok=True)
    path = places_dir / f"{slug}.yaml"
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(data, f, allow_unicode=True, sort_keys=False)


def _merge_place(existing: dict, extracted: dict, scene_id: str) -> dict:
    merged = dict(existing)
    merged.setdefault("name", extracted.get("canonical_name", ""))
    merged.setdefault("appellations", [])
    merged.setdefault("description", "")
    merged.setdefault("attributes", [])
    merged.setdefault("appearances", [])

    for app in (extracted.get("appellations") or []):
        if app.lower() not in [a.lower() for a in merged["appellations"]]:
            merged["appellations"].append(app.lower())

    new_desc = extracted.get("description") or ""
    if new_desc and len(new_desc) > len(merged["description"]):
        merged["description"] = new_desc.lower()

    for attr in (extracted.get("attributes") or []):
        if attr.lower() not in [a.lower() for a in merged["attributes"]]:
            merged["attributes"].append(attr.lower())

    if scene_id not in merged["appearances"]:
        merged["appearances"].append(scene_id)

    merged.setdefault("first_appearance", scene_id)
    return merged


def _scene_text(messages: list[dict]) -> str:
    return "\n".join(
        f"{m.get('content_en') or m.get('content', '')}"
        for m in messages
    )


def run_where(scene_file: Path, analysis_dir: Path, places_dir: Path) -> dict:
    out_path = analysis_dir / "where.json"
    if out_path.exists() and _is_valid_json(out_path):
        print(f"    [skip] where already done")
        return json.loads(out_path.read_text(encoding="utf-8"))
    if out_path.exists():
        print(f"    [corrupt] where.json malformed, re-analyzing...")

    if not _is_valid_json(scene_file):
        print(f"    [error] scene file {scene_file.name} is malformed, delete it and re-run step 3")
        return {}
    with open(scene_file, encoding="utf-8") as f:
        scene = json.load(f)

    scene_id = scene["scene_id"]
    messages = scene["messages"]
    text = _scene_text(messages)

    # Load known places (all of them as context, manual overrides injected)
    known = {}
    if places_dir.exists():
        for yf in places_dir.glob("*.yaml"):
            with open(yf, encoding="utf-8") as f:
                d = yaml.safe_load(f) or {}
                if d.get("name"):
                    known[d["name"]] = d
    manual_places = load_all_manual_places()
    for name, mp in manual_places.items():
        if name in known:
            known[name] = merge_manual_into_place(known[name], mp)
        else:
            known[name] = mp

    known_yaml = yaml.dump(known, allow_unicode=True) if known else "none"
    result = call_llm_json(_PROMPT.format(known_yaml=known_yaml, text=text), num_predict=2048)

    locations = [l for l in (result.get("locations") or []) if isinstance(l, dict) and l.get("canonical_name")]

    # Update place YAMLs
    for loc in locations:
        existing = _load_place_yaml(places_dir, loc["canonical_name"])
        merged = _merge_place(existing, loc, scene_id)
        merged = merge_manual_into_place(merged, load_manual_place(loc["canonical_name"]))
        _save_place_yaml(places_dir, loc["canonical_name"], merged)
        print(f"    place updated: {loc['canonical_name']}")

    output = {
        "locations":        [l.get("canonical_name") for l in locations],
        "location_changes": bool(result.get("location_changes", False)),
        "details":          locations,
    }

    analysis_dir.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    return output
