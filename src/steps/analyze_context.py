"""Step 4a — Context: when + where in a single LLM call (replaces analyze_when + analyze_where)."""

import json
import re
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from llm import call_llm_json
from steps.manual_lore import load_manual_place, load_all_manual_places, merge_manual_into_place
from steps.synthesis import synthesis_context_block
from lore_summary import update_summary
try:
    from store import upsert as _store_upsert
except Exception:
    def _store_upsert(*a, **kw): pass


def _is_valid_json(path: Path) -> bool:
    try:
        json.loads(path.read_text(encoding="utf-8"))
        return True
    except Exception:
        return False


_PROMPT = """\
Analyze the TEMPORAL CONTEXT and LOCATIONS of this RP scene in one pass.
IMPORTANT: if context seams too informal ignore analyse, return struct with empty fields (maybe a casual discussion)

STORY SYNTHESIS (use to anchor location identity — do not confuse distinct places):
{synthesis}

Known locations (may be incomplete):
{known_yaml}

TEMPORAL — base analysis on narrative content only, NOT on timestamps:
- summary: one sentence describing the temporal context
- duration: estimated duration (e.g. "a few minutes", "an evening", "unknown")
- time_of_day: morning | afternoon | evening | night | unknown
- time_scales: list of time references in the narrative (years, seasons, past events…)
- time_gaps: temporal gaps mentioned between fragments (e.g. "three days later")

LOCATIONS — for each location present:
- canonical_name: main name in lowercase
- appellations: all names/references used for this place
- description: physical description from the scene
- attributes: list of descriptive qualities (e.g. ["dimly lit", "crowded"])
- is_primary: true if main location of the scene

Also: location_changes: true if the location changes during the scene.

JSON:
{{
  "when": {{
    "summary": "", "duration": "", "time_of_day": "",
    "time_scales": [], "time_gaps": []
  }},
  "where": {{
    "locations": [{{"canonical_name": "", "appellations": [], "description": "", "attributes": [], "is_primary": true}}],
    "location_changes": false
  }}
}}

Scene:
---
{text}
---"""


def _scene_text(messages: list[dict]) -> str:
    return "\n".join(
        f"{m.get('content_en') or m.get('content', '')}"
        for m in messages
    )


def _slug(name: str) -> str:
    return re.sub(r'\s+', '_', name.lower().strip())


def _load_place_yaml(places_dir: Path, canonical_name: str) -> dict:
    path = places_dir / f"{_slug(canonical_name)}.yaml"
    if path.exists():
        with open(path, encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return {}


def _save_place_yaml(places_dir: Path, canonical_name: str, data: dict):
    places_dir.mkdir(parents=True, exist_ok=True)
    path = places_dir / f"{_slug(canonical_name)}.yaml"
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


def run_context(scene_file: Path, analysis_dir: Path, places_dir: Path, lore_dir: Path | None = None) -> tuple[dict, dict]:
    """Returns (when_dict, where_dict). Writes when.json and where.json."""
    when_path  = analysis_dir / "when.json"
    where_path = analysis_dir / "where.json"

    when_done  = when_path.exists()  and _is_valid_json(when_path)
    where_done = where_path.exists() and _is_valid_json(where_path)

    if when_done and where_done:
        print("    [skip] context already done")
        return (
            json.loads(when_path.read_text(encoding="utf-8")),
            json.loads(where_path.read_text(encoding="utf-8")),
        )

    if not _is_valid_json(scene_file):
        print(f"    [error] scene file {scene_file.name} is malformed")
        return {}, {}

    with open(scene_file, encoding="utf-8") as f:
        scene = json.load(f)

    scene_id = scene["scene_id"]
    text = _scene_text(scene["messages"])

    # Known places with manual overrides
    known = {}
    if places_dir.exists():
        for yf in places_dir.glob("*.yaml"):
            with open(yf, encoding="utf-8") as f:
                d = yaml.safe_load(f) or {}
                if d.get("name"):
                    known[d["name"]] = d
    for name, mp in load_all_manual_places().items():
        known[name] = merge_manual_into_place(known.get(name, {}), mp) if name in known else mp

    known_yaml = "\n".join(f"- {n}: {d.get('_summary', '')}" for n, d in known.items()) or "none"
    result = call_llm_json(
        _PROMPT.format(
            synthesis=synthesis_context_block(lore_dir, current_scene_id=scene_id) if lore_dir else "none",
            known_yaml=known_yaml,
            text=text,
        ),
        num_predict=1024,
    )

    # — When —
    raw_when = result.get("when") or {}
    when_out = {
        "summary":     str(raw_when.get("summary") or ""),
        "duration":    str(raw_when.get("duration") or "unknown"),
        "time_of_day": str(raw_when.get("time_of_day") or "unknown"),
        "time_scales": [str(t) for t in (raw_when.get("time_scales") or [])],
        "time_gaps":   [str(t) for t in (raw_when.get("time_gaps") or [])],
    }

    # — Where —
    raw_where = result.get("where") or {}
    locations = [l for l in (raw_where.get("locations") or []) if isinstance(l, dict) and l.get("canonical_name")]

    for loc in locations:
        existing = _load_place_yaml(places_dir, loc["canonical_name"])
        merged   = _merge_place(existing, loc, scene_id)
        merged   = merge_manual_into_place(merged, load_manual_place(loc["canonical_name"]))
        _save_place_yaml(places_dir, loc["canonical_name"], merged)
        summary  = update_summary(places_dir / f"{_slug(loc['canonical_name'])}.yaml", "places")
        _store_upsert("places", merged["name"], summary)
        print(f"    place updated: {loc['canonical_name']}")

    where_out = {
        "locations":        [l.get("canonical_name") for l in locations],
        "location_changes": bool(raw_where.get("location_changes", False)),
        "details":          locations,
    }

    analysis_dir.mkdir(parents=True, exist_ok=True)
    when_path.write_text(json.dumps(when_out, ensure_ascii=False, indent=2), encoding="utf-8")
    where_path.write_text(json.dumps(where_out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"    context: {when_out['time_of_day']} / {when_out['duration']} | {len(locations)} location(s)")
    return when_out, where_out
