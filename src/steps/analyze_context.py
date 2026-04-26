"""Step 4a — Context: when + where in a single LLM call (replaces analyze_when + analyze_where)."""

import json
import re
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from llm import call_llm_json
from steps.manual_lore import load_manual_place, merge_manual_into_place
from steps.scene_patch import write_enrichment, chunk_messages
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
Extract ONLY what is explicitly present in the scene text below. Do NOT infer or invent locations.

LOCATIONS — for each location present:
- canonical_name: main name in lowercase
- appellations: all names/references used for this place
- type: building | wilderness | city | dungeon | dimension | vessel | other
- description: physical description from the scene
- attributes: list of descriptive qualities (e.g. ["dimly lit", "crowded"])
- atmosphere: emotional tone of the place (e.g. "oppressive", "welcoming", "eerie") — one word or short phrase
- control: who controls or owns this place — character name, faction name, or "unknown"
- known_inhabitants: characters who live or work here regularly (not just visitors in this scene)
- access: public | restricted | secret | dangerous | unknown
- is_primary: true if main location of the scene

Also: location_changes: true if the location changes during the scene.

TEMPORAL — base analysis on narrative content only, NOT on timestamps:
- summary: one sentence describing ONLY time of day, duration and atmosphere — no character names
- duration: estimated duration (e.g. "a few minutes", "an evening", "unknown")
- time_of_day: morning | afternoon | evening | night | unknown
- time_scales: list of time references in the narrative (years, seasons, past events…)
- time_gaps: temporal gaps mentioned between fragments (e.g. "three days later")

LOCATIONS AND TEMPORAL DATA ALREADY IDENTIFIED IN EARLIER PARTS OF THIS SAME SCENE (do not duplicate):
{prior_chunk_context}

INCONSISTENCIES — flag any clear problem: a character appearing in an impossible location,
a location that contradicts the known world, a temporal jump not explained by the narrative, etc.
inconsistencies: [{{"message_idx": int_or_null, "type": "impossible_location|temporal_gap|location_conflict|other", "description": "..."}}]

JSON:
{{
  "where": {{
    "locations": [{{"canonical_name": "", "appellations": [], "type": "other", "description": "", "attributes": [], "atmosphere": "", "control": "unknown", "known_inhabitants": [], "access": "unknown", "is_primary": true}}],
    "location_changes": false
  }},
  "when": {{
    "summary": "", "duration": "", "time_of_day": "",
    "time_scales": [], "time_gaps": []
  }},
  "inconsistencies": []
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
    merged.setdefault("known_inhabitants", [])
    merged.setdefault("appearances", [])
    for field in ("type", "atmosphere", "control", "access"):
        merged.setdefault(field, "")

    for app in (extracted.get("appellations") or []):
        if app.lower() not in [a.lower() for a in merged["appellations"]]:
            merged["appellations"].append(app.lower())

    new_desc = extracted.get("description") or ""
    if new_desc and len(new_desc) > len(merged["description"]):
        merged["description"] = new_desc.lower()

    for attr in (extracted.get("attributes") or []):
        if attr.lower() not in [a.lower() for a in merged["attributes"]]:
            merged["attributes"].append(attr.lower())

    for inh in (extracted.get("known_inhabitants") or []):
        if inh.lower() not in [i.lower() for i in merged["known_inhabitants"]]:
            merged["known_inhabitants"].append(inh.lower())

    for field in ("type", "atmosphere", "control", "access"):
        new_val = (extracted.get(field) or "").strip().lower()
        if new_val and new_val not in ("unknown", "other", ""):
            merged[field] = new_val

    if scene_id not in merged["appearances"]:
        merged["appearances"].append(scene_id)

    merged.setdefault("first_appearance", scene_id)
    return merged


def _enrich_place_from_lore(extracted: dict, existing: dict) -> dict:
    """Fill gaps in the freshly-extracted location with data from existing lore. Extracted data takes priority."""
    enriched = dict(extracted)
    for field in ("type", "atmosphere", "control", "access"):
        v = (enriched.get(field) or "").strip().lower()
        if not v or v in ("unknown", "other", ""):
            lore_val = (existing.get(field) or "").strip().lower()
            if lore_val and lore_val not in ("unknown", "other", ""):
                enriched[field] = lore_val
    if not enriched.get("description") and existing.get("description"):
        enriched["description"] = existing["description"]
    for lst_field in ("appellations", "attributes", "known_inhabitants"):
        lore_items = existing.get(lst_field) or []
        scene_items = enriched.get(lst_field) or []
        seen = {str(x).lower() for x in scene_items}
        enriched[lst_field] = list(scene_items)
        for item in lore_items:
            if str(item).lower() not in seen:
                enriched[lst_field].append(item)
                seen.add(str(item).lower())
    return enriched


def _merge_context(results: list[dict]) -> dict:
    if len(results) == 1:
        return results[0]
    whens  = [r.get("when")  or {} for r in results]
    wheres = [r.get("where") or {} for r in results]

    merged_when = {
        "summary":     " ".join(w.get("summary", "")    for w in whens if w.get("summary")),
        "duration":    next((w["duration"]    for w in whens if w.get("duration")    and w["duration"]    != "unknown"), "unknown"),
        "time_of_day": next((w["time_of_day"] for w in whens if w.get("time_of_day") and w["time_of_day"] != "unknown"), "unknown"),
        "time_scales": list(dict.fromkeys(t for w in whens for t in (w.get("time_scales") or []))),
        "time_gaps":   list(dict.fromkeys(t for w in whens for t in (w.get("time_gaps")   or []))),
    }

    loc_map: dict[str, dict] = {}
    for w in wheres:
        for loc in (w.get("locations") or []):
            name = (loc.get("canonical_name") or "").lower()
            if not name or name.startswith("unknown"):
                continue
            if name not in loc_map:
                loc_map[name] = dict(loc)
            else:
                ex = loc_map[name]
                for list_field, key in (("appellations", None), ("attributes", None), ("known_inhabitants", None)):
                    seen = {a.lower() for a in (ex.get(list_field) or [])}
                    for a in (loc.get(list_field) or []):
                        if a.lower() not in seen:
                            ex.setdefault(list_field, []).append(a.lower()); seen.add(a.lower())
                if len(loc.get("description", "")) > len(ex.get("description", "")):
                    ex["description"] = loc["description"]
                for field in ("type", "atmosphere", "control", "access"):
                    v = (loc.get(field) or "").strip().lower()
                    if v and v not in ("unknown", "other", "") and not ex.get(field):
                        ex[field] = v

    incs = [i for r in results for i in (r.get("inconsistencies") or []) if isinstance(i, dict)]
    return {
        "when":            merged_when,
        "where":           {"locations": list(loc_map.values()), "location_changes": any(w.get("location_changes") for w in wheres)},
        "inconsistencies": incs,
    }


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
    messages = scene["messages"]

    chunks = chunk_messages(messages)
    if len(chunks) > 1:
        print(f"    context: {len(chunks)} chunks")
    results = []
    prior_chunk_context = "none"
    for chunk in chunks:
        r = call_llm_json(
            _PROMPT.format(
                prior_chunk_context=prior_chunk_context,
                text=_scene_text(chunk),
            ),
            num_predict=2048,
        )
        results.append(r)
        locs = [l.get("canonical_name", "") for l in (r.get("where") or {}).get("locations", []) if l.get("canonical_name")]
        when = r.get("when") or {}
        prior_chunk_context = (
            f"locations: {', '.join(locs) or 'none'} | "
            f"time_of_day: {when.get('time_of_day', 'unknown')} | "
            f"duration: {when.get('duration', 'unknown')}"
        )
    result = _merge_context(results)

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
    locations = [
        l for l in (raw_where.get("locations") or [])
        if isinstance(l, dict) and l.get("canonical_name")
        and not (l["canonical_name"] or "").lower().strip().startswith("unknown")
    ]

    for i, loc in enumerate(locations):
        existing = _load_place_yaml(places_dir, loc["canonical_name"])
        loc = _enrich_place_from_lore(loc, existing)
        locations[i] = loc
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

    incs = [i for i in (result.get("inconsistencies") or []) if isinstance(i, dict) and i.get("description")]
    if incs:
        write_enrichment(scene_file, "context", {"inconsistencies": incs})
        for inc in incs:
            print(f"    [context] inconsistency: {inc.get('type')} — {inc.get('description')[:80]}")

    print(f"    context: {when_out['time_of_day']} / {when_out['duration']} | {len(locations)} location(s)")
    return when_out, where_out
