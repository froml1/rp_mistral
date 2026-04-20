"""
Manual lore additions — user-supplied ground-truth data stored separately
from LLM-extracted data and merged with higher priority.

Structure:
  data/lore/manual/characters/{slug}.yaml
  data/lore/manual/places/{slug}.yaml
  data/lore/manual/concepts/{slug}.yaml
  data/lore/manual/events/{slug}.yaml   (keyed by scene_id or topic)
"""

import re
from pathlib import Path

import yaml

MANUAL_DIR = Path(__file__).parent.parent.parent / "data" / "lore" / "manual"


def _slug(name: str) -> str:
    return re.sub(r'[^\w]', '_', name.lower().strip())


def _load(path: Path) -> dict:
    try:
        return yaml.safe_load(path.read_text(encoding="utf-8")) or {} if path.exists() else {}
    except Exception:
        return {}


def _save(path: Path, data: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(data, f, allow_unicode=True, sort_keys=False)


# ── Characters ────────────────────────────────────────────────────────────────

def manual_char_path(name: str) -> Path:
    return MANUAL_DIR / "characters" / f"{_slug(name)}.yaml"


def load_manual_char(name: str) -> dict:
    return _load(manual_char_path(name))


def save_manual_char(name: str, data: dict):
    data["name"] = name.lower().strip()
    data["_manual"] = True
    _save(manual_char_path(name), data)


def load_all_manual_chars() -> dict:
    d = MANUAL_DIR / "characters"
    if not d.exists():
        return {}
    result = {}
    for f in d.glob("*.yaml"):
        entry = _load(f)
        if entry.get("name"):
            result[entry["name"]] = entry
    return result


# ── Places ────────────────────────────────────────────────────────────────────

def manual_place_path(name: str) -> Path:
    return MANUAL_DIR / "places" / f"{_slug(name)}.yaml"


def load_manual_place(name: str) -> dict:
    return _load(manual_place_path(name))


def save_manual_place(name: str, data: dict):
    data["name"] = name.lower().strip()
    data["_manual"] = True
    _save(manual_place_path(name), data)


def load_all_manual_places() -> dict:
    d = MANUAL_DIR / "places"
    if not d.exists():
        return {}
    result = {}
    for f in d.glob("*.yaml"):
        entry = _load(f)
        if entry.get("name"):
            result[entry["name"]] = entry
    return result


# ── Concepts ─────────────────────────────────────────────────────────────────

def manual_concept_path(name: str) -> Path:
    return MANUAL_DIR / "concepts" / f"{_slug(name)}.yaml"


def load_manual_concept(name: str) -> dict:
    return _load(manual_concept_path(name))


def save_manual_concept(name: str, data: dict):
    data["name"] = name.lower().strip()
    data["_manual"] = True
    _save(manual_concept_path(name), data)


def load_all_manual_concepts() -> dict:
    d = MANUAL_DIR / "concepts"
    if not d.exists():
        return {}
    result = {}
    for f in d.glob("*.yaml"):
        entry = _load(f)
        if entry.get("name"):
            result[entry["name"]] = entry
    return result


# ── Events (what) ─────────────────────────────────────────────────────────────

def manual_events_path(scene_id: str) -> Path:
    return MANUAL_DIR / "events" / f"{scene_id}.yaml"


def load_manual_events(scene_id: str) -> dict:
    return _load(manual_events_path(scene_id))


def save_manual_events(scene_id: str, data: dict):
    data["scene_id"] = scene_id
    data["_manual"] = True
    _save(manual_events_path(scene_id), data)


# ── Merge helpers (manual takes priority over LLM) ───────────────────────────

def _merge_str(manual_val: str, llm_val: str) -> str:
    """Manual wins if non-empty."""
    return manual_val.strip() if manual_val and manual_val.strip() else llm_val


def _merge_list_priority(manual: list, llm: list) -> list:
    """Manual items first, then LLM items not already present."""
    result = [x.lower() for x in (manual or [])]
    seen = set(result)
    for item in (llm or []):
        if item.lower() not in seen:
            result.append(item.lower())
            seen.add(item.lower())
    return result


def _merge_dict_priority(manual: dict, llm: dict) -> dict:
    """Manual values override LLM values key by key."""
    merged = dict(llm or {})
    for k, v in (manual or {}).items():
        if v is not None and v != "" and v != [] and v != {}:
            merged[k] = v
    return merged


def merge_manual_into_char(llm_char: dict, manual_char: dict) -> dict:
    """Merge manual character data into LLM-extracted character, manual wins."""
    if not manual_char:
        return llm_char
    result = dict(llm_char)
    for field in ("description_physical", "description_psychological", "job"):
        result[field] = _merge_str(manual_char.get(field, ""), result.get(field, ""))
    for field in ("appellations", "beliefs", "likes", "dislikes", "main_locations", "misc"):
        result[field] = _merge_list_priority(manual_char.get(field, []), result.get(field, []))
    if manual_char.get("personality_axes"):
        result["personality_axes"] = _merge_dict_priority(
            manual_char["personality_axes"], result.get("personality_axes", {})
        )
    if manual_char.get("competency_axes"):
        result["competency_axes"] = _merge_dict_priority(
            manual_char["competency_axes"], result.get("competency_axes", {})
        )
    if manual_char.get("emotional_polarity"):
        result["emotional_polarity"] = _merge_dict_priority(
            manual_char["emotional_polarity"], result.get("emotional_polarity", {})
        )
    return result


def merge_manual_into_place(llm_place: dict, manual_place: dict) -> dict:
    if not manual_place:
        return llm_place
    result = dict(llm_place)
    result["description"] = _merge_str(manual_place.get("description", ""), result.get("description", ""))
    result["appellations"] = _merge_list_priority(manual_place.get("appellations", []), result.get("appellations", []))
    result["attributes"]   = _merge_list_priority(manual_place.get("attributes", []),   result.get("attributes", []))
    return result


def merge_manual_into_concept(llm_concept: dict, manual_concept: dict) -> dict:
    if not manual_concept:
        return llm_concept
    result = dict(llm_concept)
    for field in ("description", "significance", "type"):
        result[field] = _merge_str(manual_concept.get(field, ""), result.get(field, ""))
    result["appellations"]      = _merge_list_priority(manual_concept.get("appellations", []),      result.get("appellations", []))
    result["related_characters"] = _merge_list_priority(manual_concept.get("related_characters", []), result.get("related_characters", []))
    return result


def merge_manual_into_what(llm_what: dict, manual_events: dict) -> dict:
    if not manual_events:
        return llm_what
    result = dict(llm_what)
    # Prepend manual summary if provided
    manual_summary = manual_events.get("summary", "").strip()
    if manual_summary:
        result["summary"] = manual_summary + (" — " + result.get("summary", "") if result.get("summary") else "")
    # Prepend manual events (they are gold-standard, listed first)
    manual_evs = manual_events.get("events") or []
    llm_evs    = result.get("events") or []
    result["events"] = manual_evs + llm_evs
    return result
