"""Step 4b — Entities: who + which in a single LLM call (replaces analyze_who + analyze_which)."""

import json
import re
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from llm import call_llm_json
from steps.manual_lore import (
    load_manual_char, load_all_manual_chars, merge_manual_into_char,
    load_manual_concept, load_all_manual_concepts, merge_manual_into_concept,
)
from lore_summary import update_summary, summaries_for_dir
try:
    from store import upsert as _store_upsert
except Exception:
    def _store_upsert(*a, **kw): pass  # store optional at analysis time

_PERSONALITY_AXES = [
    "calm_vs_impulsive", "introvert_vs_extrovert", "cautious_vs_reckless",
    "loyal_vs_treacherous", "compassionate_vs_cold", "honest_vs_deceptive",
    "humble_vs_arrogant", "trusting_vs_suspicious", "patient_vs_irritable",
    "conformist_vs_rebel", "optimist_vs_pessimist", "pragmatic_vs_idealist",
]
_COMPETENCY_AXES = [
    "combat_melee", "combat_ranged", "combat_unarmed", "athleticism", "stealth", "survival",
    "persuasion", "deception", "intimidation", "empathy", "leadership",
    "investigation", "knowledge_general",
    "medicine", "crafting", "magic_power", "magic_control",
    "alchemy_or_science", "subterfuge", "performance",
]
_COMPETENCY_RANK = {"none": 0, "low": 1, "medium": 2, "high": 3, "exceptional": 4}


def _is_valid_json(path: Path) -> bool:
    try:
        json.loads(path.read_text(encoding="utf-8"))
        return True
    except Exception:
        return False


_PROMPT = """\
Analyze the CHARACTERS and CONCEPTS present in this RP scene in one pass.
IMPORTANT: Discord authors (who write the scene) are NOT characters. Authors to ignore: {authors}

Known characters (may be incomplete):
{known_chars_yaml}

Known concepts (may be incomplete):
{known_concepts_yaml}

── CHARACTERS ──────────────────────────────────────────────────────────────────
For each character (not author) present or active in this scene:

Identity: canonical_name (lowercase full name), appellations (all references), description_physical, job, main_locations
Psychology: description_psychological, likes (list), dislikes (list), beliefs (list)
Emotional polarity: dominant_emotions (1-3 emotions), emotional_range (narrow|moderate|wide), emotional_triggers (list)

Personality axes — score -2 to +2 (0 = unknown):
calm_vs_impulsive, introvert_vs_extrovert, cautious_vs_reckless, loyal_vs_treacherous,
compassionate_vs_cold, honest_vs_deceptive, humble_vs_arrogant, trusting_vs_suspicious,
patient_vs_irritable, conformist_vs_rebel, optimist_vs_pessimist, pragmatic_vs_idealist

Competency axes — none|low|medium|high|exceptional (null = unknown):
Physical: combat_melee, combat_ranged, combat_unarmed, athleticism, stealth, survival
Social/mental: persuasion, deception, intimidation, empathy, leadership, investigation, knowledge_general
Craft/special: medicine, crafting, magic_power, magic_control, alchemy_or_science, subterfuge, performance

misc: notable traits not covered above (quirks, habits, signature items, speech patterns)

── CONCEPTS ─────────────────────────────────────────────────────────────────────
Named, specific elements that are neither characters nor locations nor events.
Include: named objects of significance, factions/organizations, ideologies, laws/systems, technologies, rituals, artifacts.
Exclude: generic items, pronouns, vague references, purely incidental elements.

For each concept: canonical_name, type (object|faction|ideology|system|artifact|other),
appellations, description, related_characters, significance (one sentence).

JSON:
{{
  "characters": [{{
    "canonical_name": "", "appellations": [], "description_physical": "", "job": "",
    "main_locations": [], "description_psychological": "", "likes": [], "dislikes": [],
    "beliefs": [],
    "emotional_polarity": {{"dominant_emotions": [], "emotional_range": "moderate", "emotional_triggers": []}},
    "personality_axes": {{"calm_vs_impulsive": 0, "introvert_vs_extrovert": 0, "cautious_vs_reckless": 0,
      "loyal_vs_treacherous": 0, "compassionate_vs_cold": 0, "honest_vs_deceptive": 0,
      "humble_vs_arrogant": 0, "trusting_vs_suspicious": 0, "patient_vs_irritable": 0,
      "conformist_vs_rebel": 0, "optimist_vs_pessimist": 0, "pragmatic_vs_idealist": 0}},
    "competency_axes": {{"combat_melee": null, "combat_ranged": null, "combat_unarmed": null,
      "athleticism": null, "stealth": null, "survival": null, "persuasion": null, "deception": null,
      "intimidation": null, "empathy": null, "leadership": null, "investigation": null,
      "knowledge_general": null, "medicine": null, "crafting": null, "magic_power": null,
      "magic_control": null, "alchemy_or_science": null, "subterfuge": null, "performance": null}},
    "misc": []
  }}],
  "concepts": [{{"canonical_name": "", "type": "", "appellations": [], "description": "", "related_characters": [], "significance": ""}}]
}}

Scene:
---
{text}
---"""


def _slug(name: str) -> str:
    return re.sub(r'\s+', '_', name.lower().strip())


# ── Character helpers ─────────────────────────────────────────────────────────

def _load_char_yaml(chars_dir: Path, name: str) -> dict:
    path = chars_dir / f"{_slug(name)}.yaml"
    if path.exists():
        with open(path, encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return {}


def _save_char_yaml(chars_dir: Path, name: str, data: dict):
    chars_dir.mkdir(parents=True, exist_ok=True)
    path = chars_dir / f"{_slug(name)}.yaml"
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(data, f, allow_unicode=True, sort_keys=False)


def _merge_list(existing: list, new_items: list) -> list:
    low = [x.lower() for x in existing]
    for item in (new_items or []):
        if item and item.lower() not in low:
            existing.append(item.lower())
            low.append(item.lower())
    return existing


def _merge_personality_axes(existing: dict, new_axes: dict) -> dict:
    if not new_axes:
        return existing
    for axis in _PERSONALITY_AXES:
        new_val = new_axes.get(axis)
        if new_val is None:
            continue
        try:
            new_val = int(new_val)
        except (TypeError, ValueError):
            continue
        existing[axis] = round((existing[axis] + new_val) / 2) if axis in existing else new_val
    return existing


def _merge_competency_axes(existing: dict, new_axes: dict) -> dict:
    if not new_axes:
        return existing
    for axis in _COMPETENCY_AXES:
        new_val = new_axes.get(axis)
        if not new_val:
            continue
        new_val = str(new_val).lower()
        cur_val = existing.get(axis)
        if not cur_val or _COMPETENCY_RANK.get(new_val, 0) > _COMPETENCY_RANK.get(cur_val, 0):
            existing[axis] = new_val
    return existing


def _merge_emotional_polarity(existing: dict, new_ep: dict) -> dict:
    if not new_ep:
        return existing
    existing = dict(existing)
    existing.setdefault("dominant_emotions", [])
    existing.setdefault("emotional_range", "moderate")
    existing.setdefault("emotional_triggers", [])
    _merge_list(existing["dominant_emotions"], new_ep.get("dominant_emotions") or [])
    existing["dominant_emotions"] = existing["dominant_emotions"][-5:]
    if new_ep.get("emotional_range"):
        existing["emotional_range"] = new_ep["emotional_range"]
    _merge_list(existing["emotional_triggers"], new_ep.get("emotional_triggers") or [])
    return existing


def _merge_char(existing: dict, extracted: dict, scene_id: str) -> dict:
    merged = dict(existing)
    merged.setdefault("name", extracted.get("canonical_name", ""))
    for field in ("appellations", "beliefs", "likes", "dislikes", "main_locations", "misc", "appearances"):
        merged.setdefault(field, [])
    for field in ("description_physical", "description_psychological", "job"):
        merged.setdefault(field, "")
    merged.setdefault("emotional_polarity", {})
    merged.setdefault("personality_axes", {})
    merged.setdefault("competency_axes", {})

    _merge_list(merged["appellations"], extracted.get("appellations") or [])
    for field in ("description_physical", "description_psychological", "job"):
        new_val = (extracted.get(field) or "").strip().lower()
        if new_val and len(new_val) > len(merged.get(field) or ""):
            merged[field] = new_val
    for field in ("beliefs", "likes", "dislikes", "main_locations", "misc"):
        _merge_list(merged[field], extracted.get(field) or [])

    merged["emotional_polarity"] = _merge_emotional_polarity(merged["emotional_polarity"], extracted.get("emotional_polarity") or {})
    merged["personality_axes"]   = _merge_personality_axes(merged["personality_axes"],   extracted.get("personality_axes") or {})
    merged["competency_axes"]    = _merge_competency_axes(merged["competency_axes"],    extracted.get("competency_axes") or {})

    if scene_id not in merged["appearances"]:
        merged["appearances"].append(scene_id)
    merged.setdefault("first_appearance", scene_id)
    return merged


# ── Concept helpers ───────────────────────────────────────────────────────────

def _load_concept_yaml(concepts_dir: Path, name: str) -> dict:
    path = concepts_dir / f"{_slug(name)}.yaml"
    if path.exists():
        with open(path, encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return {}


def _save_concept_yaml(concepts_dir: Path, name: str, data: dict):
    concepts_dir.mkdir(parents=True, exist_ok=True)
    path = concepts_dir / f"{_slug(name)}.yaml"
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(data, f, allow_unicode=True, sort_keys=False)


def _merge_concept(existing: dict, extracted: dict, scene_id: str) -> dict:
    merged = dict(existing)
    merged.setdefault("name", extracted.get("canonical_name", ""))
    merged.setdefault("type", extracted.get("type", "other"))
    merged.setdefault("appellations", [])
    merged.setdefault("description", "")
    merged.setdefault("significance", "")
    merged.setdefault("related_characters", [])
    merged.setdefault("appearances", [])

    for app in (extracted.get("appellations") or []):
        if app.lower() not in [a.lower() for a in merged["appellations"]]:
            merged["appellations"].append(app.lower())

    for field in ("description", "significance"):
        new_val = extracted.get(field) or ""
        if new_val and len(new_val) > len(merged[field]):
            merged[field] = new_val.lower()

    for char in (extracted.get("related_characters") or []):
        if char.lower() not in [c.lower() for c in merged["related_characters"]]:
            merged["related_characters"].append(char.lower())

    if scene_id not in merged["appearances"]:
        merged["appearances"].append(scene_id)
    merged.setdefault("first_appearance", scene_id)
    return merged


# ── Main entry ────────────────────────────────────────────────────────────────

def _scene_text(messages: list[dict]) -> str:
    return "\n".join(
        f"{(m.get('author') or {}).get('name', '?') if isinstance(m.get('author'), dict) else m.get('author', '?')}: "
        f"{m.get('content_en') or m.get('content', '')}"
        for m in messages
    )


def run_entities(scene_file: Path, analysis_dir: Path, chars_dir: Path, concepts_dir: Path) -> tuple[dict, dict]:
    """Returns (who_dict, which_dict). Writes who.json and which.json."""
    who_path   = analysis_dir / "who.json"
    which_path = analysis_dir / "which.json"

    who_done   = who_path.exists()   and _is_valid_json(who_path)
    which_done = which_path.exists() and _is_valid_json(which_path)

    if who_done and which_done:
        print("    [skip] entities already done")
        return (
            json.loads(who_path.read_text(encoding="utf-8")),
            json.loads(which_path.read_text(encoding="utf-8")),
        )

    if not _is_valid_json(scene_file):
        print(f"    [error] scene file {scene_file.name} is malformed")
        return {}, {}

    with open(scene_file, encoding="utf-8") as f:
        scene = json.load(f)

    scene_id = scene["scene_id"]
    messages = scene["messages"]
    text = _scene_text(messages)

    authors = list({
        (m.get("author") or {}).get("name", "") if isinstance(m.get("author"), dict) else str(m.get("author", ""))
        for m in messages
    } - {""})

    # Known characters with manual overrides
    known_chars = {}
    if chars_dir.exists():
        for yf in chars_dir.glob("*.yaml"):
            with open(yf, encoding="utf-8") as f:
                d = yaml.safe_load(f) or {}
                if d.get("name"):
                    known_chars[d["name"]] = d
    for name, mc in load_all_manual_chars().items():
        known_chars[name] = merge_manual_into_char(known_chars.get(name, {}), mc) if name in known_chars else mc

    # Known concepts with manual overrides
    known_concepts = {}
    if concepts_dir.exists():
        for yf in concepts_dir.glob("*.yaml"):
            with open(yf, encoding="utf-8") as f:
                d = yaml.safe_load(f) or {}
                if d.get("name"):
                    known_concepts[d["name"]] = d
    for name, mc in load_all_manual_concepts().items():
        known_concepts[name] = merge_manual_into_concept(known_concepts.get(name, {}), mc) if name in known_concepts else mc

    # Inject only summary cards — keeps the prompt compact regardless of corpus size
    known_chars_yaml    = "\n".join(f"- {n}: {d.get('_summary', '')}" for n, d in known_chars.items())    or "none"
    known_concepts_yaml = "\n".join(f"- {n}: {d.get('_summary', '')}" for n, d in known_concepts.items()) or "none"

    result = call_llm_json(
        _PROMPT.format(
            authors=", ".join(authors),
            known_chars_yaml=known_chars_yaml,
            known_concepts_yaml=known_concepts_yaml,
            text=text,
        ),
        num_predict=3072,
        num_ctx=8192,
    )

    # — Characters —
    characters = [c for c in (result.get("characters") or []) if isinstance(c, dict) and c.get("canonical_name")]
    for char in characters:
        existing = _load_char_yaml(chars_dir, char["canonical_name"])
        merged   = _merge_char(existing, char, scene_id)
        merged   = merge_manual_into_char(merged, load_manual_char(char["canonical_name"]))
        _save_char_yaml(chars_dir, char["canonical_name"], merged)
        summary  = update_summary(chars_dir / f"{_slug(char['canonical_name'])}.yaml", "characters")
        _store_upsert("characters", merged["name"], summary)
        print(f"    character updated: {char['canonical_name']}")

    who_out = {"characters": [c.get("canonical_name") for c in characters], "details": characters}

    # — Concepts —
    concepts = [c for c in (result.get("concepts") or []) if isinstance(c, dict) and c.get("canonical_name")]
    for concept in concepts:
        existing = _load_concept_yaml(concepts_dir, concept["canonical_name"])
        merged   = _merge_concept(existing, concept, scene_id)
        merged   = merge_manual_into_concept(merged, load_manual_concept(concept["canonical_name"]))
        _save_concept_yaml(concepts_dir, concept["canonical_name"], merged)
        summary  = update_summary(concepts_dir / f"{_slug(concept['canonical_name'])}.yaml", "concepts")
        _store_upsert("concepts", merged["name"], summary)
        print(f"    concept updated: {concept['canonical_name']}")

    which_out = {"concepts": [c.get("canonical_name") for c in concepts], "details": concepts}

    analysis_dir.mkdir(parents=True, exist_ok=True)
    who_path.write_text(json.dumps(who_out, ensure_ascii=False, indent=2), encoding="utf-8")
    which_path.write_text(json.dumps(which_out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"    entities: {len(characters)} character(s), {len(concepts)} concept(s)")
    return who_out, which_out
