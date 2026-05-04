"""Step 4c — Who: character analysis, updates character YAMLs."""

import json
import re
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from llm import call_llm_json
from steps.manual_lore import load_manual_char, load_all_manual_chars, merge_manual_into_char


def _is_valid_json(path: Path) -> bool:
    try:
        json.loads(path.read_text(encoding="utf-8"))
        return True
    except Exception:
        return False

_PROMPT = """\
Analyze the CHARACTERS in this RP scene.

IMPORTANT: Discord authors (who write the messages) are NOT characters. Authors to ignore as characters: {authors}
IMPORTANT: if the scene is too informal / OOC, return struct with empty character list.
IMPORTANT: if an unknown name appears, try to match it to a known character; if you cannot, ignore it.

Discord authors and what they write (use this to identify which author plays which character):
{author_hints}

Known characters (may be incomplete):
{known_yaml}

For each character (not author) active in this scene, extract:

IDENTITY
- canonical_name: full name in lowercase (e.g. "lena marchal")
- author: Discord username who plays this character (e.g. "yaya") — pick from the author hints above
- appellations: all names/references excluding pronouns (e.g. ["lena", "miss marchal", "the chrome mask"])
- description_physical: physical appearance details
- job: occupation or social role
- main_locations: places associated with this character
- relations: relationships with other characters, each as {{"character": "name", "relation": "type"}} (e.g. frère, allié, ennemi)

PSYCHOLOGY
- description_psychological: general personality and behavior summary
- likes: things the character explicitly enjoys, seeks, values (list)
- dislikes: things the character explicitly rejects, avoids, fears (list)
- beliefs: values, convictions, moral stances (list)
- emotional_polarity:
    dominant_emotions: the 1-3 most present emotions in this scene (list of lowercase words)
    emotional_range: "narrow" (controlled/flat) | "moderate" | "wide" (volatile/expressive)
    emotional_triggers: situations or topics that visibly affect this character (list)

PERSONALITY AXES (savoir-être) — score from -2 to +2, 0 = unknown/neutral:
- calm_vs_impulsive:         -2 = very calm/composed, +2 = very impulsive/volatile
- introvert_vs_extrovert:    -2 = very introverted/reserved, +2 = very extroverted/sociable
- cautious_vs_reckless:      -2 = very cautious/methodical, +2 = very reckless/spontaneous
- loyal_vs_treacherous:      -2 = deeply loyal/devoted, +2 = treacherous/self-serving
- compassionate_vs_cold:     -2 = very compassionate/empathetic, +2 = very cold/detached
- honest_vs_deceptive:       -2 = radically honest/blunt, +2 = highly deceptive/manipulative
- humble_vs_arrogant:        -2 = very humble/self-effacing, +2 = very arrogant/grandiose
- trusting_vs_suspicious:    -2 = very trusting/naive, +2 = very suspicious/paranoid
- patient_vs_irritable:      -2 = very patient/tolerant, +2 = very irritable/short-tempered
- conformist_vs_rebel:       -2 = strict conformist/obedient, +2 = strong rebel/defiant
- optimist_vs_pessimist:     -2 = strong optimist, +2 = strong pessimist/fatalist
- pragmatic_vs_idealist:     -2 = pure pragmatist/realist, +2 = pure idealist/dreamer

COMPETENCY AXES (savoir-faire) — "none" | "low" | "medium" | "high" | "exceptional", null if unknown/not shown:

Physical skills:
- combat_melee: close-range fighting (swords, fists, blades)
- combat_ranged: ranged combat (bows, firearms, thrown)
- combat_unarmed: hand-to-hand without weapons
- athleticism: speed, agility, endurance, acrobatics
- stealth: sneaking, hiding, moving silently
- survival: wilderness, tracking, foraging, navigation

Social & mental skills:
- persuasion: convincing others, rhetoric, negotiation
- deception: lying, disguise, manipulation, bluffing
- intimidation: threatening, imposing presence, coercion
- empathy: reading emotions, comforting, understanding others
- leadership: inspiring, commanding, organizing groups
- investigation: deduction, observation, finding clues
- knowledge_general: culture, history, lore, academic learning

Craft & special skills:
- medicine: healing, anatomy, surgery, herbalism
- crafting: building, repairing, smithing, tinkering
- magic_power: raw magical ability or special power level
- magic_control: precision and mastery of magic/abilities
- alchemy_or_science: potions, experiments, technical knowledge
- subterfuge: pickpocketing, lockpicking, trap-setting
- performance: music, acting, storytelling, entertainment

MISC: list of notable characteristics that don't fit above categories
(quirks, habits, physical mannerisms, signature items, speech patterns, etc.)

Only extract what is explicitly shown in THIS scene. Use null for unknowns.
Update known info, correct inconsistencies.

JSON:
{{
  "characters": [{{
    "author": "",
    "canonical_name": "",
    "appellations": [],
    "description_physical": "",
    "job": "",
    "main_locations": [],
    "relations": [{{"character": "", "relation": ""}}],
    "description_psychological": "",
    "likes": [],
    "dislikes": [],
    "beliefs": [],
    "emotional_polarity": {{
      "dominant_emotions": [],
      "emotional_range": "moderate",
      "emotional_triggers": []
    }},
    "personality_axes": {{
      "calm_vs_impulsive": 0,
      "introvert_vs_extrovert": 0,
      "cautious_vs_reckless": 0,
      "loyal_vs_treacherous": 0,
      "compassionate_vs_cold": 0,
      "honest_vs_deceptive": 0,
      "humble_vs_arrogant": 0,
      "trusting_vs_suspicious": 0,
      "patient_vs_irritable": 0,
      "conformist_vs_rebel": 0,
      "optimist_vs_pessimist": 0,
      "pragmatic_vs_idealist": 0
    }},
    "competency_axes": {{
      "combat_melee": null,
      "combat_ranged": null,
      "combat_unarmed": null,
      "athleticism": null,
      "stealth": null,
      "survival": null,
      "persuasion": null,
      "deception": null,
      "intimidation": null,
      "empathy": null,
      "leadership": null,
      "investigation": null,
      "knowledge_general": null,
      "medicine": null,
      "crafting": null,
      "magic_power": null,
      "magic_control": null,
      "alchemy_or_science": null,
      "subterfuge": null,
      "performance": null
    }},
    "misc": []
  }}]
}}

Scene:
---
{text}
---"""


def _slug(name: str) -> str:
    return re.sub(r'\s+', '_', name.lower().strip())


def _load_char_yaml(chars_dir: Path, canonical_name: str) -> dict:
    path = chars_dir / f"{_slug(canonical_name)}.yaml"
    if path.exists():
        with open(path, encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return {}


def _save_char_yaml(chars_dir: Path, canonical_name: str, data: dict):
    chars_dir.mkdir(parents=True, exist_ok=True)
    path = chars_dir / f"{_slug(canonical_name)}.yaml"
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(data, f, allow_unicode=True, sort_keys=False)


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
        if axis not in existing:
            existing[axis] = new_val
        else:
            # Running average weighted toward most recent
            existing[axis] = round((existing[axis] + new_val) / 2)
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
        if not cur_val:
            existing[axis] = new_val
        else:
            # Keep the higher rating
            if _COMPETENCY_RANK.get(new_val, 0) > _COMPETENCY_RANK.get(cur_val, 0):
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
    # Keep only the 5 most recent dominant emotions to avoid bloat
    existing["dominant_emotions"] = existing["dominant_emotions"][-5:]

    if new_ep.get("emotional_range"):
        existing["emotional_range"] = new_ep["emotional_range"]

    _merge_list(existing["emotional_triggers"], new_ep.get("emotional_triggers") or [])
    return existing


def _merge_char(existing: dict, extracted: dict, scene_id: str) -> dict:
    merged = dict(existing)
    merged.setdefault("name", extracted.get("canonical_name", ""))
    merged.setdefault("appellations", [])
    merged.setdefault("description_physical", "")
    merged.setdefault("description_psychological", "")
    merged.setdefault("job", "")
    merged.setdefault("beliefs", [])
    merged.setdefault("likes", [])
    merged.setdefault("dislikes", [])
    merged.setdefault("relations", [])
    merged.setdefault("main_locations", [])
    merged.setdefault("author", "")
    merged.setdefault("emotional_polarity", {})
    merged.setdefault("personality_axes", {})
    merged.setdefault("competency_axes", {})
    merged.setdefault("misc", [])
    merged.setdefault("appearances", [])

    _merge_list(merged["appellations"], extracted.get("appellations") or [])

    for field in ("description_physical", "description_psychological", "job", "author"):
        new_val = (extracted.get(field) or "").strip().lower()
        if new_val and len(new_val) > len(merged.get(field) or ""):
            merged[field] = new_val

    _merge_list(merged["beliefs"],   extracted.get("beliefs") or [])
    _merge_list(merged["likes"],     extracted.get("likes") or [])
    _merge_list(merged["dislikes"],  extracted.get("dislikes") or [])
    _merge_list(merged["main_locations"], extracted.get("main_locations") or [])
    merged["relations"] = _merge_relations(merged["relations"], extracted.get("relations") or [])
    _merge_list(merged["misc"],      extracted.get("misc") or [])

    merged["emotional_polarity"] = _merge_emotional_polarity(
        merged["emotional_polarity"], extracted.get("emotional_polarity") or {}
    )
    merged["personality_axes"] = _merge_personality_axes(
        merged["personality_axes"], extracted.get("personality_axes") or {}
    )
    merged["competency_axes"] = _merge_competency_axes(
        merged["competency_axes"], extracted.get("competency_axes") or {}
    )

    if scene_id not in merged["appearances"]:
        merged["appearances"].append(scene_id)
    merged.setdefault("first_appearance", scene_id)
    return merged


def _scene_text(messages: list[dict]) -> str:
    return "\n".join(
        f"[{(m.get('author') or {}).get('name', '?') if isinstance(m.get('author'), dict) else m.get('author', '?')}]: "
        f"{m.get('content_en') or m.get('content', '')}"
        for m in messages
    )


def _author_hints(messages: list[dict]) -> str:
    """Pre-compute author→content snippets so the LLM can match author↔character reliably."""
    from collections import defaultdict
    groups: dict[str, list[str]] = defaultdict(list)
    for m in messages:
        name = (m.get("author") or {}).get("name", "") if isinstance(m.get("author"), dict) else str(m.get("author", ""))
        content = (m.get("content_en") or m.get("content", "")).strip()
        if name and content:
            groups[name].append(content[:70])
    return "\n".join(f"- {a}: {' / '.join(snippets[:3])}" for a, snippets in groups.items()) or "unknown"


def _merge_relations(existing: list, new_rels: list) -> list:
    """Merge structured relations [{character, relation}]. Dedup by (character, relation)."""
    result = [r for r in existing if isinstance(r, dict) and r.get("character")]
    seen = {(r["character"].lower(), r.get("relation", "").lower()) for r in result}
    for r in (new_rels or []):
        if not isinstance(r, dict) or not r.get("character"):
            continue
        key = (r["character"].lower(), r.get("relation", "").lower())
        if key not in seen:
            result.append({"character": r["character"].lower(), "relation": r.get("relation", "").lower()})
            seen.add(key)
    return result


def run_who(scene_file: Path, analysis_dir: Path, chars_dir: Path) -> dict:
    out_path = analysis_dir / "who.json"
    if out_path.exists() and _is_valid_json(out_path):
        print(f"    [skip] who already done")
        return json.loads(out_path.read_text(encoding="utf-8"))
    if out_path.exists():
        print(f"    [corrupt] who.json malformed, re-analyzing...")

    if not _is_valid_json(scene_file):
        print(f"    [error] scene file {scene_file.name} is malformed, delete it and re-run step 3")
        return {}
    with open(scene_file, encoding="utf-8") as f:
        scene = json.load(f)

    scene_id = scene["scene_id"]
    messages = scene["messages"]
    text = _scene_text(messages)

    # Authors = Discord usernames (not characters)
    authors = list({
        (m.get("author") or {}).get("name", "") if isinstance(m.get("author"), dict) else str(m.get("author", ""))
        for m in messages
    } - {""})

    # Load known characters (LLM-extracted + manual overrides for context)
    known = {}
    if chars_dir.exists():
        for yf in chars_dir.glob("*.yaml"):
            with open(yf, encoding="utf-8") as f:
                d = yaml.safe_load(f) or {}
                if d.get("name"):
                    known[d["name"]] = d
    manual_chars = load_all_manual_chars()
    for name, mc in manual_chars.items():
        if name in known:
            known[name] = merge_manual_into_char(known[name], mc)
        else:
            known[name] = mc

    known_yaml = yaml.dump(known, allow_unicode=True) if known else "none"
    result = call_llm_json(
        _PROMPT.format(
            authors=", ".join(authors),
            author_hints=_author_hints(messages),
            known_yaml=known_yaml,
            text=text,
        ),
        num_predict=2048,
    )

    characters = [c for c in (result.get("characters") or []) if isinstance(c, dict) and c.get("canonical_name")]

    for char in characters:
        print(f"    character extracted: {char['canonical_name']}")

    output = {
        "characters": [c.get("canonical_name") for c in characters],
        "details":    characters,
    }

    analysis_dir.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    return output
