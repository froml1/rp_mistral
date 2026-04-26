"""Step 4b — Entities: who + which in a single LLM call (replaces analyze_who + analyze_which)."""

import json
import re
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from llm import call_llm_json
from steps.manual_lore import load_manual_char, load_all_manual_chars, merge_manual_into_char
from steps.synthesis import current_scene_synthesis
from steps.scene_patch import write_enrichment, chunk_messages
from lore_summary import update_summary, summaries_for_dir
try:
    from store import upsert as _store_upsert
except Exception:
    def _store_upsert(*a, **kw): pass  # store optional at analysis time

_PRONOUNS = {
    # English
    "he", "she", "it", "they", "him", "her", "his", "hers", "their", "theirs",
    "i", "me", "my", "mine", "you", "your", "yours", "we", "us", "our", "ours",
    "this", "that", "these", "those", "who", "whom", "whose", "one", "someone",
    "anyone", "everyone", "nobody", "somebody", "the", "a", "an",
    # French
    "il", "elle", "ils", "elles", "lui", "leur", "leurs",
    "je", "me", "moi", "tu", "te", "toi", "nous", "vous", "on", "se", "soi",
    "ce", "ceci", "cela", "ça", "qui", "que", "quoi", "dont", "où",
    "celui", "celle", "ceux", "celles", "lequel", "laquelle", "lesquels", "lesquelles",
    "le", "la", "les", "un", "une", "des", "du",
}

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
IMPORTANT: Discord authors (who write the messages) are NOT characters. Authors to ignore: {authors}

SCENE SYNTHESIS (use only to resolve identity of characters who appear in the text below — do NOT add characters from here if they are absent from the scene text):
{synthesis}

CHARACTERS FROM THE IMMEDIATELY PRECEDING SCENE: {prev_scene_chars}
If this scene contains only pronouns (il/elle/ils/he/she/they) with no explicit character names, AND the theme, location, and tone seem to directly continue from the previous scene, then these are likely the same characters. Only apply this if the continuity is clear.

CHARACTERS AND CONCEPTS ALREADY IDENTIFIED IN EARLIER PARTS OF THIS SAME SCENE (use to resolve pronouns and references — do not duplicate):
{prior_chunk_context}

Discord authors and what they write (use this to identify which author plays which character):
{author_hints}

Known characters (from corpus sweep — for identity anchoring only, do NOT include them unless they actively appear in the scene text below):
{known_chars_yaml}

── CHARACTERS ──────────────────────────────────────────────────────────────────
STRICT RULE: only extract characters who SPEAK or perform *actions* directly in the scene text below.
Do NOT extract characters who are only mentioned, referenced, or talked about by others.
For each character PHYSICALLY PRESENT AND ACTIVE in this scene:

Identity: canonical_name (lowercase full name), author (Discord username who plays this character — from the hints above), gender (male|female|non-binary|unknown), age (e.g. "young adult", "elderly", "immortal", "unknown"), species (e.g. "human", "elf", "demon", "unknown"), appellations (all references), description_physical, job, main_locations
affiliations: factions, guilds, orders the character belongs to (list of names)
relations: list of relationships as [{{"character": "name", "relation": "type"}}] (e.g. frère, allié, ennemi)
goals: {{"short_term": ["visible objective in this scene"], "long_term": ["deeper ambition if mentioned"]}}
wounds: past traumas or injuries explicitly mentioned or strongly implied (list of short descriptions)
secrets: information the character conceals — detectable by contradiction between words and actions (list)
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

── SPEAKER ATTRIBUTION ──────────────────────────────────────────────────────────
For each Discord author, identify which character they play in this scene.
author_to_character: {{"discord_author": "canonical_character_name"}}
If a message looks clearly OOC (out-of-character, player talking as themselves), list its
0-based index in ooc_messages.

── INCONSISTENCIES ──────────────────────────────────────────────────────────────
List any clear problem you notice: wrong character name used, character appears in two
places at once, author seems to play two unrelated characters in the same scene, etc.
inconsistencies: [{{"message_idx": int_or_null, "type": "ooc|wrong_name|split_author|character_conflict|other", "description": "..."}}]

JSON:
{{
  "author_to_character": {{}},
  "ooc_messages": [],
  "characters": [{{
    "canonical_name": "", "author": "", "gender": "unknown", "age": "unknown", "species": "unknown",
    "appellations": [], "description_physical": "", "job": "", "main_locations": [],
    "affiliations": [], "relations": [{{"character": "", "relation": ""}}],
    "goals": {{"short_term": [], "long_term": []}},
    "wounds": [], "secrets": [],
    "description_psychological": "", "likes": [], "dislikes": [],
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
  "inconsistencies": []
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
    low = [str(x).lower() for x in existing]
    for item in (new_items or []):
        if not item:
            continue
        item_s = str(item)
        if item_s.lower() not in low:
            existing.append(item_s.lower())
            low.append(item_s.lower())
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
    for field in ("appellations", "beliefs", "likes", "dislikes", "main_locations", "misc", "appearances",
                  "affiliations", "wounds", "secrets"):
        merged.setdefault(field, [])
    merged.setdefault("relations", [])
    merged.setdefault("goals", {"short_term": [], "long_term": []})
    for field in ("description_physical", "description_psychological", "job", "author", "gender", "age", "species"):
        merged.setdefault(field, "")
    merged.setdefault("emotional_polarity", {})
    merged.setdefault("personality_axes", {})
    merged.setdefault("competency_axes", {})

    _merge_list(merged["appellations"], extracted.get("appellations") or [])
    for field in ("description_physical", "description_psychological", "job", "author"):
        new_val = (extracted.get(field) or "").strip().lower()
        if new_val and len(new_val) > len(merged.get(field) or ""):
            merged[field] = new_val
    for single_field in ("gender", "age", "species"):
        new_val = (extracted.get(single_field) or "").strip().lower()
        if new_val and new_val != "unknown":
            merged[single_field] = new_val
    for field in ("beliefs", "likes", "dislikes", "main_locations", "misc", "affiliations", "wounds", "secrets"):
        _merge_list(merged[field], extracted.get(field) or [])
    # Merge goals
    new_goals = extracted.get("goals") or {}
    _merge_list(merged["goals"].setdefault("short_term", []), new_goals.get("short_term") or [])
    _merge_list(merged["goals"].setdefault("long_term", []),  new_goals.get("long_term")  or [])
    merged["relations"] = _merge_relations(merged.get("relations") or [], extracted.get("relations") or [])

    merged["emotional_polarity"] = _merge_emotional_polarity(merged["emotional_polarity"], extracted.get("emotional_polarity") or {})
    merged["personality_axes"]   = _merge_personality_axes(merged["personality_axes"],   extracted.get("personality_axes") or {})
    merged["competency_axes"]    = _merge_competency_axes(merged["competency_axes"],    extracted.get("competency_axes") or {})

    if scene_id not in merged["appearances"]:
        merged["appearances"].append(scene_id)
    merged.setdefault("first_appearance", scene_id)
    return merged



# ── Main entry ────────────────────────────────────────────────────────────────

def _prev_scene_characters(analysis_dir: Path) -> list[str]:
    """Return character names from the immediately preceding scene's who.json."""
    parent = analysis_dir.parent
    try:
        siblings = sorted(d for d in parent.iterdir() if d.is_dir())
        idx = siblings.index(analysis_dir)
    except (ValueError, FileNotFoundError):
        return []
    if idx == 0:
        return []
    prev_who = siblings[idx - 1] / "who.json"
    if not prev_who.exists():
        return []
    try:
        data = json.loads(prev_who.read_text(encoding="utf-8"))
        return [c for c in (data.get("characters") or []) if c]
    except Exception:
        return []


def _scene_text(messages: list[dict]) -> str:
    return "\n".join(
        f"[{(m.get('author') or {}).get('name', '?') if isinstance(m.get('author'), dict) else m.get('author', '?')}]: "
        f"{m.get('content_en') or m.get('content', '')}"
        for m in messages
    )


def _author_hints(messages: list[dict]) -> str:
    from collections import defaultdict
    groups: dict[str, list[str]] = defaultdict(list)
    for m in messages:
        name = (m.get("author") or {}).get("name", "") if isinstance(m.get("author"), dict) else str(m.get("author", ""))
        content = (m.get("content_en") or m.get("content", "")).strip()
        if name and content:
            groups[name].append(content[:70])
    return "\n".join(f"- {a}: {' / '.join(snippets[:3])}" for a, snippets in groups.items()) or "unknown"


def _merge_relations(existing: list, new_rels: list) -> list:
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


def _merge_entities(results: list[dict]) -> dict:
    if len(results) == 1:
        return results[0]
    # Merge characters by canonical_name
    char_map: dict[str, dict] = {}
    for r in results:
        for c in (r.get("characters") or []):
            if not isinstance(c, dict) or not c.get("canonical_name"):
                continue
            name = c["canonical_name"].lower()
            if name not in char_map:
                char_map[name] = dict(c)
            else:
                ex = char_map[name]
                for field in ("description_physical", "description_psychological", "job", "author"):
                    v = (c.get(field) or "").strip()
                    if v and len(v) > len(ex.get(field) or ""):
                        ex[field] = v
                for field in ("appellations", "likes", "dislikes", "beliefs", "main_locations", "misc",
                              "affiliations", "wounds", "secrets"):
                    existing_low = {str(x).lower() for x in (ex.get(field) or [])}
                    for item in (c.get(field) or []):
                        if not item:
                            continue
                        item_s = str(item)
                        if item_s.lower() not in existing_low:
                            ex.setdefault(field, []).append(item_s)
                            existing_low.add(item_s.lower())
                ex["relations"] = _merge_relations(ex.get("relations") or [], c.get("relations") or [])
                # Merge goals
                new_goals = c.get("goals") or {}
                for sub in ("short_term", "long_term"):
                    ex_sub = ex.setdefault("goals", {}).setdefault(sub, [])
                    ex_low = {x.lower() for x in ex_sub}
                    for item in (new_goals.get(sub) or []):
                        if item and item.lower() not in ex_low:
                            ex_sub.append(item); ex_low.add(item.lower())
                # Single-value fields: take non-unknown value
                for sf in ("gender", "age", "species"):
                    v = (c.get(sf) or "").strip().lower()
                    if v and v != "unknown" and not ex.get(sf):
                        ex[sf] = v

    # Merge concepts by canonical_name
    concept_map: dict[str, dict] = {}
    for r in results:
        for c in (r.get("concepts") or []):
            if not isinstance(c, dict) or not c.get("canonical_name"):
                continue
            name = c["canonical_name"].lower()
            if name not in concept_map:
                concept_map[name] = dict(c)
            else:
                ex = concept_map[name]
                if len(c.get("description", "")) > len(ex.get("description", "")):
                    ex["description"] = c["description"]

    a2c: dict[str, str] = {}
    for r in results:
        a2c.update(r.get("author_to_character") or {})

    ooc = list(dict.fromkeys(i for r in results for i in (r.get("ooc_messages") or []) if isinstance(i, int)))
    incs = [i for r in results for i in (r.get("inconsistencies") or []) if isinstance(i, dict)]

    return {
        "characters":         list(char_map.values()),
        "concepts":           list(concept_map.values()),
        "author_to_character": a2c,
        "ooc_messages":       ooc,
        "inconsistencies":    incs,
    }


def run_entities(scene_file: Path, analysis_dir: Path, chars_dir: Path, concepts_dir: Path, lore_dir: Path | None = None) -> tuple[dict, dict]:
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

    known_chars_yaml = "\n".join(f"- {n}: {d.get('_summary', '')}" for n, d in known_chars.items()) or "none"
    synthesis           = current_scene_synthesis(lore_dir, scene_id) if lore_dir else "none"
    authors_str         = ", ".join(authors)
    prev_chars          = _prev_scene_characters(analysis_dir)
    prev_scene_chars_str = ", ".join(prev_chars) if prev_chars else "none"

    chunks = chunk_messages(messages)
    if len(chunks) > 1:
        print(f"    entities: {len(chunks)} chunks")
    raw_results = []
    prior_chunk_context = "none"
    for chunk in chunks:
        r = call_llm_json(
            _PROMPT.format(
                authors=authors_str,
                synthesis=synthesis,
                prev_scene_chars=prev_scene_chars_str,
                prior_chunk_context=prior_chunk_context,
                author_hints=_author_hints(chunk),
                known_chars_yaml=known_chars_yaml,
                text=_scene_text(chunk),
            ),
            num_predict=6144,
            num_ctx=12288,
        )
        raw_results.append(r)
        # Build context for next chunk: names + roles found so far
        char_lines = [
            f"- {c.get('canonical_name', '')} ({c.get('job', '') or c.get('description_psychological', '')[:40]})"
            for c in (r.get("characters") or []) if c.get("canonical_name")
        ]
        concept_lines = [
            f"- {c.get('canonical_name', '')} [{c.get('type', '')}]"
            for c in (r.get("concepts") or []) if c.get("canonical_name")
        ]
        prior_chunk_context = "\n".join(char_lines + concept_lines) or "none"
    result = _merge_entities(raw_results)

    # — Characters —
    authors_lower = {a.lower() for a in authors}
    # Reverse author_to_character for back-fill: char_name_lower → discord_author
    a2c_raw = result.get("author_to_character") or {}
    # Keep only attributions where the discord author actually wrote in this scene
    a2c_raw = {k: v for k, v in a2c_raw.items() if k and v and k.lower() in authors_lower}
    char_to_author = {v.lower(): k for k, v in a2c_raw.items()}

    characters = []
    for c in (result.get("characters") or []):
        if not isinstance(c, dict) or not c.get("canonical_name"):
            continue
        name = c["canonical_name"].lower().strip()
        if name in authors_lower:
            print(f"    [skip] '{c['canonical_name']}' is a Discord author, not a character")
            continue
        # Reject characters with no author attribution AND no known presence in the scene
        # (author field set means the LLM saw them active; back-fill covers the rest)
        assigned_author = (c.get("author") or "").lower()
        if assigned_author and assigned_author not in authors_lower:
            print(f"    [skip] '{c['canonical_name']}' attributed to absent author '{assigned_author}'")
            continue
        # Filter appellations: remove pronouns and author names
        if "appellations" in c:
            c["appellations"] = [
                a for a in (c["appellations"] or [])
                if a and a.lower().strip() not in _PRONOUNS and a.lower().strip() not in authors_lower
            ]
        # Back-fill author from author_to_character if not already set
        if not c.get("author"):
            c["author"] = char_to_author.get(name, "")
        characters.append(c)

    for char in characters:
        existing = _load_char_yaml(chars_dir, char["canonical_name"])
        merged   = _merge_char(existing, char, scene_id)
        merged   = merge_manual_into_char(merged, load_manual_char(char["canonical_name"]))
        _save_char_yaml(chars_dir, char["canonical_name"], merged)
        summary  = update_summary(chars_dir / f"{_slug(char['canonical_name'])}.yaml", "characters")
        _store_upsert("characters", merged["name"], summary)
        print(f"    character updated: {char['canonical_name']}")

    who_out = {"characters": [c.get("canonical_name") for c in characters], "details": characters}

    analysis_dir.mkdir(parents=True, exist_ok=True)
    who_path.write_text(json.dumps(who_out, ensure_ascii=False, indent=2), encoding="utf-8")

    # — Concepts: dedicated full-scene sweep —
    from steps.analyze_which import run_which
    where_path = analysis_dir / "where.json"
    known_places = []
    if where_path.exists():
        try:
            wd = json.loads(where_path.read_text(encoding="utf-8"))
            known_places = [d.get("canonical_name") or d.get("name", "") for d in (wd.get("details") or [])]
            known_places = [p for p in known_places if p]
        except Exception:
            pass
    which_out = run_which(
        scene_file, analysis_dir, concepts_dir,
        known_chars=who_out["characters"],
        known_places=known_places,
        lore_dir=lore_dir,
    )

    # Write speaker attribution + inconsistencies back to the scene file
    enrichment: dict = {}
    a2c = result.get("author_to_character")
    if isinstance(a2c, dict) and a2c:
        enrichment["author_to_character"] = {k.lower(): v.lower() for k, v in a2c.items() if k and v}
    ooc = [i for i in (result.get("ooc_messages") or []) if isinstance(i, int)]
    if ooc:
        enrichment["ooc_messages"] = ooc
    incs = [i for i in (result.get("inconsistencies") or []) if isinstance(i, dict) and i.get("description")]
    if incs:
        enrichment["inconsistencies"] = incs
        for inc in incs:
            print(f"    [entities] inconsistency: {inc.get('type')} — {inc.get('description')[:80]}")
    if enrichment:
        write_enrichment(scene_file, "entities", enrichment)

    print(f"    entities: {len(characters)} character(s), {len(which_out.get('concepts') or [])} concept(s)")
    return who_out, which_out
