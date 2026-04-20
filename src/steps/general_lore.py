"""Global lore synthesis — updated after each scene, re-injected as context for the next ones."""

import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from llm import call_llm_json


# ---------------------------------------------------------------------------
# Who / Where / Which — compiled from individual YAMLs, no LLM needed
# ---------------------------------------------------------------------------

def update_general_who(chars_dir: Path, lore_dir: Path) -> None:
    all_chars = {}
    if chars_dir.exists():
        for yf in sorted(chars_dir.glob("*.yaml")):
            with open(yf, encoding="utf-8") as f:
                d = yaml.safe_load(f) or {}
            if d.get("name"):
                all_chars[d["name"]] = d
    lore_dir.mkdir(parents=True, exist_ok=True)
    with open(lore_dir / "general_who.yaml", "w", encoding="utf-8") as f:
        yaml.dump({"characters": all_chars}, f, allow_unicode=True, sort_keys=False)
    print(f"    general_who: {len(all_chars)} characters")


def update_general_where(places_dir: Path, lore_dir: Path) -> None:
    all_places = {}
    if places_dir.exists():
        for yf in sorted(places_dir.glob("*.yaml")):
            with open(yf, encoding="utf-8") as f:
                d = yaml.safe_load(f) or {}
            if d.get("name"):
                all_places[d["name"]] = d
    lore_dir.mkdir(parents=True, exist_ok=True)
    with open(lore_dir / "general_where.yaml", "w", encoding="utf-8") as f:
        yaml.dump({"places": all_places}, f, allow_unicode=True, sort_keys=False)
    print(f"    general_where: {len(all_places)} places")


def update_general_which(concepts_dir: Path, lore_dir: Path) -> None:
    all_concepts = {}
    if concepts_dir.exists():
        for yf in sorted(concepts_dir.glob("*.yaml")):
            with open(yf, encoding="utf-8") as f:
                d = yaml.safe_load(f) or {}
            if d.get("name"):
                all_concepts[d["name"]] = d
    lore_dir.mkdir(parents=True, exist_ok=True)
    with open(lore_dir / "general_which.yaml", "w", encoding="utf-8") as f:
        yaml.dump({"concepts": all_concepts}, f, allow_unicode=True, sort_keys=False)
    print(f"    general_which: {len(all_concepts)} concepts")


# ---------------------------------------------------------------------------
# What — recurring themes/events across scenes (LLM, incremental)
# ---------------------------------------------------------------------------

_WHAT_PROMPT = """\
You are synthesizing narrative content of a roleplay story across multiple scenes.

Scene summaries and key events:
{scenes_data}

Identify STRONG RECURRENCES only: themes, actions, objects, or character interactions that appear in at least 2 scenes.
Omit anything that appears only once.

JSON:
{{
  "recurring_themes": [],
  "recurring_actions": [],
  "recurring_objects": [],
  "recurring_character_interactions": [],
  "overall_summary": ""
}}"""


def update_general_what(lore_dir: Path, scene_id: str, what_data: dict) -> None:
    out = lore_dir / "general_what.yaml"

    existing = {}
    if out.exists():
        with open(out, encoding="utf-8") as f:
            existing = yaml.safe_load(f) or {}

    scenes = existing.get("scenes", {})

    if scene_id in scenes:
        print(f"    [skip] general_what: {scene_id} already processed")
        return

    scenes[scene_id] = {
        "summary": what_data.get("summary", ""),
        "events": [e.get("description", "") for e in (what_data.get("events") or [])[:10]],
    }
    existing["scenes"] = scenes

    recurrences = existing.get("recurrences", {})
    if len(scenes) >= 2:
        scenes_text = "\n\n".join(
            f"Scene {sid}:\nSummary: {sd['summary']}\nEvents: {'; '.join(sd['events'][:5])}"
            for sid, sd in list(scenes.items())[-10:]
        )
        result = call_llm_json(_WHAT_PROMPT.format(scenes_data=scenes_text), num_predict=1024)
        recurrences = {
            "themes":                 result.get("recurring_themes") or [],
            "actions":                result.get("recurring_actions") or [],
            "objects":                result.get("recurring_objects") or [],
            "character_interactions": result.get("recurring_character_interactions") or [],
            "overall_summary":        result.get("overall_summary") or "",
        }

    existing["recurrences"] = recurrences
    lore_dir.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        yaml.dump(existing, f, allow_unicode=True, sort_keys=False)
    print(f"    general_what: {len(scenes)} scenes, {len(recurrences.get('themes', []))} recurring themes")


# ---------------------------------------------------------------------------
# How — narrative axes, strongest links (LLM, incremental)
# ---------------------------------------------------------------------------

_HOW_GENERAL_PROMPT = """\
You are building a master narrative analysis of a roleplay story based on all scenes analyzed so far.

HOW analyses from recent scenes (causal links, character relations, synthesis):
{how_scenes}

Current narrative axes (update/extend these — do not reset them):
{current_axes}

Based on ALL of this:
1. NARRATIVE AXES — major thematic threads running through the story.
   Each axis must have strong evidence (repeated or strongly insisted upon across scenes).
   Keep strong axes from the current list; add new ones only if well supported.
2. STRONGEST LINKS — most repeated or most significant causal links across scenes.
3. OVERALL NARRATIVE SUMMARY — connecting the axes in one paragraph.

JSON:
{{
  "narrative_axes": [
    {{
      "name": "",
      "elements": [],
      "summary": "",
      "scenes": [],
      "strength": "strong|moderate"
    }}
  ],
  "strongest_links": [
    {{
      "from_element": "",
      "to_element": "",
      "link_type": "",
      "description": "",
      "occurrence_count": 1
    }}
  ],
  "overall_narrative_summary": ""
}}"""


def update_general_how(lore_dir: Path, scene_id: str, how_data: dict) -> None:
    out = lore_dir / "general_how.yaml"

    existing = {}
    if out.exists():
        with open(out, encoding="utf-8") as f:
            existing = yaml.safe_load(f) or {}

    scenes_how = existing.get("scenes_how", {})

    if scene_id in scenes_how:
        print(f"    [skip] general_how: {scene_id} already processed")
        return
    scenes_how[scene_id] = {
        "links": [
            {
                "from": l.get("from_element", ""),
                "to":   l.get("to_element", ""),
                "type": l.get("link_type", ""),
                "desc": (l.get("description") or "")[:100],
            }
            for l in (how_data.get("links") or [])[:8]
        ],
        "character_relations": [
            {
                "from":      r.get("from_char", ""),
                "to":        r.get("to_char", ""),
                "rel":       r.get("relation_type", ""),
                "sentiment": r.get("sentiment", ""),
                "desc":      (r.get("description") or "")[:80],
            }
            for r in (how_data.get("character_relations") or [])[:6]
        ],
        "synthesis": how_data.get("context_synthesis", ""),
    }
    existing["scenes_how"] = scenes_how

    current_axes = existing.get("narrative_axes") or []

    def _fmt_scene(sid, sd):
        links_str = "; ".join(
            "{} -{}-> {}".format(l["from"], l["type"], l["to"])
            for l in sd["links"][:5]
        )
        rels_str = "; ".join(
            "{} {} {} ({})".format(r["from"], r["rel"], r["to"], r["sentiment"])
            for r in sd["character_relations"][:4]
        )
        return f"Scene {sid}:\n  Synthesis: {sd['synthesis']}\n  Links: {links_str}\n  Relations: {rels_str}"

    how_text = "\n\n".join(
        _fmt_scene(sid, sd)
        for sid, sd in list(scenes_how.items())[-8:]
    )

    axes_text = yaml.dump(current_axes, allow_unicode=True) if current_axes else "none yet"

    result = call_llm_json(
        _HOW_GENERAL_PROMPT.format(how_scenes=how_text, current_axes=axes_text),
        num_predict=2048,
        num_ctx=8192,
    )

    existing["narrative_axes"]           = result.get("narrative_axes") or []
    existing["strongest_links"]          = result.get("strongest_links") or []
    existing["overall_narrative_summary"] = result.get("overall_narrative_summary") or ""

    lore_dir.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        yaml.dump(existing, f, allow_unicode=True, sort_keys=False)
    print(f"    general_how: {len(existing['narrative_axes'])} narrative axes")
