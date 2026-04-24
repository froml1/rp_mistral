"""Step 4e — How: causal links, character relations/sentiments, enriched by general narrative context."""

import json
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from llm import call_llm_json


def _is_valid_json(path: Path) -> bool:
    try:
        json.loads(path.read_text(encoding="utf-8"))
        return True
    except Exception:
        return False

_HOW_YAML = Path(__file__).parent.parent.parent / "data" / "lore" / "how_context.yaml"
_GENERAL_HOW_YAML = Path(__file__).parent.parent.parent / "data" / "lore" / "general_how.yaml"

_PROMPT = """\
Based on the full analysis of this scene, identify HOW events unfold: causal links, character relationships/sentiments, and connections between all elements.
IMPORTANT: if context seams too informal ignore analyse, return struct with empty fields (maybe a casual discussion)
TEMPORAL: {when}

LOCATIONS (with details):
{where_details}

CHARACTERS (with details):
{who_details}

CONCEPTS: {which}

EVENTS: {what}

Accumulated context from previous scenes:
{how_context}

Ongoing narrative axes (from general analysis):
{narrative_axes}

Produce THREE things:

1. ELEMENT LINKS — causal and enabling links between any elements (who, where, which, what):
   - from_element / to_element: characters, places, concepts, or events
   - link_type: causes / enables / prevents / leads_to / opposes / supports / inhabits / owns / references
   - description: how and why this link exists
   - characters_involved: list of characters

2. CHARACTER RELATIONS — explicit relationships and sentiments between characters in this scene:
   - from_char / to_char: character names
   - relation_type: ally / enemy / rival / friend / mentor / subordinate / stranger / family / romantic / neutral
   - sentiment: the emotional tone (trust / distrust / fear / admiration / resentment / love / indifference / etc.)
   - description: what in the scene reveals this relation/sentiment

3. CONTEXT SYNTHESIS — how this scene connects to the broader narrative based on accumulated context.

JSON: {{
  "links": [{{"from_element": "", "to_element": "", "link_type": "", "description": "", "characters_involved": []}}],
  "character_relations": [{{"from_char": "", "to_char": "", "relation_type": "", "sentiment": "", "description": ""}}],
  "context_synthesis": ""
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


def _load_how_context() -> dict:
    if _HOW_YAML.exists():
        with open(_HOW_YAML, encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return {}


def _save_how_context(scene_id: str, synthesis: str):
    context = _load_how_context()
    context[scene_id] = synthesis.lower()
    _HOW_YAML.parent.mkdir(parents=True, exist_ok=True)
    with open(_HOW_YAML, "w", encoding="utf-8") as f:
        yaml.dump(context, f, allow_unicode=True, sort_keys=False)


def _load_narrative_axes() -> str:
    if _GENERAL_HOW_YAML.exists():
        with open(_GENERAL_HOW_YAML, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        axes = data.get("narrative_axes") or []
        if axes:
            return "\n".join(
                f"- {a.get('name', '')}: {a.get('summary', '')} (elements: {', '.join(a.get('elements', [])[:5])})"
                for a in axes[:6]
            )
    return "none yet"


def _format_who_details(who: dict) -> str:
    details = who.get("details") or []
    if not details:
        return ", ".join(who.get("characters") or []) or "none"
    parts = []
    for c in details[:8]:
        name = c.get("canonical_name", "")
        psych = c.get("description_psychological", "")
        locs = ", ".join((c.get("main_locations") or [])[:3])
        rels = ", ".join((c.get("relations") or [])[:3])
        parts.append(f"- {name}: {psych[:80]}" + (f" [locations: {locs}]" if locs else ""))
        parts.append(f"- {name}: {psych[:80]}" + (f" [rels: {rels}]" if rels else ""))
    return "\n".join(parts)


def _format_where_details(where: dict) -> str:
    details = where.get("details") or []
    if not details:
        return ", ".join(where.get("locations") or []) or "none"
    parts = []
    for loc in details[:6]:
        name = loc.get("canonical_name", "")
        desc = loc.get("description", "")
        attrs = ", ".join((loc.get("attributes") or [])[:4])
        parts.append(f"- {name}: {desc[:80]}" + (f" [{attrs}]" if attrs else ""))
    return "\n".join(parts)


def run_how(scene_file: Path, analysis_dir: Path, when: dict, where: dict, who: dict, which: dict, what: dict) -> dict:
    out_path = analysis_dir / "how.json"
    if out_path.exists() and _is_valid_json(out_path):
        print(f"    [skip] how already done")
        return json.loads(out_path.read_text(encoding="utf-8"))
    if out_path.exists():
        print(f"    [corrupt] how.json malformed, re-analyzing...")

    if not _is_valid_json(scene_file):
        print(f"    [error] scene file {scene_file.name} is malformed, delete it and re-run step 3")
        return {}
    with open(scene_file, encoding="utf-8") as f:
        scene = json.load(f)

    scene_id = scene["scene_id"]
    text     = _scene_text(scene["messages"])
    how_ctx  = _load_how_context()

    recent_ctx = "\n".join(
        f"- {sid}: {synth}"
        for sid, synth in list(how_ctx.items())[-5:]
    ) or "none yet"

    events_summary = "; ".join(
        e.get("description", "") for e in (what.get("events") or [])[:10]
    )

    result = call_llm_json(
        _PROMPT.format(
            when=f"{when.get('time_of_day')} / {when.get('duration')}",
            where_details=_format_where_details(where),
            who_details=_format_who_details(who),
            which=", ".join(which.get("concepts") or []) or "none",
            what=events_summary,
            how_context=recent_ctx,
            narrative_axes=_load_narrative_axes(),
            text=text,
        ),
        num_predict=1536,
        num_ctx=8192,
    )

    output = {
        "links": [
            l for l in (result.get("links") or [])
            if isinstance(l, dict) and l.get("description")
        ],
        "character_relations": [
            r for r in (result.get("character_relations") or [])
            if isinstance(r, dict) and r.get("from_char") and r.get("to_char")
        ],
        "context_synthesis": str(result.get("context_synthesis") or ""),
    }

    analysis_dir.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")

    if output["context_synthesis"]:
        _save_how_context(scene_id, output["context_synthesis"])

    print(f"    how: {len(output['links'])} links, {len(output['character_relations'])} character relations")
    return output
