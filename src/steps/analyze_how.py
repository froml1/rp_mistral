"""Step 4e — How: causal links, character relations/sentiments, enriched by general narrative context."""

import json
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from llm import call_llm_json
from steps.synthesis import synthesis_context_block
from steps.scene_patch import read_enrichments, write_enrichment, format_inconsistencies, chunk_messages


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
TEMPORAL: {when}

LOCATIONS (with details):
{where_details}

CHARACTERS (with details):
{who_details}

SPEAKER ATTRIBUTION (Discord author → character they play in this scene):
{speaker_attribution}

CONCEPTS: {which}

EVENTS: {what}

Accumulated context from previous scenes:
{how_context}

Ongoing narrative axes (from general analysis):
{narrative_axes}

Known inconsistencies flagged by prior analysis steps:
{prior_inconsistencies}

Produce FOUR things:

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

4. INCONSISTENCIES — any causal contradiction, impossible relation, or logical impossibility you detect
   in the scene (e.g. character A claims to be allied with B but actions show the opposite).
   inconsistencies: [{{"message_idx": int_or_null, "type": "contradictory_relation|impossible_causality|character_conflict|other", "description": "..."}}]

JSON: {{
  "links": [{{"from_element": "", "to_element": "", "link_type": "", "description": "", "characters_involved": []}}],
  "character_relations": [{{"from_char": "", "to_char": "", "relation_type": "", "sentiment": "", "description": ""}}],
  "context_synthesis": "",
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
        rels = ", ".join(
            f"{r.get('character', '')} ({r.get('relation', '')})" if isinstance(r, dict) else str(r)
            for r in (c.get("relations") or [])[:3]
        )
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


def run_how(scene_file: Path, analysis_dir: Path, when: dict, where: dict, who: dict, which: dict, what: dict, lore_dir: Path | None = None) -> dict:
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
    messages = scene["messages"]

    # Read enrichments written by prior steps (entities, context, what)
    enrichments = read_enrichments(scene_file)
    a2c = enrichments.get("entities", {}).get("author_to_character") or {}
    speaker_attribution = (
        "\n".join(f"- {author} → {char}" for author, char in a2c.items())
        or "not available"
    )
    prior_inconsistencies = format_inconsistencies(
        {k: v for k, v in enrichments.items() if k != "how"}
    )

    if lore_dir is not None:
        recent_ctx = synthesis_context_block(lore_dir, current_scene_id=scene_id)
    else:
        how_ctx = _load_how_context()
        recent_ctx = "\n".join(
            f"- {sid}: {synth}" for sid, synth in list(how_ctx.items())[-5:]
        ) or "none yet"

    events_summary = "; ".join(
        e.get("description", "") for e in (what.get("events") or [])[:10]
    )

    def _call_chunk(chunk):
        return call_llm_json(
            _PROMPT.format(
                when=f"{when.get('time_of_day')} / {when.get('duration')}",
                where_details=_format_where_details(where),
                who_details=_format_who_details(who),
                speaker_attribution=speaker_attribution,
                which=", ".join(which.get("concepts") or []) or "none",
                what=events_summary,
                how_context=recent_ctx,
                narrative_axes=_load_narrative_axes(),
                prior_inconsistencies=prior_inconsistencies,
                text=_scene_text(chunk),
            ),
            num_predict=3072,
            num_ctx=8192,
        )

    chunks = chunk_messages(messages)
    if len(chunks) > 1:
        print(f"    how: {len(chunks)} chunks")
    raw_results = [_call_chunk(chunk) for chunk in chunks]

    # Merge links (deduplicate by from+to)
    seen_links: set[tuple] = set()
    merged_links = []
    for r in raw_results:
        for l in (r.get("links") or []):
            if not isinstance(l, dict) or not l.get("description"):
                continue
            key = (l.get("from_element", "").lower(), l.get("to_element", "").lower())
            if key not in seen_links:
                merged_links.append(l); seen_links.add(key)

    # Merge character_relations (deduplicate by from+to+type)
    seen_rels: set[tuple] = set()
    merged_rels = []
    for r in raw_results:
        for rel in (r.get("character_relations") or []):
            if not isinstance(rel, dict) or not rel.get("from_char") or not rel.get("to_char"):
                continue
            key = (rel.get("from_char", "").lower(), rel.get("to_char", "").lower(), rel.get("relation_type", "").lower())
            if key not in seen_rels:
                merged_rels.append(rel); seen_rels.add(key)

    synthesis_parts = [str(r.get("context_synthesis") or "") for r in raw_results if r.get("context_synthesis")]

    output = {
        "links":               merged_links,
        "character_relations": merged_rels,
        "context_synthesis":   " ".join(synthesis_parts),
    }

    incs = [i for i in (result.get("inconsistencies") or []) if isinstance(i, dict) and i.get("description")]
    if incs:
        write_enrichment(scene_file, "how", {"inconsistencies": incs})
        for inc in incs:
            print(f"    [how] inconsistency: {inc.get('type')} — {inc.get('description')[:80]}")

    analysis_dir.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")

    if output["context_synthesis"]:
        _save_how_context(scene_id, output["context_synthesis"])

    print(f"    how: {len(output['links'])} links, {len(output['character_relations'])} character relations")
    return output
