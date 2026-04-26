"""Step 4d — What: exhaustive scene events, fed by When + Where + Who results."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from llm import call_llm_json
from steps.manual_lore import load_manual_events, merge_manual_into_what
from steps.synthesis import synthesis_context_block
from steps.scene_patch import write_enrichment, chunk_messages


def _is_valid_json(path: Path) -> bool:
    try:
        json.loads(path.read_text(encoding="utf-8"))
        return True
    except Exception:
        return False

_PROMPT = """\
Based on the analysis context below, produce an EXHAUSTIVE list of everything that happens in this scene.


PRIOR SCENE CONTEXT (what happened before — use to avoid misidentifying characters or events):
{narrative_context}

TEMPORAL CONTEXT:
{when}

LOCATIONS:
{where}

CHARACTERS:
{who}

CONCEPTS & THEMES:
{which}

EVENTS ALREADY IDENTIFIED IN EARLIER PARTS OF THIS SAME SCENE (do not duplicate, use to maintain continuity):
{prior_chunk_context}

List ALL events without omission:
- conversations: topics discussed, questions asked, information shared, agreements/disagreements
- actions: physical actions, movements, gestures
- revelations: new information disclosed
- decisions: choices made by characters
- emotional moments: significant emotional reactions

Reference the relevant concepts/themes when they appear in events.
For each event include which characters are involved.

INCONSISTENCIES — flag any clear problem: an event involving a character not present in the scene,
a physical impossibility, an event that contradicts prior scene context, etc.
inconsistencies: [{{"message_idx": int_or_null, "type": "absent_character|impossible_action|event_conflict|other", "description": "..."}}]

JSON: {{
  "summary": "dense narrative summary of the full scene",
  "events": [
    {{"type": "conversation|action|revelation|decision|emotional", "description": "", "characters": []}}
  ],
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


def run_what(scene_file: Path, analysis_dir: Path, when: dict, where: dict, who: dict, which: dict, lore_dir: Path | None = None) -> dict:
    out_path = analysis_dir / "what.json"
    if out_path.exists() and _is_valid_json(out_path):
        print(f"    [skip] what already done")
        return json.loads(out_path.read_text(encoding="utf-8"))
    if out_path.exists():
        print(f"    [corrupt] what.json malformed, re-analyzing...")

    if not _is_valid_json(scene_file):
        print(f"    [error] scene file {scene_file.name} is malformed, delete it and re-run step 3")
        return {}
    with open(scene_file, encoding="utf-8") as f:
        scene = json.load(f)

    scene_id  = scene["scene_id"]
    messages  = scene["messages"]

    when_ctx      = f"Time of day: {when.get('time_of_day')}. Duration: {when.get('duration')}. {when.get('summary', '')}"
    where_ctx     = f"Locations: {', '.join(where.get('locations') or [])}. Changes: {where.get('location_changes')}"
    who_ctx       = f"Characters present: {', '.join(who.get('characters') or [])}"
    which_ctx     = ", ".join(which.get("concepts") or []) or "none identified"
    narrative_ctx = synthesis_context_block(lore_dir, current_scene_id=scene_id, window=15, characters=who.get("characters")) if lore_dir else "none"

    chunks = chunk_messages(messages)
    if len(chunks) > 1:
        print(f"    what: {len(chunks)} chunks")
    raw_results = []
    prior_chunk_context = "none"
    for chunk in chunks:
        r = call_llm_json(
            _PROMPT.format(
                narrative_context=narrative_ctx,
                when=when_ctx, where=where_ctx, who=who_ctx, which=which_ctx,
                prior_chunk_context=prior_chunk_context,
                text=_scene_text(chunk),
            ),
            num_predict=3072,
            num_ctx=8192,
        )
        raw_results.append(r)
        prior_events = "; ".join(
            e.get("description", "")[:60] for e in (r.get("events") or [])[:5] if e.get("description")
        )
        prior_chunk_context = prior_events or "none"

    all_events = [
        e for r in raw_results for e in (r.get("events") or [])
        if isinstance(e, dict) and e.get("description")
    ]
    summaries = [str(r.get("summary") or "") for r in raw_results if r.get("summary")]
    output = {
        "summary": " ".join(summaries),
        "events":  all_events,
    }
    output = merge_manual_into_what(output, load_manual_events(scene_id))

    incs = [i for i in (result.get("inconsistencies") or []) if isinstance(i, dict) and i.get("description")]
    if incs:
        write_enrichment(scene_file, "what", {"inconsistencies": incs})
        for inc in incs:
            print(f"    [what] inconsistency: {inc.get('type')} — {inc.get('description')[:80]}")

    analysis_dir.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"    what: {len(output['events'])} events")
    return output
