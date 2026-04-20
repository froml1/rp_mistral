"""Step 4d — What: exhaustive scene events, fed by When + Where + Who results."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from llm import call_llm_json
from steps.manual_lore import load_manual_events, merge_manual_into_what


def _is_valid_json(path: Path) -> bool:
    try:
        json.loads(path.read_text(encoding="utf-8"))
        return True
    except Exception:
        return False

_PROMPT = """\
Based on the analysis context below, produce an EXHAUSTIVE list of everything that happens in this scene.

TEMPORAL CONTEXT:
{when}

LOCATIONS:
{where}

CHARACTERS:
{who}

CONCEPTS & THEMES:
{which}

List ALL events without omission:
- conversations: topics discussed, questions asked, information shared, agreements/disagreements
- actions: physical actions, movements, gestures
- revelations: new information disclosed
- decisions: choices made by characters
- emotional moments: significant emotional reactions

Reference the relevant concepts/themes when they appear in events.
For each event include which characters are involved.

JSON: {{
  "summary": "dense narrative summary of the full scene",
  "events": [
    {{"type": "conversation|action|revelation|decision|emotional", "description": "", "characters": []}}
  ]
}}

Scene:
---
{text}
---"""


def _scene_text(messages: list[dict]) -> str:
    return "\n".join(
        f"{(m.get('author') or {}).get('name', '?') if isinstance(m.get('author'), dict) else m.get('author', '?')}: "
        f"{m.get('content_en') or m.get('content', '')}"
        for m in messages
    )


def run_what(scene_file: Path, analysis_dir: Path, when: dict, where: dict, who: dict, which: dict) -> dict:
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

    text = _scene_text(scene["messages"])

    when_ctx  = f"Time of day: {when.get('time_of_day')}. Duration: {when.get('duration')}. {when.get('summary', '')}"
    where_ctx = f"Locations: {', '.join(where.get('locations') or [])}. Changes: {where.get('location_changes')}"
    who_ctx   = f"Characters present: {', '.join(who.get('characters') or [])}"
    which_ctx = ", ".join(which.get("concepts") or []) or "none identified"

    result = call_llm_json(
        _PROMPT.format(when=when_ctx, where=where_ctx, who=who_ctx, which=which_ctx, text=text),
        num_predict=2048,
        num_ctx=8192,
    )

    scene_id = scene["scene_id"]
    output = {
        "summary": str(result.get("summary") or ""),
        "events":  [
            e for e in (result.get("events") or [])
            if isinstance(e, dict) and e.get("description")
        ],
    }
    output = merge_manual_into_what(output, load_manual_events(scene_id))

    analysis_dir.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"    what: {len(output['events'])} events")
    return output
