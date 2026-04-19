"""Step 4e — How: links between elements, enriched by previous How YAML context."""

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

_PROMPT = """\
Based on the full analysis of this scene, identify HOW events unfold: the causal links, means, and relationships between elements.

TEMPORAL: {when}
LOCATIONS: {where}
CHARACTERS: {who}
CONCEPTS: {which}
EVENTS: {what}

Accumulated context from previous scenes (use to enrich links):
{how_context}

For each significant link or causal relationship:
- from_element: what/who initiates or causes
- to_element: what/who is affected or results
- link_type: causes / enables / prevents / leads_to / opposes / supports
- description: how and why this link exists, what it reveals
- characters_involved: list of characters

Also produce a context_synthesis: how this scene connects to the broader narrative based on accumulated context.

JSON: {{
  "links": [{{"from_element": "", "to_element": "", "link_type": "", "description": "", "characters_involved": []}}],
  "context_synthesis": ""
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

    # Summarize context (last 5 scenes to keep prompt size reasonable)
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
            where=", ".join(where.get("locations") or []),
            who=", ".join(who.get("characters") or []),
            which=", ".join(which.get("concepts") or []) or "none",
            what=events_summary,
            how_context=recent_ctx,
            text=text,
        ),
        num_predict=1024,
        num_ctx=8192,
    )

    output = {
        "links": [
            l for l in (result.get("links") or [])
            if isinstance(l, dict) and l.get("description")
        ],
        "context_synthesis": str(result.get("context_synthesis") or ""),
    }

    analysis_dir.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")

    if output["context_synthesis"]:
        _save_how_context(scene_id, output["context_synthesis"])

    print(f"    how: {len(output['links'])} links")
    return output
