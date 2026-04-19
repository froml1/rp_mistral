"""Step 4a — When: temporal context analysis."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from llm import call_llm_json


def _is_valid_json(path: Path) -> bool:
    try:
        json.loads(path.read_text(encoding="utf-8"))
        return True
    except Exception:
        return False

_PROMPT = """\
Analyze the TEMPORAL context of this RP scene.
Base your analysis ONLY on narrative content, NOT on timestamps.

Extract:
- summary: one sentence describing the temporal context
- duration: estimated duration of the scene (e.g. "a few minutes", "an evening", "unknown")
- time_of_day: morning / afternoon / evening / night / unknown
- time_scales: list of time references mentioned in the narrative (years, seasons, past events, etc.)
- time_gaps: any temporal gaps mentioned between fragments of the scene (e.g. "three days later")

JSON: {{"summary": "", "duration": "", "time_of_day": "", "time_scales": [], "time_gaps": []}}

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


def run_when(scene_file: Path, analysis_dir: Path) -> dict:
    out_path = analysis_dir / "when.json"
    if out_path.exists() and _is_valid_json(out_path):
        print(f"    [skip] when already done")
        return json.loads(out_path.read_text(encoding="utf-8"))
    if out_path.exists():
        print(f"    [corrupt] when.json malformed, re-analyzing...")

    if not _is_valid_json(scene_file):
        print(f"    [error] scene file {scene_file.name} is malformed, delete it and re-run step 3")
        return {}
    with open(scene_file, encoding="utf-8") as f:
        scene = json.load(f)

    text = _scene_text(scene["messages"])
    result = call_llm_json(_PROMPT.format(text=text), num_predict=512)
    result = {
        "summary":     str(result.get("summary") or ""),
        "duration":    str(result.get("duration") or "unknown"),
        "time_of_day": str(result.get("time_of_day") or "unknown"),
        "time_scales": [str(t) for t in (result.get("time_scales") or [])],
        "time_gaps":   [str(t) for t in (result.get("time_gaps") or [])],
    }

    analysis_dir.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"    when: {result['time_of_day']} / {result['duration']}")
    return result
