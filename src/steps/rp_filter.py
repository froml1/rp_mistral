"""
Step 5 — RP Filter: detect non-RP scenes before deep analysis.

One LLM call per scene, uses lore sweep as context anchor.
Writes per-scene rp_check.json and a global data/rp_report.json
listing all non-RP scenes for manual review.
Step 6 (analyze) skips scenes where is_rp is false.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from llm import call_llm_json
from steps.synthesis import synthesis_context_block

_PROMPT = """\
You are a RP quality filter. Decide if this scene is genuine narrative roleplay.

Known story context (characters, arcs, narrative axes — use to judge whether this scene fits the story):
{story_context}

A scene is NOT RP (is_rp: false) if it is mainly:
- Out-of-character chat: players talking as themselves (scheduling, jokes, reactions as players)
- Technical/meta: rules discussion, session setup, dice mechanics without narrative
- Too casual/familiar: no character voices, no *actions*, no story framing at all

A scene IS RP (is_rp: true) if it contains:
- Character voices and in-story dialogue
- Narrative actions in *italics*
- Story-driven interactions between characters

Score 0.0 (not RP) to 1.0 (definitely RP). Threshold for is_rp=true: 0.5.
Flags — list only those that apply: "ooc" | "no_narrative" | "too_casual" | "meta" | "technical"

JSON: {{"is_rp": true, "score": 0.9, "reason": "one sentence", "flags": []}}

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


def _is_valid_json(path: Path) -> bool:
    try:
        json.loads(path.read_text(encoding="utf-8"))
        return True
    except Exception:
        return False


def is_scene_rp(analysis_dir: Path, scene_id: str) -> bool:
    """Return True if the scene passed the RP filter (or was never checked → assume RP)."""
    check_path = analysis_dir / scene_id / "rp_check.json"
    if not check_path.exists():
        return True
    try:
        data = json.loads(check_path.read_text(encoding="utf-8"))
        return bool(data.get("is_rp", True))
    except Exception:
        return True


def run_rp_filter(scenes_dir: Path, lore_dir: Path, analysis_dir: Path) -> dict:
    """
    Filter all scenes. Returns stats dict.
    Writes per-scene rp_check.json + data/rp_report.json for manual review.
    """
    story_context = synthesis_context_block(lore_dir)

    report_path = analysis_dir.parent / "rp_report.json"
    report: dict = {}
    if report_path.exists() and _is_valid_json(report_path):
        report = json.loads(report_path.read_text(encoding="utf-8"))

    scene_files = sorted(scenes_dir.glob("**/*.json"))
    n_rp = n_non_rp = n_skip = 0

    for scene_file in scene_files:
        try:
            scene = json.loads(scene_file.read_text(encoding="utf-8"))
        except Exception:
            continue

        scene_id = scene.get("scene_id", scene_file.stem)
        out_path = analysis_dir / scene_id / "rp_check.json"

        if out_path.exists() and _is_valid_json(out_path):
            n_skip += 1
            continue

        text = _scene_text(scene.get("messages", []))
        if not text.strip():
            continue

        result = call_llm_json(
            _PROMPT.format(
                story_context=story_context,
                text=text,
            ),
            num_predict=256,
            num_ctx=4096,
        )

        is_rp  = bool(result.get("is_rp", True))
        score  = float(result.get("score", 1.0))
        reason = str(result.get("reason", ""))
        flags  = [str(f) for f in (result.get("flags") or [])]

        check = {"scene_id": scene_id, "is_rp": is_rp, "score": round(score, 3), "reason": reason, "flags": flags}

        (analysis_dir / scene_id).mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(check, ensure_ascii=False, indent=2), encoding="utf-8")

        if is_rp:
            n_rp += 1
            print(f"    [rp] {scene_id} — {score:.2f} ✓")
        else:
            n_non_rp += 1
            report[scene_id] = check
            print(f"    [rp] {scene_id} — {score:.2f} ✗  {reason}")

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    total = n_rp + n_non_rp
    print(f"  [rp filter] {total} checked ({n_skip} skipped) — {n_rp} RP / {n_non_rp} non-RP")
    if report:
        print(f"  [rp filter] non-RP report → {report_path}")

    return {"rp": n_rp, "non_rp": n_non_rp, "skipped": n_skip}
