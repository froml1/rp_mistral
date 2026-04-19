"""Step 3 - Subdivide: group messages into scenes, verify coherence, split if needed."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from llm import call_llm_json

_COHERENCE_PROMPT = """\
You are analyzing a sequence of RP messages to determine scene coherence.

Does this sequence form ONE coherent scene (same location, same main characters, same narrative arc)?
Or does it contain MULTIPLE distinct scenes chained together (location change, character group split, clear narrative break)?

Note: simultaneous parallel contexts (group A in location X while group B in location Y) count as MULTIPLE scenes.

If multiple scenes, return the index of the FIRST message of the second scene.

JSON: {{"coherent": true, "split_at": null}}
  or: {{"coherent": false, "split_at": 12, "reason": "location change / character split / theme break"}}

Messages (showing first 100 chars each):
{messages}"""


def _scene_text(messages: list[dict]) -> str:
    lines = []
    for i, msg in enumerate(messages):
        author = msg.get("author", {})
        name = author.get("name", "?") if isinstance(author, dict) else str(author)
        content = (msg.get("content_en") or msg.get("content", ""))[:100]
        lines.append(f"[{i}] {name}: {content}")
    return "\n".join(lines)


def _subdivide(messages: list[dict], depth: int = 0) -> list[list[dict]]:
    if len(messages) < 6 or depth > 3:
        return [messages]

    data = call_llm_json(
        _COHERENCE_PROMPT.format(messages=_scene_text(messages)),
        num_predict=80,
    )

    if data.get("coherent", True):
        return [messages]

    split_at = data.get("split_at")
    if not isinstance(split_at, int) or not (1 <= split_at < len(messages) - 1):
        return [messages]

    print(f"  {'  ' * depth}split at msg {split_at} - {data.get('reason', '')}")
    return _subdivide(messages[:split_at], depth + 1) + _subdivide(messages[split_at:], depth + 1)


def _group_by_scene_tag(messages: list[dict]) -> list[list[dict]]:
    scenes: dict[int, list[dict]] = {}
    for msg in messages:
        sid = msg.get("_scene", 0)
        sid = sid if isinstance(sid, int) else 0
        scenes.setdefault(sid, []).append(msg)
    return [scenes[k] for k in sorted(scenes)] if scenes else [messages]


def run_subdivide(translated_dir: Path, out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    files = list(translated_dir.glob("**/*.json"))
    if not files:
        print(f"  No translated files found in {translated_dir}")
        return []

    produced = []
    for fp in files:
        file_out_dir = out_dir / fp.stem
        if file_out_dir.exists() and any(file_out_dir.glob("*.json")):
            existing = list(file_out_dir.glob("*.json"))
            print(f"  [skip] {fp.name} -> {len(existing)} scenes already split")
            produced.extend(existing)
            continue

        file_out_dir.mkdir(parents=True, exist_ok=True)
        print(f"  Subdividing {fp.name}...")

        with open(fp, encoding="utf-8") as f:
            data = json.load(f)
        messages = data.get("messages", data) if isinstance(data, dict) else data

        pre_scenes = _group_by_scene_tag(messages)
        print(f"  {len(pre_scenes)} scenes from time gaps")

        scene_idx = 0
        for pre in pre_scenes:
            sub_scenes = _subdivide(pre)
            for sub in sub_scenes:
                scene_id = f"{fp.stem}_{scene_idx:03d}"
                scene_path = file_out_dir / f"{scene_id}.json"
                scene_path.write_text(
                    json.dumps({"scene_id": scene_id, "source": fp.name, "messages": sub},
                               ensure_ascii=False, indent=2),
                    encoding="utf-8"
                )
                produced.append(scene_path)
                scene_idx += 1

        print(f"  -> {scene_idx} final scenes")

    return produced
