"""Step 3 - Subdivide: group messages into scenes, split agglomerates in one LLM pass."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from llm import call_llm_json

_SPLIT_PROMPT = """\
You are analyzing a sequence of RP messages. Find ALL scene boundaries.

A NEW scene starts when ANY of the following occurs:
- The active character group changes (different characters now on stage)
- The location changes (they move elsewhere, or a new group appears elsewhere)
- A clear narrative break: timeskip, topic shift, end of one arc and start of another
- Parallel scenes: group A in location X while group B is simultaneously in location Y

Return the index of the FIRST message of each new scene.
If this is already ONE continuous scene, return an empty list.

JSON: {{"boundaries": [5, 12, 23], "reasons": ["char group split", "location change", "timeskip"]}}
  or: {{"boundaries": []}} if single scene

Messages (first 150 chars each):
{messages}"""


def _scene_text(messages: list[dict]) -> str:
    lines = []
    for i, msg in enumerate(messages):
        author = msg.get("author", {})
        name = author.get("name", "?") if isinstance(author, dict) else str(author)
        content = (msg.get("content_en") or msg.get("content", ""))[:150]
        lines.append(f"[{i}] {name}: {content}")
    return "\n".join(lines)


def _subdivide(messages: list[dict]) -> list[list[dict]]:
    if len(messages) < 4:
        return [messages]

    data = call_llm_json(
        _SPLIT_PROMPT.format(messages=_scene_text(messages)),
        num_predict=200,
    )

    raw_boundaries = data.get("boundaries") or []
    reasons = data.get("reasons") or []

    boundaries = sorted(set(
        b for b in raw_boundaries
        if isinstance(b, int) and 1 <= b < len(messages)
    ))

    if not boundaries:
        return [messages]

    for i, b in enumerate(boundaries):
        reason = reasons[i] if i < len(reasons) else ""
        print(f"    split at [{b}]" + (f" — {reason}" if reason else ""))

    segments = []
    prev = 0
    for b in boundaries:
        segments.append(messages[prev:b])
        prev = b
    segments.append(messages[prev:])
    return [s for s in segments if s]


def _group_by_scene_tag(messages: list[dict]) -> list[list[dict]]:
    scenes: dict[int, list[dict]] = {}
    for msg in messages:
        sid = msg.get("_scene", 0)
        sid = sid if isinstance(sid, int) else 0
        scenes.setdefault(sid, []).append(msg)
    return [scenes[k] for k in sorted(scenes)] if scenes else [messages]


def _is_valid_json(path: Path) -> bool:
    try:
        json.loads(path.read_text(encoding="utf-8"))
        return True
    except Exception:
        return False


_PROGRESS_FILE = "_progress.json"


def _load_manifest(file_out_dir: Path) -> dict:
    mf = file_out_dir / _PROGRESS_FILE
    if mf.exists():
        try:
            return json.loads(mf.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}


def _save_manifest(file_out_dir: Path, manifest: dict):
    (file_out_dir / _PROGRESS_FILE).write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def run_subdivide(translated_dir: Path, out_dir: Path, purged_dir: Path | None = None) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    files = list(translated_dir.glob("**/*.json"))
    if not files:
        print(f"  No translated files found in {translated_dir}")
        return []

    produced = []
    for fp in files:
        file_out_dir = out_dir / fp.stem
        file_out_dir.mkdir(parents=True, exist_ok=True)

        # Detect corrupt scene files and reset manifest if any found
        scene_files = [f for f in file_out_dir.glob("*.json") if f.name != _PROGRESS_FILE]
        invalid = [f for f in scene_files if not _is_valid_json(f)]
        if invalid:
            print(f"  [corrupt] {len(invalid)} malformed scene(s) in {file_out_dir.name}, resetting progress")
            for f in invalid:
                f.unlink()
            (file_out_dir / _PROGRESS_FILE).unlink(missing_ok=True)

        manifest = _load_manifest(file_out_dir)
        processed_tmp = manifest.get("processed", {})
        manifest_count = sum(processed_tmp.values())
        existing_count = sum(1 for f in file_out_dir.glob("*.json") if f.name != _PROGRESS_FILE)

        # Scene files deleted but manifest still claims them → full reset to avoid wrong scene_idx
        if manifest_count > 0 and existing_count < manifest_count:
            print(f"  [reset] {fp.name}: {existing_count}/{manifest_count} scene files present — resetting")
            for f in file_out_dir.glob("*.json"):
                f.unlink()
            (file_out_dir / _PROGRESS_FILE).unlink(missing_ok=True)
            manifest = {}

        if manifest.get("done"):
            existing = sorted(f for f in file_out_dir.glob("*.json") if f.name != _PROGRESS_FILE)
            print(f"  [skip] {fp.name} -> {len(existing)} scenes already done")
            produced.extend(existing)
            continue

        if not _is_valid_json(fp):
            if purged_dir:
                print(f"  [corrupt] {fp.name} is malformed, re-translating...")
                from steps.translate import run_translate
                fp.unlink()
                run_translate(purged_dir, translated_dir, exports_dir=None)
            else:
                print(f"  [error] {fp.name} is malformed and purged_dir unknown, skipping")
                continue
        with open(fp, encoding="utf-8") as f:
            data = json.load(f)
        messages = data.get("messages", data) if isinstance(data, dict) else data
        pre_scenes = _group_by_scene_tag(messages)
        processed = manifest.get("processed", {})  # {str(pre_idx): sub_scene_count}
        scene_idx = sum(processed.values())

        remaining = len(pre_scenes) - len(processed)
        if remaining < len(pre_scenes):
            print(f"  Resuming {fp.name}: {len(processed)}/{len(pre_scenes)} pre-scenes done")
        else:
            print(f"  Subdividing {fp.name}: {len(pre_scenes)} scenes from time gaps")

        for pre_idx, pre in enumerate(pre_scenes):
            str_idx = str(pre_idx)
            if str_idx in processed:
                continue

            sub_scenes = _subdivide(pre)
            for sub in sub_scenes:
                scene_id = f"{fp.stem}_{scene_idx:03d}"
                scene_path = file_out_dir / f"{scene_id}.json"
                scene_path.write_text(
                    json.dumps({"scene_id": scene_id, "source": fp.name, "messages": sub},
                               ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                scene_idx += 1

            processed[str_idx] = len(sub_scenes)
            _save_manifest(file_out_dir, {"processed": processed})

        manifest["done"] = True
        manifest["processed"] = processed
        _save_manifest(file_out_dir, manifest)

        all_scenes = sorted(f for f in file_out_dir.glob("*.json") if f.name != _PROGRESS_FILE)
        produced.extend(all_scenes)
        print(f"  -> {scene_idx} final scenes")

    return produced
