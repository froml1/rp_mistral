"""Step 3 - Subdivide: LLM coherence split on per-scene files from translate."""

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

_MANIFEST = "_manifest.json"


def _is_valid_json(path: Path) -> bool:
    try:
        json.loads(path.read_text(encoding="utf-8"))
        return True
    except Exception:
        return False


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
    reasons        = data.get("reasons") or []
    boundaries = sorted(set(
        b for b in raw_boundaries
        if isinstance(b, int) and 1 <= b < len(messages)
    ))

    if not boundaries:
        return [messages]

    for i, b in enumerate(boundaries):
        reason = reasons[i] if i < len(reasons) else ""
        print(f"    split [{b}]" + (f" — {reason}" if reason else ""))

    segments, prev = [], 0
    for b in boundaries:
        segments.append(messages[prev:b])
        prev = b
    segments.append(messages[prev:])
    return [s for s in segments if s]


def _load_manifest(out_dir: Path) -> dict:
    mf = out_dir / _MANIFEST
    if mf.exists():
        try:
            return json.loads(mf.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}


def _save_manifest(out_dir: Path, manifest: dict):
    (out_dir / _MANIFEST).write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def run_subdivide(translated_dir: Path, out_dir: Path, purged_dir: Path | None = None) -> list[Path]:
    """
    For each scene file in translated_dir/**/*.json, run LLM coherence split.
    Input scene files are already time-gap separated by purge.
    May produce multiple output files per input if a split is detected.
    Output: out_dir/{stem}/{scene_id}.json
    """
    scene_files = sorted(translated_dir.glob("**/*.json"))
    if not scene_files:
        print(f"  No translated scene files found in {translated_dir}")
        return []

    produced = []
    for fp in scene_files:
        # Output subdir mirrors the translated subdir name
        stem     = fp.parent.name          # e.g. "channel_rp"
        out_subdir = out_dir / stem
        out_subdir.mkdir(parents=True, exist_ok=True)

        manifest = _load_manifest(out_subdir)

        # Skip if this input scene was already processed
        if fp.stem in manifest:
            for sid in manifest[fp.stem]:
                sp = out_subdir / f"{sid}.json"
                if sp.exists():
                    produced.append(sp)
            continue

        if not _is_valid_json(fp):
            print(f"  [error] {fp.name} is malformed, skipping")
            continue

        with open(fp, encoding="utf-8") as f:
            scene = json.load(f)

        messages = scene.get("messages", [])
        source   = scene.get("source", fp.name)
        print(f"  {fp.name} ({len(messages)} msgs)", end="")

        sub_scenes = _subdivide(messages)

        if len(sub_scenes) == 1:
            print(" → 1 scene")
        else:
            print(f" → {len(sub_scenes)} sub-scenes")

        # Count existing scene files to assign sequential IDs without collision
        existing_ids = [
            int(p.stem.rsplit("_", 1)[-1])
            for p in out_subdir.glob("*.json")
            if p.name != _MANIFEST and p.stem.rsplit("_", 1)[-1].isdigit()
        ]
        next_idx = max(existing_ids, default=-1) + 1

        written_ids = []
        for sub in sub_scenes:
            scene_id   = f"{stem}_{next_idx:03d}"
            scene_path = out_subdir / f"{scene_id}.json"
            scene_path.write_text(
                json.dumps({"scene_id": scene_id, "source": source, "messages": sub},
                           ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            produced.append(scene_path)
            written_ids.append(scene_id)
            next_idx += 1

        manifest[fp.stem] = written_ids
        _save_manifest(out_subdir, manifest)

    return produced
