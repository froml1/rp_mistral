"""
Step 4 — Clean: trim OOC edges, merge short adjacent scenes.

For each scene file in scenes_dir:
  1. Remove non-RP messages from the start and end
  2. If the scene has fewer than MIN_MERGE_MESSAGES after trimming and there is a
     preceding scene in the same subdir, ask the LLM whether they should merge.
     If yes, append messages to the previous scene file and delete the current one.

Modifies scene files in-place. Uses _clean_manifest.json per subdir for incremental runs.
"""

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from llm import call_llm_json

MIN_MERGE_MESSAGES = 15

_OOC_PATTERNS = re.compile(
    r'\b(lol|lmao|mdr|xd|haha|hihi|ahah|irl|ooc|brb|afk|ok\s|okay|'
    r'ce\s+soir|demain|ce\s+week|disponible|dispo|pas\s+là|absent|'
    r'on\s+joue|on\s+reporte|on\s+reprend|prêt\b|ready\b|'
    r'désolé\b|sorry\b|connexion|internet|reboot|bug\s+technique)\b',
    re.IGNORECASE,
)

_MANIFEST = "_clean_manifest.json"

_MERGE_PROMPT = """\
Two adjacent segments from a Discord RP channel.

SEGMENT A — end of previous scene:
{seg_a}

SEGMENT B — start of next scene (may be short):
{seg_b}

Should B be merged INTO A as a continuous scene?
Merge if: same characters on stage, same location, narrative flows directly from A into B.
Keep separate if: different character group, location change, timeskip, clear narrative break.

JSON: {{"merge": true}} or {{"merge": false}}"""


def _is_rp_msg(msg: dict) -> bool:
    content = (msg.get("content_en") or msg.get("content", "")).strip()
    if not content:
        return False
    if re.search(r'\*[^*]{10,}\*', content):
        return True
    if len(content) >= 80 and not _OOC_PATTERNS.search(content):
        return True
    return False


def _trim_edges(messages: list[dict]) -> tuple[list[dict], int, int]:
    """Remove non-RP messages from start and end. Returns (trimmed, n_prefix, n_suffix)."""
    if not messages:
        return messages, 0, 0

    start = 0
    for i, m in enumerate(messages):
        if _is_rp_msg(m):
            start = i
            break
    else:
        return messages, 0, 0  # no RP found — leave as-is

    end = len(messages)
    for i in range(len(messages) - 1, start - 1, -1):
        if _is_rp_msg(messages[i]):
            end = i + 1
            break

    return messages[start:end], start, len(messages) - end


def _fmt_msgs(messages: list[dict], n: int = 15) -> str:
    lines = []
    for m in messages[:n]:
        author = m.get("author", {})
        name = author.get("name", "?") if isinstance(author, dict) else str(author)
        content = (m.get("content_en") or m.get("content", ""))[:120]
        lines.append(f"{name}: {content}")
    return "\n".join(lines)


def _should_merge(prev_msgs: list[dict], curr_msgs: list[dict]) -> bool:
    seg_a = _fmt_msgs(prev_msgs[-15:])
    seg_b = _fmt_msgs(curr_msgs[:15])
    result = call_llm_json(
        _MERGE_PROMPT.format(seg_a=seg_a, seg_b=seg_b),
        num_predict=60,
    )
    return bool(result.get("merge"))


def _load_manifest(subdir: Path) -> dict:
    mf = subdir / _MANIFEST
    if mf.exists():
        try:
            return json.loads(mf.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {"cleaned": [], "deleted": []}


def _save_manifest(subdir: Path, manifest: dict):
    (subdir / _MANIFEST).write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def run_clean(scenes_dir: Path) -> None:
    """Trim OOC edges and merge short adjacent scenes. Modifies scene files in-place."""
    subdirs = sorted(p for p in scenes_dir.iterdir() if p.is_dir())
    if not subdirs:
        print(f"  No scene subdirs found in {scenes_dir}")
        return

    total_trimmed = total_merged = 0

    for subdir in subdirs:
        scene_files = sorted(
            p for p in subdir.glob("*.json")
            if p.name not in ("_manifest.json", _MANIFEST)
        )
        if not scene_files:
            continue

        manifest   = _load_manifest(subdir)
        cleaned    = set(manifest.get("cleaned", []))
        deleted    = set(manifest.get("deleted", []))

        prev_path: Path | None = None
        prev_msgs: list[dict] = []

        for scene_path in scene_files:
            stem = scene_path.stem

            if stem in deleted:
                continue

            if stem in cleaned:
                try:
                    data = json.loads(scene_path.read_text(encoding="utf-8"))
                    prev_msgs = data.get("messages", [])
                    prev_path = scene_path
                except Exception:
                    pass
                continue

            try:
                data = json.loads(scene_path.read_text(encoding="utf-8"))
            except Exception:
                print(f"    [error] cannot read {scene_path.name}")
                continue

            messages = data.get("messages", [])

            # Step 1 — trim non-RP edges
            trimmed, n_pre, n_suf = _trim_edges(messages)
            if n_pre or n_suf:
                print(f"    [trim] {stem}: -{n_pre} prefix, -{n_suf} suffix  ({len(trimmed)} remain)")
                total_trimmed += 1

            # Step 2 — merge check if short and a preceding scene exists
            if len(trimmed) < MIN_MERGE_MESSAGES and prev_path is not None and prev_msgs:
                merge = _should_merge(prev_msgs, trimmed)
                if merge:
                    print(f"    [merge] {stem} → {prev_path.stem}")
                    prev_data = json.loads(prev_path.read_text(encoding="utf-8"))
                    prev_data["messages"] = prev_data["messages"] + trimmed
                    prev_path.write_text(
                        json.dumps(prev_data, ensure_ascii=False, indent=2),
                        encoding="utf-8",
                    )
                    prev_msgs = prev_data["messages"]
                    scene_path.unlink()
                    deleted.add(stem)
                    manifest["deleted"] = list(deleted)
                    manifest["cleaned"] = list(cleaned)
                    _save_manifest(subdir, manifest)
                    total_merged += 1
                    continue

            # Write trimmed scene back
            data["messages"] = trimmed
            scene_path.write_text(
                json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            cleaned.add(stem)
            manifest["cleaned"] = list(cleaned)
            manifest["deleted"] = list(deleted)
            _save_manifest(subdir, manifest)

            prev_msgs = trimmed
            prev_path = scene_path

    print(f"  [clean] {total_trimmed} trimmed, {total_merged} merged")
