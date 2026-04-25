"""
Step 4 — Clean: sort scenes by timestamp, trim OOC edges, merge short adjacent scenes,
purge tiny scenes, then renumber to fill gaps.

For each scene subdir:
  0. Sort scene files by their first message timestamp (oldest → newest) and renumber
     them sequentially (stem_000, stem_001, …). Resets the manifest if order changed.
  1. Remove non-RP messages from the start and end of each scene.
  2. If the scene has fewer than MIN_MERGE_MESSAGES after trimming and there is a
     preceding scene in the same subdir, ask the LLM whether they should merge.
     If yes, append messages to the previous scene file and delete the current one.
  3. Purge any remaining scene with fewer than MIN_PURGE_MESSAGES messages.
  4. Renumber surviving files sequentially to fill gaps.

Modifies scene files in-place. Uses _clean_manifest.json per subdir for incremental runs.
"""

import json
import re
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from llm import call_llm_json

MIN_MERGE_MESSAGES = 15
MIN_PURGE_MESSAGES = 10

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


def _parse_ts(ts: str) -> datetime | None:
    if not ts:
        return None
    try:
        return datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except ValueError:
        return None


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


def _scene_files(subdir: Path) -> list[Path]:
    return sorted(
        p for p in subdir.glob("*.json")
        if p.name not in ("_manifest.json", _MANIFEST)
    )


def _first_ts(scene_path: Path) -> datetime:
    try:
        data = json.loads(scene_path.read_text(encoding="utf-8"))
        for m in data.get("messages", []):
            ts = _parse_ts(m.get("timestamp", ""))
            if ts:
                return ts
    except Exception:
        pass
    return datetime.min


def _sort_and_renumber(subdir: Path) -> bool:
    """
    Sort scene files by first message timestamp (oldest first) and renumber them
    stem_000, stem_001, … Rewrites scene_id inside each file.
    Returns True if any file was renamed.
    """
    files = _scene_files(subdir)
    if not files:
        return False

    ordered = sorted(files, key=_first_ts)

    # Derive stem from the first filename: everything before the last _NNN
    stem_match = re.match(r'^(.+?)_\d+$', files[0].stem)
    stem = stem_match.group(1) if stem_match else files[0].stem

    # Build target names
    targets = [subdir / f"{stem}_{i:03d}.json" for i in range(len(ordered))]

    if [p.name for p in ordered] == [p.name for p in targets]:
        return False  # already in order with no gaps

    print(f"    [sort] {subdir.name}: renumbering {len(ordered)} scene(s) by timestamp")

    # Rename via temp names to avoid collisions
    tmp_paths = []
    for i, src in enumerate(ordered):
        tmp = subdir / f"_tmp_sort_{i:03d}.json"
        src.rename(tmp)
        tmp_paths.append(tmp)

    for i, tmp in enumerate(tmp_paths):
        new_stem = f"{stem}_{i:03d}"
        new_path = subdir / f"{new_stem}.json"
        data = json.loads(tmp.read_text(encoding="utf-8"))
        data["scene_id"] = new_stem
        new_path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        tmp.unlink()

    return True


def _purge_tiny(subdir: Path, manifest: dict, deleted: set) -> int:
    """Delete scene files with fewer than MIN_PURGE_MESSAGES messages."""
    n = 0
    for scene_path in _scene_files(subdir):
        if scene_path.stem in deleted:
            continue
        try:
            data = json.loads(scene_path.read_text(encoding="utf-8"))
            if len(data.get("messages", [])) < MIN_PURGE_MESSAGES:
                print(f"    [purge] {scene_path.stem}: {len(data['messages'])} msgs < {MIN_PURGE_MESSAGES}")
                scene_path.unlink()
                deleted.add(scene_path.stem)
                n += 1
        except Exception:
            pass
    return n


def run_clean(scenes_dir: Path) -> None:
    """Sort by timestamp, trim OOC edges, merge short adjacent scenes, purge tiny scenes."""
    subdirs = sorted(p for p in scenes_dir.iterdir() if p.is_dir())
    if not subdirs:
        print(f"  No scene subdirs found in {scenes_dir}")
        return

    total_trimmed = total_merged = total_purged = 0

    for subdir in subdirs:
        # ── Phase 0: sort and renumber by timestamp ───────────────────────────
        renamed = _sort_and_renumber(subdir)
        if renamed:
            mf = subdir / _MANIFEST
            if mf.exists():
                mf.unlink()

        files = _scene_files(subdir)
        if not files:
            continue

        manifest = _load_manifest(subdir)
        cleaned  = set(manifest.get("cleaned", []))
        deleted  = set(manifest.get("deleted", []))

        prev_path: Path | None = None
        prev_msgs: list[dict] = []

        # ── Phase 1+2: trim OOC edges + merge short scenes ───────────────────
        for scene_path in files:
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

            trimmed, n_pre, n_suf = _trim_edges(messages)
            if n_pre or n_suf:
                print(f"    [trim] {stem}: -{n_pre} prefix, -{n_suf} suffix  ({len(trimmed)} remain)")
                total_trimmed += 1

            if len(trimmed) < MIN_MERGE_MESSAGES and prev_path is not None and prev_msgs:
                if _should_merge(prev_msgs, trimmed):
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

        # ── Phase 3: purge scenes < MIN_PURGE_MESSAGES ───────────────────────
        n_purged = _purge_tiny(subdir, manifest, deleted)
        if n_purged:
            manifest["deleted"] = list(deleted)
            _save_manifest(subdir, manifest)
            total_purged += n_purged

        # ── Phase 4: renumber to fill gaps from merges + purges ──────────────
        _sort_and_renumber(subdir)

    print(f"  [clean] {total_trimmed} trimmed, {total_merged} merged, {total_purged} purged")
