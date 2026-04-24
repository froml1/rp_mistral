"""
Purge — two-pass scene extraction.

Pass 1 — message filtering (no LLM):
  Drop: empty content, pure separators (---, ===), custom-emoji-only, URL-only.
  Tag each kept message with _scene based on time gaps:
    gap > 30 min  → new scene
    gap > 60 min  → new session (same effect: new scene)

Pass 2 — scene size filter:
  Drop entire scene groups with fewer than MIN_SCENE_MESSAGES messages.

Output: one JSON file per surviving scene, written to out_dir/.
  Format: {"scene_id": "stem_000", "source": "stem.json", "messages": [...]}
"""

import csv
import json
import re
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

_SCENE_BREAK  = re.compile(r'^[\-_*=~]{3,}\s*$')
_EMOJI_CUSTOM = re.compile(r'<:\w+:\d+>')
_MENTION      = re.compile(r'<@!?\d+>')
_URL          = re.compile(r'https?://\S+')
_NARRATIVE    = re.compile(r'[a-zA-ZÀ-ÿ0-9]{2,}')

_SCENE_BREAK_SECS  = 1800  # 30 min → new scene
_SESSION_END_SECS  = 3600  # 60 min → new session (= new scene)
MIN_SCENE_MESSAGES = 20


def _parse_ts(ts: str) -> datetime | None:
    if not ts:
        return None
    try:
        return datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except ValueError:
        return None


def _clean(text: str) -> str:
    text = _MENTION.sub("", text)
    text = _EMOJI_CUSTOM.sub("", text)
    text = _URL.sub("", text)
    return re.sub(r'\s+', ' ', text).strip()


def _should_drop(content: str) -> bool:
    s = content.strip()
    if not s:
        return True
    if _SCENE_BREAK.match(s):
        return True
    if not _NARRATIVE.search(_clean(s)):
        return True
    return False


def load_messages(filepath: Path) -> list[dict]:
    if filepath.suffix.lower() == ".csv":
        messages = []
        with open(filepath, encoding="utf-8-sig", newline="") as f:
            sample = f.read(4096)
            f.seek(0)
            try:
                delimiter = csv.Sniffer().sniff(sample, delimiters=",;\t|").delimiter
            except csv.Error:
                delimiter = ","
            for row in csv.DictReader(f, delimiter=delimiter):
                author  = (row.get("author") or row.get("Author") or "").strip()
                content = (row.get("content") or row.get("Content") or "").strip()
                ts      = (row.get("timestamp") or row.get("Timestamp") or "").strip()
                if content:
                    messages.append({"author": {"name": author}, "content": content, "timestamp": ts})
        return messages
    with open(filepath, encoding="utf-8") as f:
        data = json.load(f)
    return data.get("messages", data) if isinstance(data, dict) else data


def purge_export(filepath: Path, out_dir: Path, verbose: bool = True,
                 min_scene_messages: int = MIN_SCENE_MESSAGES) -> list[Path]:
    """
    Parse filepath, split by time gaps, drop small scenes.
    Returns list of written scene file paths.
    """
    raw_messages = load_messages(filepath)
    if not raw_messages:
        return []

    raw_messages.sort(key=lambda m: _parse_ts(m.get("timestamp", "")) or datetime.min)
    total = len(raw_messages)
    stem  = filepath.stem

    # ── Pass 1: filter + tag by time gap ─────────────────────────────────────
    scene_idx = 0
    last_ts: datetime | None = None
    buckets: dict[int, list[dict]] = defaultdict(list)

    for msg in raw_messages:
        content = msg.get("content", "").strip()
        ts = _parse_ts(msg.get("timestamp", ""))

        if last_ts and ts:
            gap = (ts - last_ts).total_seconds()
            if gap > _SCENE_BREAK_SECS:          # covers both 30-min and 60-min cases
                scene_idx += 1
                if verbose:
                    tag = "session" if gap > _SESSION_END_SECS else "scene"
                    print(f"  [new {tag}] gap {int(gap//60)}min → #{scene_idx} | {ts.strftime('%Y-%m-%d %H:%M')}")
        if ts:
            last_ts = ts

        if _should_drop(content):
            continue

        buckets[scene_idx].append(msg)

    # ── Pass 2: drop small scenes, write surviving ones ───────────────────────
    out_dir.mkdir(parents=True, exist_ok=True)

    small   = {sid for sid, msgs in buckets.items() if len(msgs) < min_scene_messages}
    kept_n  = len(buckets) - len(small)
    dropped_msgs = sum(len(buckets[s]) for s in small)

    if verbose and small:
        print(f"  [pass 2] {len(small)} scene(s) < {min_scene_messages} msgs dropped "
              f"({dropped_msgs} messages removed)")

    written: list[Path] = []
    file_idx = 0
    for sid in sorted(buckets):
        if sid in small:
            continue
        msgs = buckets[sid]
        scene_id   = f"{stem}_{file_idx:03d}"
        scene_path = out_dir / f"{scene_id}.json"
        scene_path.write_text(
            json.dumps({"scene_id": scene_id, "source": filepath.name, "messages": msgs},
                       ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        written.append(scene_path)
        file_idx += 1

    if verbose:
        kept_msgs = sum(len(buckets[s]) for s in buckets if s not in small)
        print(f"  {filepath.name}: {total} raw → {kept_msgs} msgs in {len(written)} scene(s)")

    return written


if __name__ == "__main__":
    exports_dir = next((a for a in sys.argv[1:] if not a.startswith("--")), "data/exports")
    input_path  = Path(exports_dir)
    out_root    = Path(__file__).resolve().parent.parent / "data" / "purged"

    files = [input_path] if input_path.is_file() else (
        list(input_path.glob("**/*.json")) + list(input_path.glob("**/*.csv"))
    )
    for fp in files:
        purge_export(fp, out_root / fp.stem, verbose=True)
