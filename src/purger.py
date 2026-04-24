"""
Light purge — keeps maximum content, only drops structurally empty messages.

Filtering (no LLM, no block logic):
  - Empty content after cleaning
  - Pure separator lines (---, ===, ~~~)
  - Custom Discord emoji only (no readable text)
  - URL-only messages (image links, attachments)

Scene tagging (_scene field) based on time gaps only:
  - gap > 30 min  → new scene within same session
  - gap > 60 min  → new session (reset)
"""

import csv
import json
import re
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

_SCENE_BREAK  = re.compile(r'^[\-_*=~]{3,}\s*$')
_EMOJI_CUSTOM = re.compile(r'<:\w+:\d+>')
_MENTION      = re.compile(r'<@!?\d+>')
_URL          = re.compile(r'https?://\S+')
_NARRATIVE    = re.compile(r'[a-zA-ZÀ-ÿ0-9]{2,}')

_SCENE_BREAK_SECS = 1800   # 30 min → new scene tag
_SESSION_END_SECS = 3600   # 60 min → new session (scene reset)


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
    cleaned = _clean(s)
    if not _NARRATIVE.search(cleaned):
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


def purge_export(filepath: Path, out_path: Path, verbose: bool = True) -> int:
    raw_messages = load_messages(filepath)
    if not raw_messages:
        out_path.write_text('{"messages": []}', encoding="utf-8")
        return 0

    raw_messages.sort(key=lambda m: _parse_ts(m.get("timestamp", "")) or datetime.min)
    total = len(raw_messages)

    scene_id = 0
    last_ts: datetime | None = None
    kept = 0
    first = True

    with open(out_path, "w", encoding="utf-8") as f:
        f.write('{"messages": [')

        for msg in raw_messages:
            content = msg.get("content", "").strip()
            ts = _parse_ts(msg.get("timestamp", ""))

            # Scene tagging from time gaps only
            if last_ts and ts:
                gap = (ts - last_ts).total_seconds()
                if gap > _SESSION_END_SECS:
                    scene_id += 1
                    if verbose:
                        print(f"  [new session] gap {int(gap//60)}min → scene {scene_id} | {ts.strftime('%Y-%m-%d %H:%M')}")
                elif gap > _SCENE_BREAK_SECS:
                    scene_id += 1
                    if verbose:
                        print(f"  [new scene]   gap {int(gap//60)}min → scene {scene_id} | {ts.strftime('%Y-%m-%d %H:%M')}")
            if ts:
                last_ts = ts

            if _should_drop(content):
                continue

            f.write(("" if first else ",") + "\n  ")
            json.dump({**msg, "_scene": scene_id}, f, ensure_ascii=False)
            first = False
            kept += 1

        f.write("\n]}")

    if verbose:
        print(f"  {filepath.name}: {total} → {kept} kept ({total - kept} dropped)")

    return kept


if __name__ == "__main__":
    exports_dir = next((a for a in sys.argv[1:] if not a.startswith("--")), "data/exports")
    input_path  = Path(exports_dir)
    out_dir     = Path(__file__).resolve().parent.parent / "data" / "purged"
    out_dir.mkdir(parents=True, exist_ok=True)

    files = [input_path] if input_path.is_file() else (
        list(input_path.glob("**/*.json")) + list(input_path.glob("**/*.csv"))
    )
    for fp in files:
        purge_export(fp, out_dir / (fp.stem + ".json"), verbose=True)
