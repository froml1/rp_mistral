"""
RP/HRP separator - pre-processing pass before indexing.

Reads raw Discord exports, writes filtered exports (RP only).

Algorithm:
  - A RP block opens on a message with *asterisks* confirmed by LLM
  - Block ends on gap > 1h
  - Within block: all non-trivial-HRP messages are kept
  - Gap > 30min within block: new scene tag, block stays active
"""

import csv
import json
import re
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from llm import call_llm_json

_SCENE_BREAK  = re.compile(r'^[\-_*=~]{3,}\s*$')
_PARENS       = re.compile(r'[()]')
_NARRATIVE    = re.compile(r'[a-zA-ZÀ-ÿ]{3,}')
_STAR_CONTENT = re.compile(r'\*([^*]+)\*')
_MENTION      = re.compile(r'<@!?\d+>')
_EMOJI_CUSTOM = re.compile(r'<:\w+:\d+>')
_URL          = re.compile(r'https?://\S+')

_BLOCK_END_SECS   = 3600
_SCENE_BREAK_SECS = 1800

_OPENER_PROMPT = """\
Is this message a literary RP scene opener?
A scene opener: narrative action between *asterisks*, descriptive style, can open a scene autonomously.
Answer JSON only.

Message: {content}

JSON: {{"is_opener": true}}"""


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


def is_preflight_hrp(content: str) -> bool:
    s = content.strip()
    if not s or _SCENE_BREAK.match(s):
        return True
    if _PARENS.search(s):
        return True
    if not _NARRATIVE.search(s):
        return True
    return False


def is_preflight_rp(content: str) -> bool:
    s = content.strip()
    if _PARENS.search(s):
        return False
    matches = _STAR_CONTENT.findall(s)
    return any(len(m.split()) > 3 for m in matches)


def classify_opener(content: str) -> bool:
    cleaned = _clean(content)
    result = call_llm_json(
        _OPENER_PROMPT.format(content=cleaned),
        num_predict=16,
        num_ctx=512,
    )
    return bool(result.get("is_opener", False))


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

    block_active = False
    last_ts      = None
    scene_id     = 0
    kept         = 0
    first        = True

    with open(out_path, "w", encoding="utf-8") as f:
        f.write('{"messages": [')

        for i, msg in enumerate(raw_messages):
            content = msg.get("content", "").strip()
            ts      = _parse_ts(msg.get("timestamp", ""))

            # gap handling
            if last_ts and ts:
                gap = (ts - last_ts).total_seconds()
                if verbose and gap > 60:
                    mins = int(gap // 60)
                    print(f"\n  [gap {mins}min] {'block active' if block_active else 'outside block'} | scene {scene_id} | {ts.strftime('%Y-%m-%d %H:%M')}")
                if block_active:
                    if gap > _BLOCK_END_SECS:
                        block_active = False
                        if verbose:
                            print(f"  -> block ended (gap {int(gap//60)}min > 60min)")
                    elif gap > _SCENE_BREAK_SECS:
                        scene_id += 1
                        if verbose:
                            print(f"  -> new scene {scene_id} (gap {int(gap//60)}min > 30min)")
            if ts:
                last_ts = ts

            if is_preflight_hrp(content):
                continue

            if not block_active:
                if is_preflight_rp(content):
                    if verbose:
                        print(f"  [preflight ok] {content[:60]}")
                    if classify_opener(content):
                        block_active = True
                        scene_id += 1
                        if verbose:
                            print(f"  -> BLOCK OPENED - scene {scene_id} | {ts.strftime('%Y-%m-%d %H:%M') if ts else '?'}")
                    else:
                        if verbose:
                            print(f"  [rejete] {content[:60]}")
                        continue
                else:
                    continue

            f.write(("" if first else ",") + "\n  ")
            json.dump({**msg, "_scene": scene_id}, f, ensure_ascii=False)
            first = False
            kept += 1

            if verbose and (i + 1) % 10 == 0:
                pct = (i + 1) * 100 // total
                print(f"  {filepath.name}: {i+1}/{total} ({pct}%) - {kept} RP [{scene_id} scenes]", end="\r")

        f.write("\n]}")

    if verbose:
        print(f"  {filepath.name}: {total} -> {kept} RP, {total - kept} HRP")

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
