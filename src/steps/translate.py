"""Step 2 - Translate: add content_en field to each message."""

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


def _repurge(purged_path: Path, exports_dir: Path):
    from purger import purge_export
    stem = purged_path.stem
    for ext in (".json", ".csv"):
        src = exports_dir / (stem + ext)
        if src.exists():
            purge_export(src, purged_path, verbose=True)
            return
    print(f"  [warn] original export for {stem} not found in {exports_dir}")

BATCH_SIZE = 20

_PROMPT = """\
Translate the following Discord RP messages from French to English.
Preserve narrative style, names, and formatting (*actions*, "dialogue").
Return JSON: {{"messages": [{{"index": 0, "content_en": "..."}}]}}

Messages:
{messages}"""


def _log(msg: str):
    print(msg, flush=True)


def _translate_batch(messages: list[dict], batch_label: str) -> dict[int, str]:
    formatted = "\n".join(
        f"[{i}] {m.get('content', '')}"
        for i, m in enumerate(messages)
    )
    _log(f"    -> envoi batch {batch_label} au LLM ({len(messages)} msgs)...")
    data = call_llm_json(_PROMPT.format(messages=formatted), num_predict=2048, num_ctx=8192)
    return {
        item["index"]: item.get("content_en", "")
        for item in (data.get("messages") or [])
        if isinstance(item, dict) and "index" in item
    }


def run_translate(purged_dir: Path, out_dir: Path, exports_dir: Path | None = None) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    files = list(purged_dir.glob("**/*.json"))
    if not files:
        _log(f"  No purged files found in {purged_dir}")
        return []

    produced = []
    for fp in files:
        out_path = out_dir / fp.name
        if out_path.exists() and _is_valid_json(out_path):
            _log(f"  [skip] {fp.name} -> already translated")
            produced.append(out_path)
            continue
        if out_path.exists():
            _log(f"  [corrupt] {out_path.name} is malformed, re-translating...")

        _log(f"  Translating {fp.name}...")
        if not _is_valid_json(fp):
            if exports_dir:
                _log(f"  [corrupt] {fp.name} is malformed, re-purging...")
                _repurge(fp, exports_dir)
            else:
                _log(f"  [error] {fp.name} is malformed and exports_dir unknown, skipping")
                continue
        with open(fp, encoding="utf-8") as f:
            data = json.load(f)
        messages = data.get("messages", data) if isinstance(data, dict) else data
        total = len(messages)
        _log(f"  {total} messages à traduire par batches de {BATCH_SIZE}")

        for start in range(0, total, BATCH_SIZE):
            batch = messages[start:start + BATCH_SIZE]
            end = min(start + BATCH_SIZE, total)
            label = f"{end}/{total}"
            translations = _translate_batch(batch, label)
            for j, msg in enumerate(batch):
                msg["content_en"] = translations.get(j, msg.get("content", ""))
            _log(f"    OK {label} messages traduits")

        out_path.write_text(
            json.dumps({"messages": messages}, ensure_ascii=False, indent=2),
            encoding="utf-8"
        )
        produced.append(out_path)

    return produced
