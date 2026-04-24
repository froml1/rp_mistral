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
    _log(f"    -> sending batch {batch_label} to LLM ({len(messages)} msgs)...")
    data = call_llm_json(_PROMPT.format(messages=formatted), num_predict=2048, num_ctx=8192)
    return {
        item["index"]: item.get("content_en", "")
        for item in (data.get("messages") or [])
        if isinstance(item, dict) and "index" in item
    }


def _save(out_path: Path, messages: list[dict]):
    out_path.write_text(
        json.dumps({"messages": messages}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def run_passthrough(purged_dir: Path, out_dir: Path) -> list[Path]:
    """Copy purged files to translated dir with content_en = content (no LLM call)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    files = list(purged_dir.glob("**/*.json"))
    if not files:
        _log(f"  No purged files found in {purged_dir}")
        return []

    produced = []
    for fp in files:
        out_path = out_dir / fp.name
        if out_path.exists() and _is_valid_json(out_path):
            _log(f"  [skip] {fp.name} -> already done")
            produced.append(out_path)
            continue
        if not _is_valid_json(fp):
            _log(f"  [error] {fp.name} is malformed, skipping")
            continue
        with open(fp, encoding="utf-8") as f:
            data = json.load(f)
        messages = data.get("messages", data) if isinstance(data, dict) else data
        for msg in messages:
            if not msg.get("content_en"):
                msg["content_en"] = msg.get("content", "")
        _save(out_path, messages)
        _log(f"  [passthrough] {fp.name}: {len(messages)} messages")
        produced.append(out_path)
    return produced


def run_translate(purged_dir: Path, out_dir: Path, exports_dir: Path | None = None, passthrough: bool = False) -> list[Path]:
    if passthrough:
        return run_passthrough(purged_dir, out_dir)

    out_dir.mkdir(parents=True, exist_ok=True)
    files = list(purged_dir.glob("**/*.json"))
    if not files:
        _log(f"  No purged files found in {purged_dir}")
        return []

    produced = []
    for fp in files:
        out_path = out_dir / fp.name

        # Load source
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

        # Load partial progress if the output already exists
        if out_path.exists() and _is_valid_json(out_path):
            with open(out_path, encoding="utf-8") as f:
                existing = json.load(f)
            existing_msgs = existing.get("messages", existing) if isinstance(existing, dict) else existing
            # Merge already-translated content_en back into messages (matched by index)
            for i, msg in enumerate(messages):
                if i < len(existing_msgs) and existing_msgs[i].get("content_en"):
                    msg["content_en"] = existing_msgs[i]["content_en"]

        # Identify which messages still need translation
        missing_indices = [i for i, m in enumerate(messages) if not m.get("content_en")]
        if not missing_indices:
            _log(f"  [skip] {fp.name} -> all {len(messages)} messages already translated")
            produced.append(out_path)
            continue

        _log(f"  Translating {fp.name}: {len(missing_indices)}/{len(messages)} messages missing")

        # Process missing messages in batches
        for batch_start in range(0, len(missing_indices), BATCH_SIZE):
            batch_idx = missing_indices[batch_start:batch_start + BATCH_SIZE]
            batch = [messages[i] for i in batch_idx]
            end = batch_start + len(batch_idx)
            label = f"{end}/{len(missing_indices)} missing"
            translations = _translate_batch(batch, label)
            for j, global_i in enumerate(batch_idx):
                messages[global_i]["content_en"] = translations.get(j, messages[global_i].get("content", ""))
            # Save after each batch so a crash doesn't lose progress
            _save(out_path, messages)
            _log(f"    OK {label} translated")

        produced.append(out_path)

    return produced
