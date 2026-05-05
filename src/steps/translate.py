"""Step 2 - Translate: add content_en to each scene file."""

import json
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from llm import call_llm_json, model_for

BATCH_SIZE = 20
WORKERS    = int(os.getenv("RP_TRANSLATE_WORKERS", "4"))

_print_lock = threading.Lock()

_PROMPT = """\
Translate these Discord roleplay messages from French to English.

Rules:
- Preserve *narrative action* and "dialogue" markers exactly as-is
- Do NOT translate proper nouns: character names, place names, item or faction names
- Match the tone of each message (formal noble speech stays formal, crude stays crude)
- Empty or punctuation-only messages: copy unchanged
- Output one entry per input index, same order, no skips

Return ONLY valid JSON: {{"messages": [{{"index": 0, "content_en": "..."}}]}}

Messages:
{messages}"""


def _log(msg: str):
    with _print_lock:
        print(msg, flush=True)


def _is_valid_json(path: Path) -> bool:
    try:
        json.loads(path.read_text(encoding="utf-8"))
        return True
    except Exception:
        return False


def _save_scene(out_path: Path, scene: dict):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(scene, ensure_ascii=False, indent=2), encoding="utf-8")


def _translate_batch(messages: list[dict], batch_label: str) -> dict[int, str]:
    formatted = "\n".join(f"[{i}] {m.get('content', '')}" for i, m in enumerate(messages))
    _log(f"    -> batch {batch_label} ({len(messages)} msgs)...")
    # num_predict: 20 msgs × ~150 tokens avg output = 3000, +json overhead → 4096
    data = call_llm_json(_PROMPT.format(messages=formatted), num_predict=4096, num_ctx=8192,
                         model=model_for("translate"))
    result = {
        item["index"]: item.get("content_en", "")
        for item in (data.get("messages") or [])
        if isinstance(item, dict) and "index" in item
    }
    if not result:
        _log(f"    [warn] batch {batch_label}: LLM returned empty — keeping originals")
    elif len(result) < len(messages):
        _log(f"    [warn] batch {batch_label}: got {len(result)}/{len(messages)} translations")
    return result


def _translate_scene_file(fp: Path, out_path: Path):
    """Translate one scene file. Resumes from partial output if it exists."""
    with open(fp, encoding="utf-8") as f:
        scene = json.load(f)
    messages = scene.get("messages", [])

    # Resume: merge already-translated content_en from partial output
    if out_path.exists() and _is_valid_json(out_path):
        with open(out_path, encoding="utf-8") as f:
            existing = json.load(f)
        for i, msg in enumerate(messages):
            ex_msgs = existing.get("messages", [])
            if i < len(ex_msgs) and ex_msgs[i].get("content_en"):
                msg["content_en"] = ex_msgs[i]["content_en"]

    missing = [i for i, m in enumerate(messages) if not m.get("content_en")]
    if not missing:
        _log(f"  [skip] {fp.name}")
        return

    _log(f"  Translating {fp.name}: {len(missing)}/{len(messages)} missing")
    for batch_start in range(0, len(missing), BATCH_SIZE):
        batch_idx = missing[batch_start: batch_start + BATCH_SIZE]
        batch = [messages[i] for i in batch_idx]
        label = f"{batch_start + len(batch_idx)}/{len(missing)}"
        translations = _translate_batch(batch, label)
        for j, global_i in enumerate(batch_idx):
            messages[global_i]["content_en"] = translations.get(j, messages[global_i].get("content", ""))
        _save_scene(out_path, {**scene, "messages": messages})
        _log(f"    OK {label}")


def _passthrough_scene_file(fp: Path, out_path: Path):
    """Copy scene file with content_en = content (no LLM)."""
    if out_path.exists() and _is_valid_json(out_path):
        _log(f"  [skip] {fp.name}")
        return
    with open(fp, encoding="utf-8") as f:
        scene = json.load(f)
    for msg in scene.get("messages", []):
        if not msg.get("content_en"):
            msg["content_en"] = msg.get("content", "")
    _save_scene(out_path, scene)
    _log(f"  [passthrough] {fp.name}: {len(scene.get('messages', []))} msgs")


def run_translate(purged_dir: Path, out_dir: Path,
                  exports_dir: Path | None = None,
                  passthrough: bool = False,
                  workers: int = WORKERS) -> list[Path]:
    """
    Translate all scene files under purged_dir/**/*.json.
    Mirrors the subdir structure into out_dir.
    Scene files are processed in parallel (workers). Each scene serialises its
    own batches internally. Requires Ollama started with OLLAMA_NUM_PARALLEL≥workers.
    """
    scene_files = sorted(purged_dir.glob("**/*.json"))
    if not scene_files:
        _log(f"  No scene files found in {purged_dir}")
        return []

    pairs: list[tuple[Path, Path]] = []
    for fp in scene_files:
        if not _is_valid_json(fp):
            _log(f"  [error] {fp.name} is malformed, skipping")
            continue
        rel      = fp.relative_to(purged_dir)
        out_path = out_dir / rel
        out_path.parent.mkdir(parents=True, exist_ok=True)
        pairs.append((fp, out_path))

    produced: list[Path] = []
    produced_lock = threading.Lock()

    fn = _passthrough_scene_file if passthrough else _translate_scene_file

    _log(f"  [translate] {len(pairs)} scene files — {workers} parallel workers, batch={BATCH_SIZE}")
    _log(f"  [translate] tip: start Ollama with OLLAMA_NUM_PARALLEL={workers} for full speedup")

    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_to_fp = {executor.submit(fn, fp, out_path): out_path for fp, out_path in pairs}
        for future in as_completed(future_to_fp):
            out_path = future_to_fp[future]
            try:
                future.result()
                with produced_lock:
                    produced.append(out_path)
            except Exception as exc:
                _log(f"  [error] {out_path.name}: {exc}")

    return produced
