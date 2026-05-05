"""Step 2 - Translate: add content_en to each scene file via Google Translate."""

import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

# Workers: parallel scene files. Google Translate handles concurrency well.
WORKERS    = int(os.getenv("RP_TRANSLATE_WORKERS", "8"))
_RETRY_MAX = 3
_RETRY_DELAY = 2.0  # seconds between retries on rate-limit

_print_lock = threading.Lock()


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


def _google_translate(text: str) -> str:
    """Translate one string FR→EN via Google Translate (unofficial API)."""
    from deep_translator import GoogleTranslator
    for attempt in range(_RETRY_MAX):
        try:
            result = GoogleTranslator(source="fr", target="en").translate(text)
            return result or text
        except Exception as exc:
            if attempt < _RETRY_MAX - 1:
                time.sleep(_RETRY_DELAY * (attempt + 1))
            else:
                _log(f"    [warn] Google Translate failed: {exc}")
                return text  # keep original on total failure
    return text


def _translate_scene_file(fp: Path, out_path: Path):
    """Translate one scene file. Resumes from partial output if it exists."""
    with open(fp, encoding="utf-8") as f:
        scene = json.load(f)
    messages = scene.get("messages", [])

    # Resume: pull already-translated content_en (marked with _translated flag)
    if out_path.exists() and _is_valid_json(out_path):
        with open(out_path, encoding="utf-8") as f:
            existing = json.load(f)
        ex_msgs = existing.get("messages", [])
        for i, msg in enumerate(messages):
            if i < len(ex_msgs) and ex_msgs[i].get("_translated"):
                msg["content_en"] = ex_msgs[i]["content_en"]
                msg["_translated"] = True

    missing = [i for i, m in enumerate(messages) if not m.get("_translated")]
    if not missing:
        _log(f"  [skip] {fp.name}")
        return

    _log(f"  Translating {fp.name}: {len(missing)}/{len(messages)} messages")

    # Translate missing messages individually (fast enough with Google Translate)
    for i in missing:
        content = messages[i].get("content", "").strip()
        if content:
            messages[i]["content_en"] = _google_translate(content)
        else:
            messages[i]["content_en"] = content
        messages[i]["_translated"] = True

    _save_scene(out_path, {**scene, "messages": messages})
    _log(f"  [done] {fp.name}")


def _passthrough_scene_file(fp: Path, out_path: Path):
    """Copy scene file with content_en = content (no translation)."""
    if out_path.exists() and _is_valid_json(out_path):
        _log(f"  [skip] {fp.name}")
        return
    with open(fp, encoding="utf-8") as f:
        scene = json.load(f)
    for msg in scene.get("messages", []):
        if not msg.get("content_en"):
            msg["content_en"] = msg.get("content", "")
            msg["_translated"] = True
    _save_scene(out_path, scene)
    _log(f"  [passthrough] {fp.name}: {len(scene.get('messages', []))} msgs")


def run_translate(purged_dir: Path, out_dir: Path,
                  exports_dir: Path | None = None,
                  passthrough: bool = False,
                  workers: int = WORKERS) -> list[Path]:
    """
    Translate all scene files under purged_dir/**/*.json via Google Translate.
    Processes scene files in parallel. Each message is translated individually.
    Resume-safe: only messages without the _translated flag are (re)processed.
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

    _log(f"  [translate] {len(pairs)} scene files — {workers} workers (Google Translate FR→EN)")

    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_to_out = {executor.submit(fn, fp, out_path): out_path for fp, out_path in pairs}
        for future in as_completed(future_to_out):
            out_path = future_to_out[future]
            try:
                future.result()
                with produced_lock:
                    produced.append(out_path)
            except Exception as exc:
                _log(f"  [error] {out_path.name}: {exc}")

    return produced
