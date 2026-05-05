"""Step 2 - Translate: add content_en to each scene file via Google Translate."""

import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

WORKERS   = int(os.getenv("RP_TRANSLATE_WORKERS", "4"))
_MAX_CHARS = 4000   # safe margin under Google's 5000-char limit
_SEP       = "\n⁂\n"  # asterism — Google Translate leaves symbols like this untouched
_RETRY_MAX = 3

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


def _gt(text: str) -> str:
    """Single Google Translate call with retry."""
    from deep_translator import GoogleTranslator
    for attempt in range(_RETRY_MAX):
        try:
            return GoogleTranslator(source="fr", target="en").translate(text) or text
        except Exception as exc:
            if attempt < _RETRY_MAX - 1:
                time.sleep(2 ** attempt)
            else:
                _log(f"    [warn] GT failed: {exc}")
                return text
    return text


def _translate_batch(texts: list[str]) -> list[str]:
    """
    Translate a list of strings in one Google Translate call.
    Strings are joined with ⁂ (a symbol GT ignores) then split back.
    Falls back to individual calls if the separator is mangled.
    """
    if not texts:
        return []
    combined = _SEP.join(texts)
    translated = _gt(combined)
    parts = translated.split(_SEP)
    if len(parts) == len(texts):
        return parts
    # separator got mangled — translate individually (slower but correct)
    _log(f"    [warn] separator mangled in batch of {len(texts)}, falling back to individual calls")
    return [_gt(t) for t in texts]


def _make_batches(items: list[tuple[int, str]]) -> list[list[tuple[int, str]]]:
    """Group (index, text) pairs into batches not exceeding _MAX_CHARS."""
    batches: list[list[tuple[int, str]]] = []
    current: list[tuple[int, str]] = []
    current_len = 0
    for idx, text in items:
        cost = len(text) + len(_SEP)
        if current and current_len + cost > _MAX_CHARS:
            batches.append(current)
            current = []
            current_len = 0
        current.append((idx, text))
        current_len += cost
    if current:
        batches.append(current)
    return batches


def _translate_scene_file(fp: Path, out_path: Path):
    """Translate one scene file. Resume-safe via _translated flag."""
    with open(fp, encoding="utf-8") as f:
        scene = json.load(f)
    messages = scene.get("messages", [])

    if out_path.exists() and _is_valid_json(out_path):
        with open(out_path, encoding="utf-8") as f:
            existing = json.load(f)
        ex_msgs = existing.get("messages", [])
        for i, msg in enumerate(messages):
            if i < len(ex_msgs) and ex_msgs[i].get("_translated"):
                msg["content_en"] = ex_msgs[i]["content_en"]
                msg["_translated"] = True

    to_translate = [(i, m.get("content", "").strip())
                    for i, m in enumerate(messages)
                    if not m.get("_translated")]
    if not to_translate:
        _log(f"  [skip] {fp.name}")
        return

    _log(f"  {fp.name}: {len(to_translate)}/{len(messages)} to translate")

    batches = _make_batches(to_translate)
    for batch in batches:
        indices = [idx for idx, _ in batch]
        texts   = [text for _, text in batch]
        results = _translate_batch(texts)
        for idx, translated in zip(indices, results):
            messages[idx]["content_en"] = translated
            messages[idx]["_translated"] = True
        _save_scene(out_path, {**scene, "messages": messages})

    _log(f"  [done] {fp.name} ({len(batches)} batches)")


def _passthrough_scene_file(fp: Path, out_path: Path):
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
    _log(f"  [passthrough] {fp.name}")


def run_translate(purged_dir: Path, out_dir: Path,
                  exports_dir: Path | None = None,
                  passthrough: bool = False,
                  workers: int = WORKERS) -> list[Path]:
    scene_files = sorted(purged_dir.glob("**/*.json"))
    if not scene_files:
        _log(f"  No scene files found in {purged_dir}")
        return []

    pairs: list[tuple[Path, Path]] = []
    for fp in scene_files:
        if not _is_valid_json(fp):
            _log(f"  [error] {fp.name} malformed, skipping")
            continue
        rel = fp.relative_to(purged_dir)
        out_path = out_dir / rel
        out_path.parent.mkdir(parents=True, exist_ok=True)
        pairs.append((fp, out_path))

    produced: list[Path] = []
    produced_lock = threading.Lock()
    fn = _passthrough_scene_file if passthrough else _translate_scene_file

    _log(f"  [translate] {len(pairs)} scenes — {workers} workers, ~{_MAX_CHARS} chars/batch")

    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_to_out = {executor.submit(fn, fp, op): op for fp, op in pairs}
        for future in as_completed(future_to_out):
            op = future_to_out[future]
            try:
                future.result()
                with produced_lock:
                    produced.append(op)
            except Exception as exc:
                _log(f"  [error] {op.name}: {exc}")

    return produced
