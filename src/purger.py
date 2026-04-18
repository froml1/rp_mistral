"""
RP/HRP separator — pre-processing pass before indexing.

Reads raw Discord exports from data/exports/
Writes filtered exports (RP only) to data/exports_filtered/

Uses Mistral via analyzer.py for all classification decisions.

Usage:
  python src/purger.py [data/exports/]
"""

import json
import sys
from pathlib import Path

from analyzer import classify_messages_batched, is_preflight_hrp, is_preflight_rp, _can_inherit_rp, BATCH_SIZE, _parse_ts
from datetime import datetime
from preprocessing import load_messages


OUTPUT_DIR = Path("data/exports_filtered")


def purge_export(filepath: Path, out_path: Path, verbose: bool = True) -> int:
    raw_messages = load_messages(filepath)
    if not raw_messages:
        out_path.write_text('{"messages": []}', encoding="utf-8")
        return 0

    # Tri chronologique
    raw_messages.sort(key=lambda m: _parse_ts(m.get("timestamp", "")) or datetime.min)
    total = len(raw_messages)

    # ── Phase 1 : preflight ───────────────────────────────────────────────────
    candidates = [m for m in raw_messages if not is_preflight_hrp(m.get("content", ""))]
    if verbose:
        dropped = total - len(candidates)
        print(f"  {filepath.name}: {total} msgs — {dropped} preflight, {len(candidates)} → Mistral")

    # ── Phase 2+3 : seed search + chain roll ──────────────────────────────────
    last_rp: dict | None = None   # dernière graine confirmée RP
    pending:  list[dict] = []     # messages en attente de classification LLM
    kept = 0
    first = True

    with open(out_path, "w", encoding="utf-8") as f:
        f.write('{"messages": [')

        def write_msg(msg: dict) -> None:
            nonlocal first, kept
            f.write(("" if first else ",") + "\n  ")
            json.dump(msg, f, ensure_ascii=False)
            first = False
            kept += 1

        processed = 0

        def _progress() -> None:
            pct = processed * 100 // total if total else 0
            auto = kept - _progress.llm_kept
            print(f"  {filepath.name}: {processed}/{total} ({pct}%) "
                  f"— {kept} RP [{auto} auto, {_progress.llm_kept} LLM]", end="\r")
        _progress.llm_kept = 0

        def flush_pending() -> None:
            nonlocal last_rp
            if not pending:
                return
            _, _, results = next(iter(
                classify_messages_batched(pending, batch_size=len(pending), context_overlap=0)
            ))
            for msg, clf in zip(pending, results):
                if clf["is_rp"]:
                    write_msg(msg)
                    last_rp = msg
                    _progress.llm_kept += 1
                elif last_rp is not None:
                    last_rp = None         # chaîne rompue → retour en seed search
            pending.clear()
            f.flush()
            if verbose:
                _progress()

        for msg in candidates:
            processed += 1
            content = msg.get("content", "")

            # Étoile → RP garanti, peut servir de graine
            if is_preflight_rp(content):
                if pending:
                    flush_pending()
                write_msg(msg)
                last_rp = msg

            # Héritage direct si graine active et pas de pending
            elif last_rp is not None and not pending and _can_inherit_rp(msg, last_rp):
                write_msg(msg)
                last_rp = msg

            # Incertain → batch Mistral
            else:
                pending.append(msg)
                if len(pending) >= BATCH_SIZE:
                    flush_pending()

            if verbose and processed % 10 == 0:
                _progress()

        flush_pending()
        f.write("\n]}")

    if verbose:
        print(f"  {filepath.name}: {total} → {kept} RP, {total - kept} HRP      ")

    return kept


def purge_all(exports_dir: str = "data/exports"):
    input_path = Path(exports_dir)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if input_path.is_file():
        files = [input_path]
    else:
        files = list(input_path.glob("**/*.json")) + list(input_path.glob("**/*.csv"))

    if not files:
        print(f"No export files (.json / .csv) found in {exports_dir}")
        return

    print(f"Purging {len(files)} file(s) → {OUTPUT_DIR}/")
    for filepath in files:
        out_path = OUTPUT_DIR / (filepath.stem + ".json")
        purge_export(filepath, out_path)

    print(f"Done. Filtered exports in {OUTPUT_DIR}/")
    print(f"→ Run indexer on filtered exports:")
    print(f"  python src/indexer.py {OUTPUT_DIR}/")


if __name__ == "__main__":
    exports_dir = next((a for a in sys.argv[1:] if not a.startswith("--")), "data/exports")
    purge_all(exports_dir)
