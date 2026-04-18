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

from analyzer import classify_messages_batched
from preprocessing import load_messages


OUTPUT_DIR = Path("data/exports_filtered")


def purge_export(filepath: Path, out_path: Path, verbose: bool = True) -> int:
    raw_messages = load_messages(filepath)
    total = len(raw_messages)

    if verbose:
        print(f"  {filepath.name}: {total} messages")

    kept = 0
    first = True

    with open(out_path, "w", encoding="utf-8") as f:
        f.write('{"messages": [')
        for start, end, batch_results in classify_messages_batched(raw_messages):
            for msg, clf in zip(raw_messages[start:end], batch_results):
                if clf["is_rp"] or clf["is_ooc"]:
                    f.write(("" if first else ",") + "\n  ")
                    json.dump(msg, f, ensure_ascii=False)
                    first = False
                    kept += 1
            f.flush()
            if verbose:
                pct = end * 100 // total if total else 0
                print(f"  {filepath.name}: {end}/{total} ({pct}%) — {kept} RP kept", end="\r")
        f.write("\n]}")

    if verbose:
        print(f"  {filepath.name}: {total} → {kept} RP kept, {total - kept} HRP dropped      ")

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
