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

from analyzer import analyze_messages
from preprocessing import load_messages


OUTPUT_DIR = Path("data/exports_filtered")


def purge_export(filepath: Path, verbose: bool = True) -> dict:
    raw_messages = load_messages(filepath)
    analyses = analyze_messages(raw_messages)

    rp_messages = [
        msg for msg, analysis in zip(raw_messages, analyses)
        if analysis["is_rp"] or analysis["is_ooc"]
    ]

    if verbose:
        total = len(raw_messages)
        kept = len(rp_messages)
        print(f"  {filepath.name}: {total} messages → {kept} RP kept, {total - kept} HRP dropped")

    return {"messages": rp_messages}


def purge_all(exports_dir: str = "data/exports"):
    input_path = Path(exports_dir)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    json_files = list(input_path.glob("**/*.json")) + list(input_path.glob("**/*.csv"))
    if not json_files:
        print(f"No export files (.json / .csv) found in {exports_dir}")
        return

    print(f"Purging {len(json_files)} file(s) → {OUTPUT_DIR}/")
    for filepath in json_files:
        filtered = purge_export(filepath)
        out_path = OUTPUT_DIR / (filepath.stem + ".json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(filtered, f, ensure_ascii=False, indent=2)

    print(f"Done. Filtered exports in {OUTPUT_DIR}/")
    print(f"→ Run indexer on filtered exports:")
    print(f"  python src/indexer.py {OUTPUT_DIR}/")


if __name__ == "__main__":
    exports_dir = next((a for a in sys.argv[1:] if not a.startswith("--")), "data/exports")
    purge_all(exports_dir)
