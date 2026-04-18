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

from analyzer import (
    classify_messages_batched, pre_classify_messages, extend_rp_chain,
    BATCH_SIZE, _parse_ts,
)
from datetime import datetime
from preprocessing import load_messages


OUTPUT_DIR = Path("data/exports_filtered")
CHUNK_SIZE  = BATCH_SIZE * 3  # messages traités par itération


def purge_export(filepath: Path, out_path: Path, verbose: bool = True) -> int:
    raw_messages = load_messages(filepath)
    if not raw_messages:
        out_path.write_text('{"messages": []}', encoding="utf-8")
        return 0

    # Tri chronologique — les CSV Discord sont souvent en ordre inverse
    raw_messages.sort(key=lambda m: _parse_ts(m.get("timestamp", "")) or datetime.min)

    total = len(raw_messages)
    kept  = 0
    first = True
    seed_msg: dict | None = None  # dernier message confirmé RP du chunk précédent

    with open(out_path, "w", encoding="utf-8") as f:
        f.write('{"messages": [')

        for chunk_start in range(0, total, CHUNK_SIZE):
            chunk = raw_messages[chunk_start: chunk_start + CHUNK_SIZE]

            # ── pré-classification avec graine du chunk précédent ─────────
            statuses = pre_classify_messages(chunk, seed_msg=seed_msg)

            # ── LLM sur les incertains ────────────────────────────────────
            uncertain_local = [i for i, s in enumerate(statuses) if s == "uncertain"]
            if uncertain_local:
                batch_msgs = [chunk[i] for i in uncertain_local]
                _, _, results = next(iter(
                    classify_messages_batched(batch_msgs, batch_size=len(batch_msgs), context_overlap=0)
                ))
                for local_i, clf in zip(uncertain_local, results):
                    statuses[local_i] = "rp" if (clf["is_rp"] or clf["is_ooc"]) else "non_rp"

                # ── extension de chaîne après résultats LLM ───────────────
                statuses = extend_rp_chain(chunk, statuses)

            # ── écriture du chunk dans l'ordre ────────────────────────────
            auto_rp = sum(1 for s in statuses if s == "rp")
            for msg, status in zip(chunk, statuses):
                if status == "rp":
                    f.write(("" if first else ",") + "\n  ")
                    json.dump(msg, f, ensure_ascii=False)
                    first = False
                    kept += 1
                    seed_msg = msg

            f.flush()
            if verbose:
                done = chunk_start + len(chunk)
                pct  = done * 100 // total
                print(f"  {filepath.name}: {done}/{total} ({pct}%) — {kept} RP kept"
                      f"  [{auto_rp}/{len(chunk)} auto]", end="\r")

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
