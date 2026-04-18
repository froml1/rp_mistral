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

from analyzer import classify_messages_batched, pre_classify_messages, BATCH_SIZE
from preprocessing import load_messages


OUTPUT_DIR = Path("data/exports_filtered")


def purge_export(filepath: Path, out_path: Path, verbose: bool = True) -> int:
    raw_messages = load_messages(filepath)
    total = len(raw_messages)
    if not total:
        out_path.write_text('{"messages": []}', encoding="utf-8")
        return 0

    # ── Phase 1 : pré-classification heuristique (sans LLM) ──────────────────
    pre = pre_classify_messages(raw_messages)
    uncertain_indices = [i for i, s in enumerate(pre) if s == "uncertain"]
    auto_rp = sum(1 for s in pre if s == "rp")

    if verbose:
        print(f"  {filepath.name}: {total} msgs — "
              f"{auto_rp} auto-RP, {len(uncertain_indices)} → Mistral")

    # ── Phase 2 : classification LLM des messages incertains ─────────────────
    # final_clf[i] = {"is_rp": bool, "is_ooc": bool} | None (pending)
    final_clf: list[dict | None] = [None] * total
    for i, status in enumerate(pre):
        if status == "rp":
            final_clf[i] = {"is_rp": True,  "is_ooc": False}
        elif status == "non_rp":
            final_clf[i] = {"is_rp": False, "is_ooc": False}

    # Regrouper les incertains en batches dans leur ordre d'origine
    uncertain_batches = [
        uncertain_indices[b: b + BATCH_SIZE]
        for b in range(0, len(uncertain_indices), BATCH_SIZE)
    ]

    # ── Phase 3 : écriture ordonnée, flush après chaque batch Mistral ────────
    write_ptr = 0  # prochain index à écrire dans le fichier
    kept = 0
    first = True

    def _flush_resolved(f):
        nonlocal write_ptr, kept, first
        while write_ptr < total and final_clf[write_ptr] is not None:
            clf = final_clf[write_ptr]
            if clf["is_rp"] or clf["is_ooc"]:
                f.write(("" if first else ",") + "\n  ")
                json.dump(raw_messages[write_ptr], f, ensure_ascii=False)
                first = False
                kept += 1
            write_ptr += 1

    with open(out_path, "w", encoding="utf-8") as f:
        f.write('{"messages": [')

        for batch_indices in uncertain_batches:
            batch_msgs = [raw_messages[i] for i in batch_indices]
            # classify_messages_batched gère l'overlap en interne ;
            # ici on appelle directement sur le sous-ensemble incertain
            _, _, results = next(iter(classify_messages_batched(batch_msgs, batch_size=len(batch_msgs), context_overlap=0)))
            for idx, clf in zip(batch_indices, results):
                final_clf[idx] = clf

            _flush_resolved(f)
            f.flush()

            if verbose:
                pct = write_ptr * 100 // total
                print(f"  {filepath.name}: {write_ptr}/{total} ({pct}%) — {kept} RP kept", end="\r")

        _flush_resolved(f)  # vider ce qui reste (auto-classifiés en fin de fichier)
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
