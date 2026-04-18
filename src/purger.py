"""
RP/HRP separator — pre-processing pass before indexing.

Reads raw Discord exports from data/exports/
Writes filtered exports (RP only) to data/exports_filtered/

Algorithme bloc :
  - Un bloc RP démarre obligatoirement par un message contenant '*'
  - Le bloc se termine sur un gap > 1h ou un délimiteur de scène (---, ___, ***)
  - Dans un bloc : tout message non-preflight-HRP est conservé
  - Hors bloc : tout message sans '*' est ignoré

Usage:
  python src/purger.py [data/exports/]
"""

import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

from analyzer import is_preflight_hrp, is_preflight_rp, classify_opener, _parse_ts, _SCENE_BREAK
from preprocessing import load_messages


OUTPUT_DIR      = Path("data/exports_filtered")
_BLOCK_END_SECS = 3600  # gap > 1h → fin de bloc RP


def purge_export(filepath: Path, out_path: Path, verbose: bool = True) -> int:
    raw_messages = load_messages(filepath)
    if not raw_messages:
        out_path.write_text('{"messages": []}', encoding="utf-8")
        return 0

    raw_messages.sort(key=lambda m: _parse_ts(m.get("timestamp", "")) or datetime.min)
    total = len(raw_messages)

    block_active   = False
    last_ts        = None
    scene_id       = 0
    kept           = 0
    first          = True
    prev_context: list[str] = []   # derniers contenus de la scène active (pour classify_opener)

    with open(out_path, "w", encoding="utf-8") as f:
        f.write('{"messages": [')

        for i, msg in enumerate(raw_messages):
            content = msg.get("content", "").strip()
            ts      = _parse_ts(msg.get("timestamp", ""))

            # ── fin de bloc : gap > 1h (contexte effacé — trop loin) ─────
            if block_active and last_ts and ts:
                if (ts - last_ts).total_seconds() > _BLOCK_END_SECS:
                    block_active = False
                    prev_context = []

            if ts:
                last_ts = ts

            # ── fin de bloc : délimiteur de scène ─────────────────────────
            if _SCENE_BREAK.match(content):
                block_active = False
                continue

            # ── preflight HRP : toujours éliminé (parens, smileys…) ───────
            if is_preflight_hrp(content):
                continue

            # ── hors bloc : '*' + Mistral valide ouverture et continuité ─
            if not block_active:
                if is_preflight_rp(content):
                    is_opener, is_new_scene = classify_opener(content, prev_context or None)
                    if is_opener:
                        block_active = True
                        if is_new_scene or scene_id == 0:
                            scene_id += 1
                            prev_context = []
                        if verbose:
                            tag = "nouvelle scène" if is_new_scene else "reprise scène"
                            print(f"\n  ↳ scène {scene_id} [{tag}] : {content[:55]}")
                    else:
                        continue
                else:
                    continue

            # ── dans le bloc : écriture avec tag scène ────────────────────
            f.write(("" if first else ",") + "\n  ")
            json.dump({**msg, "_scene": scene_id}, f, ensure_ascii=False)
            first = False
            kept += 1
            prev_context.append(content)
            if len(prev_context) > 10:
                prev_context.pop(0)

            if verbose and (i + 1) % 10 == 0:
                pct = (i + 1) * 100 // total
                print(f"  {filepath.name}: {i+1}/{total} ({pct}%) — {kept} RP  [{scene_id} scènes]", end="\r")

        f.write("\n]}")
        f.flush()

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
    with ThreadPoolExecutor() as pool:
        futures = {
            pool.submit(purge_export, fp, OUTPUT_DIR / (fp.stem + ".json"), False): fp
            for fp in files
        }
        for fut in as_completed(futures):
            fp = futures[fut]
            try:
                kept = fut.result()
                print(f"  {fp.name}: {kept} RP gardés")
            except Exception as exc:
                print(f"  {fp.name}: ERREUR — {exc}")

    print(f"Done. Filtered exports in {OUTPUT_DIR}/")
    print(f"  python src/indexer.py {OUTPUT_DIR}/")


if __name__ == "__main__":
    exports_dir = next((a for a in sys.argv[1:] if not a.startswith("--")), "data/exports")
    purge_all(exports_dir)
