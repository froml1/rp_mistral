"""Step 1 - Purge: filter RP messages, tag scenes by time gap."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from purger import purge_export


def _is_valid_json(path: Path) -> bool:
    try:
        import json as _json
        _json.loads(path.read_text(encoding="utf-8"))
        return True
    except Exception:
        return False


def run_purge(exports_dir: Path, out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    files = list(exports_dir.glob("**/*.json")) + list(exports_dir.glob("**/*.csv"))
    if not files:
        print(f"  No export files found in {exports_dir}")
        return []

    produced = []
    for fp in files:
        out_path = out_dir / (fp.stem + ".json")
        if out_path.exists() and _is_valid_json(out_path):
            print(f"  [skip] {fp.name} -> already purged")
            produced.append(out_path)
            continue
        if out_path.exists():
            print(f"  [corrupt] {out_path.name} is malformed, re-purging...")
        print(f"  Purging {fp.name}...")
        purge_export(fp, out_path, verbose=True)
        produced.append(out_path)

    return produced
