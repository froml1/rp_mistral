"""Step 1 - Purge: filter messages, split into scene files by time gap."""

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
        stem_dir = out_dir / fp.stem
        # Skip if already done (stem_dir exists and has at least one scene file)
        existing = sorted(stem_dir.glob("*.json")) if stem_dir.exists() else []
        if existing:
            print(f"  [skip] {fp.name} -> {len(existing)} scene(s) already purged")
            produced.extend(existing)
            continue

        print(f"  Purging {fp.name}...")
        written = purge_export(fp, stem_dir, verbose=True)
        produced.extend(written)

    return produced
