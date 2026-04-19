"""
RP_IA Pipeline - single entry point.

Usage:
  python src/pipeline.py [exports_dir] [options]

Options:
  --from-step N    Resume from step N (1-4), default 1
  --only-step N    Run only step N
  --scene SCENE_ID Process only this scene (step 4 only)

Steps:
  1 - Purge       data/exports/   -> data/purged/
  2 - Translate   data/purged/    -> data/translated/
  3 - Subdivide   data/translated/ -> data/scenes/
  4 - Analyze     data/scenes/    -> data/analysis/{scene_id}/
                                     data/lore/characters/
                                     data/lore/places/
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from steps.purge        import run_purge
from steps.translate    import run_translate
from steps.subdivide    import run_subdivide
from steps.analyze_when  import run_when
from steps.analyze_where import run_where
from steps.analyze_who   import run_who
from steps.analyze_which import run_which
from steps.analyze_what  import run_what
from steps.analyze_how   import run_how
from steps.general_lore  import (
    update_general_who, update_general_where, update_general_which,
    update_general_what, update_general_how,
)

DATA_DIR     = ROOT / "data"
EXPORTS_DIR  = DATA_DIR / "exports"
PURGED_DIR   = DATA_DIR / "purged"
TRANSLATED_DIR = DATA_DIR / "translated"
SCENES_DIR   = DATA_DIR / "scenes"
LORE_DIR     = DATA_DIR / "lore"
ANALYSIS_DIR = DATA_DIR / "analysis"


def run_step4(scene_files: list[Path], only_scene: str | None = None):
    chars_dir    = LORE_DIR / "characters"
    places_dir   = LORE_DIR / "places"
    concepts_dir = LORE_DIR / "concepts"

    for scene_file in sorted(scene_files):
        scene_id = scene_file.stem
        if only_scene and scene_id != only_scene:
            continue

        print(f"\n  [{scene_id}]")
        ad = ANALYSIS_DIR / scene_id

        when  = run_when(scene_file, ad)
        where = run_where(scene_file, ad, places_dir)
        who   = run_who(scene_file, ad, chars_dir)
        which = run_which(scene_file, ad, concepts_dir)
        what  = run_what(scene_file, ad, when, where, who, which)
        how   = run_how(scene_file, ad, when, where, who, which, what)

        # Update general syntheses — re-injected as context for the next scenes
        update_general_who(chars_dir, LORE_DIR)
        update_general_where(places_dir, LORE_DIR)
        update_general_which(concepts_dir, LORE_DIR)
        update_general_what(LORE_DIR, scene_id, what)
        update_general_how(LORE_DIR, scene_id, how)


def run_pipeline(
    exports_dir: str = "data/exports",
    from_step: int = 1,
    only_step: int | None = None,
    only_scene: str | None = None,
):
    def should_run(step: int) -> bool:
        if only_step is not None:
            return step == only_step
        return step >= from_step

    if should_run(1):
        print("\n== STEP 1 - PURGE ==")
        run_purge(Path(exports_dir), PURGED_DIR)

    if should_run(2):
        print("\n== STEP 2 - TRANSLATE ==")
        run_translate(PURGED_DIR, TRANSLATED_DIR, exports_dir=Path(exports_dir))

    if should_run(3):
        print("\n== STEP 3 - SUBDIVIDE ==")
        run_subdivide(TRANSLATED_DIR, SCENES_DIR, purged_dir=PURGED_DIR)

    if should_run(4):
        print("\n== STEP 4 - ANALYZE ==")
        scene_files = sorted(SCENES_DIR.glob("**/*.json"))
        if not scene_files:
            print("  No scene files found. Run step 3 first.")
            return
        print(f"  {len(scene_files)} scenes to analyze")
        run_step4(scene_files, only_scene=only_scene)

    print("\n== PIPELINE DONE ==")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RP_IA Pipeline")
    parser.add_argument("exports_dir", nargs="?", default="data/exports",
                        help="Raw exports directory (step 1 input)")
    parser.add_argument("--from-step", type=int, default=1, dest="from_step",
                        help="Resume from step N (1-4)")
    parser.add_argument("--only-step", type=int, default=None, dest="only_step",
                        help="Run only step N")
    parser.add_argument("--scene", type=str, default=None,
                        help="Process only this scene ID (step 4)")
    args = parser.parse_args()

    run_pipeline(
        exports_dir=args.exports_dir,
        from_step=args.from_step,
        only_step=args.only_step,
        only_scene=args.scene,
    )
