"""
RP_IA Pipeline - single entry point.

Usage:
  python src/pipeline.py [exports_dir] [options]

Options:
  --from-step N    Resume from step N (1-6), default 1
  --only-step N    Run only step N
  --scene SCENE_ID Process only this scene (step 5 only)

Steps:
  1 - Purge       data/exports/    -> data/purged/
  2 - Translate   data/purged/     -> data/translated/
  3 - Subdivide   data/translated/ -> data/scenes/
  4 - Synthesis   data/scenes/     -> data/lore/lore_how.yaml  (narrative + RP quality check)
                                      data/rp_report.json      (non-RP scenes for review)
  5 - Analyze     data/scenes/     -> data/analysis/{scene_id}/  (skips non-RP scenes)
                                      data/lore/characters/ places/ concepts/
  6 - Post        voice fingerprints + general syntheses (batch, after all scenes)
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from steps.purge         import run_purge
from steps.translate     import run_translate
from steps.subdivide     import run_subdivide
from steps.synthesis     import run_synthesis, is_scene_rp
from steps.analyze_context  import run_context
from steps.analyze_entities import run_entities
from steps.analyze_what     import run_what
from steps.analyze_how      import run_how
from steps.analyze_voice    import run_voice
from steps.general_lore  import (
    update_general_who, update_general_where, update_general_which,
    update_general_what,
)

DATA_DIR       = ROOT / "data"
EXPORTS_DIR    = DATA_DIR / "exports"
PURGED_DIR     = DATA_DIR / "purged"
TRANSLATED_DIR = DATA_DIR / "translated"
SCENES_DIR     = DATA_DIR / "scenes"
LORE_DIR       = DATA_DIR / "lore"
ANALYSIS_DIR   = DATA_DIR / "analysis"


def run_step5(scene_files: list[Path], only_scene: str | None = None):
    chars_dir    = LORE_DIR / "characters"
    places_dir   = LORE_DIR / "places"
    concepts_dir = LORE_DIR / "concepts"

    skipped_rp = 0
    for scene_file in sorted(scene_files):
        scene_id = scene_file.stem
        if only_scene and scene_id != only_scene:
            continue

        if not is_scene_rp(LORE_DIR, scene_id):
            print(f"\n  [skip non-RP] {scene_id}")
            skipped_rp += 1
            continue

        print(f"\n  [{scene_id}]")
        ad = ANALYSIS_DIR / scene_id

        when, where = run_context(scene_file, ad, places_dir, lore_dir=LORE_DIR)
        who, which  = run_entities(scene_file, ad, chars_dir, concepts_dir, lore_dir=LORE_DIR)
        what        = run_what(scene_file, ad, when, where, who, which, lore_dir=LORE_DIR)
        run_how(scene_file, ad, when, where, who, which, what, lore_dir=LORE_DIR)

    if skipped_rp:
        print(f"\n  [analyze] {skipped_rp} non-RP scene(s) skipped — see data/rp_report.json")


def run_step6(scene_files: list[Path], only_scene: str | None = None):
    """Post-processing: voice fingerprints + general syntheses. Runs after all scenes."""
    chars_dir    = LORE_DIR / "characters"
    places_dir   = LORE_DIR / "places"
    concepts_dir = LORE_DIR / "concepts"

    for scene_file in sorted(scene_files):
        scene_id = scene_file.stem
        if only_scene and scene_id != only_scene:
            continue
        if not is_scene_rp(LORE_DIR, scene_id):
            continue
        ad = ANALYSIS_DIR / scene_id
        who_path = ad / "who.json"
        if not who_path.exists():
            continue
        who = json.loads(who_path.read_text(encoding="utf-8"))
        print(f"\n  [voice] {scene_id}")
        run_voice(scene_file, ad, who)

    print("\n  [generals] compiling who / where / which…")
    update_general_who(chars_dir, LORE_DIR)
    update_general_where(places_dir, LORE_DIR)
    update_general_which(concepts_dir, LORE_DIR)

    print("  [store] rebuilding ChromaDB index…")
    try:
        from store import reindex
        reindex(LORE_DIR)
    except Exception as e:
        print(f"  [store] warning — index failed: {e}")

    print("  [generals] what + how (LLM summary per scene)…")
    for scene_file in sorted(scene_files):
        scene_id = scene_file.stem
        if only_scene and scene_id != only_scene:
            continue
        if not is_scene_rp(LORE_DIR, scene_id):
            continue
        ad = ANALYSIS_DIR / scene_id
        what_path = ad / "what.json"
        if what_path.exists():
            what = json.loads(what_path.read_text(encoding="utf-8"))
            update_general_what(LORE_DIR, scene_id, what)


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
        print("\n== STEP 4 - SYNTHESIS ==")
        scene_files = sorted(SCENES_DIR.glob("**/*.json"))
        if not scene_files:
            print("  No scene files found. Run step 3 first.")
            return
        print(f"  {len(scene_files)} scenes to synthesize")
        run_synthesis(SCENES_DIR, LORE_DIR, report_path=DATA_DIR / "rp_report.json")

    if should_run(5):
        print("\n== STEP 5 - ANALYZE ==")
        scene_files = sorted(SCENES_DIR.glob("**/*.json"))
        if not scene_files:
            print("  No scene files found. Run step 3 first.")
            return
        print(f"  {len(scene_files)} scenes to analyze")
        run_step5(scene_files, only_scene=only_scene)

    if should_run(6):
        print("\n== STEP 6 - POST (voice + generals) ==")
        scene_files = sorted(SCENES_DIR.glob("**/*.json"))
        if not scene_files:
            print("  No scene files found.")
            return
        run_step6(scene_files, only_scene=only_scene)

    print("\n== PIPELINE DONE ==")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RP_IA Pipeline")
    parser.add_argument("exports_dir", nargs="?", default="data/exports",
                        help="Raw exports directory (step 1 input)")
    parser.add_argument("--from-step", type=int, default=1, dest="from_step",
                        help="Resume from step N (1-6)")
    parser.add_argument("--only-step", type=int, default=None, dest="only_step",
                        help="Run only step N (1-6)")
    parser.add_argument("--scene", type=str, default=None,
                        help="Process only this scene ID (step 6)")
    args = parser.parse_args()

    run_pipeline(
        exports_dir=args.exports_dir,
        from_step=args.from_step,
        only_step=args.only_step,
        only_scene=args.scene,
    )
