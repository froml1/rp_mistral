"""
Export lore to Markdown reference documents.

Generates:
  - data/export_md/characters/{name}.md  — full character sheet
  - data/export_md/places/{name}.md      — place reference
  - data/export_md/story_so_far.md       — global narrative summary
  - data/export_md/index.md              — master index

Usage:
  python src/export.py [--all] [--character "rhys"] [--story]
"""

import argparse
import json
import re
import sys
from pathlib import Path

import yaml

ROOT         = Path(__file__).resolve().parent.parent
LORE_DIR     = ROOT / "data" / "lore"
ANALYSIS_DIR = ROOT / "data" / "analysis"
OUT_DIR      = ROOT / "data" / "export_md"

sys.path.insert(0, str(ROOT / "src"))


def _load_yaml(path: Path) -> dict:
    try:
        return yaml.safe_load(path.read_text(encoding="utf-8")) or {} if path.exists() else {}
    except Exception:
        return {}


def _load_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    except Exception:
        return {}


def _load_yaml_dir(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out = []
    for f in sorted(path.glob("*.yaml")):
        try:
            d = yaml.safe_load(f.read_text(encoding="utf-8")) or {}
            d["_file"] = f.stem
            out.append(d)
        except Exception:
            pass
    return out


def _slug(name: str) -> str:
    return re.sub(r'[^\w]', '_', name.lower().strip())


def _list_md(items: list, bullet: str = "-") -> str:
    return "\n".join(f"{bullet} {item}" for item in items) if items else "_none_"


# ── Character sheet ────────────────────────────────────────────────────────────

def export_character(char: dict) -> str:
    name    = char.get("name", "unknown")
    lines   = [f"# {name.title()}", ""]

    # Identity
    lines.append("## Identity")
    if char.get("job"):
        lines.append(f"**Role:** {char['job']}")
    if char.get("appellations"):
        lines.append(f"**Known as:** {', '.join(char['appellations'])}")
    lines.append("")

    # Physical
    if char.get("description_physical"):
        lines += ["## Physical description", char["description_physical"], ""]

    # Psychological
    if char.get("description_psychological"):
        lines += ["## Psychology", char["description_psychological"], ""]

    # Beliefs
    if char.get("beliefs"):
        lines += ["## Beliefs & values", _list_md(char["beliefs"]), ""]

    # Locations
    if char.get("main_locations"):
        lines += ["## Main locations", _list_md(char["main_locations"]), ""]

    # Relations from how.json (cross-scene aggregation)
    char_name_low = name.lower()
    rel_agg: dict[str, list] = {}
    for scene_dir in sorted(ANALYSIS_DIR.iterdir()):
        if not scene_dir.is_dir():
            continue
        how = _load_json(scene_dir / "how.json")
        for rel in (how.get("character_relations") or []):
            frm  = (rel.get("from_char") or "").lower()
            to   = (rel.get("to_char") or "").lower()
            if char_name_low not in frm and char_name_low not in to:
                continue
            other = to if char_name_low in frm else frm
            key   = f"{other}|{rel.get('relation_type','?')}|{rel.get('sentiment','?')}"
            rel_agg.setdefault(other, []).append(
                f"{rel.get('relation_type','?')} ({rel.get('sentiment','?')})"
            )

    if rel_agg:
        lines.append("## Relations")
        for other, rels in sorted(rel_agg.items()):
            # most recent relation type
            lines.append(f"- **{other}**: {rels[-1]}")
        lines.append("")

    # Arc summary across scenes
    appearances = char.get("appearances") or []
    if appearances:
        lines.append(f"## Scene appearances ({len(appearances)})")
        for sid in appearances[:20]:
            what = _load_json(ANALYSIS_DIR / str(sid) / "what.json")
            summary = what.get("summary", "")
            lines.append(f"- `{sid}`: {summary[:80]}" if summary else f"- `{sid}`")
        if len(appearances) > 20:
            lines.append(f"- _... {len(appearances) - 20} more_")
        lines.append("")

    lines.append(f"_First appearance: {char.get('first_appearance', '?')}_")
    return "\n".join(lines)


# ── Place sheet ────────────────────────────────────────────────────────────────

def export_place(place: dict) -> str:
    name  = place.get("name", "unknown")
    lines = [f"# {name.title()}", ""]

    if place.get("appellations"):
        lines.append(f"**Also called:** {', '.join(place['appellations'])}")
        lines.append("")

    if place.get("description"):
        lines += ["## Description", place["description"], ""]

    if place.get("attributes"):
        lines += ["## Attributes", _list_md(place["attributes"]), ""]

    appearances = place.get("appearances") or []
    if appearances:
        lines.append(f"## Scene appearances ({len(appearances)})")
        for sid in appearances[:15]:
            what = _load_json(ANALYSIS_DIR / str(sid) / "what.json")
            summary = what.get("summary", "")
            lines.append(f"- `{sid}`: {summary[:80]}" if summary else f"- `{sid}`")
        lines.append("")

    lines.append(f"_First appearance: {place.get('first_appearance', '?')}_")
    return "\n".join(lines)


# ── Story so far ───────────────────────────────────────────────────────────────

def export_story() -> str:
    general_how  = _load_yaml(LORE_DIR / "general_how.yaml")
    general_what = _load_yaml(LORE_DIR / "general_what.yaml")

    lines = ["# Story so far", ""]

    overall = general_how.get("overall_narrative_summary", "")
    if overall:
        lines += ["## Narrative overview", overall, ""]

    axes = [a for a in (general_how.get("narrative_axes") or []) if a.get("strength") == "strong"]
    if axes:
        lines.append("## Active narrative axes")
        for a in axes:
            lines.append(f"\n### {a.get('name', '?')}")
            lines.append(a.get("summary", ""))
            if a.get("elements"):
                lines.append(f"\n**Key elements:** {', '.join(a['elements'])}")
            if a.get("scenes"):
                lines.append(f"\n**Scenes:** {', '.join(a['scenes'])}")
        lines.append("")

    rec = general_what.get("recurrences") or {}
    if rec.get("themes"):
        lines += ["## Recurring themes", _list_md(rec["themes"]), ""]
    if rec.get("character_interactions"):
        lines += ["## Recurring character interactions", _list_md(rec["character_interactions"]), ""]

    strongest = (general_how.get("strongest_links") or [])[:8]
    if strongest:
        lines.append("## Strongest causal links")
        for lnk in strongest:
            lines.append(
                f"- **{lnk.get('from_element')}** —{lnk.get('link_type')}→ "
                f"**{lnk.get('to_element')}**: {lnk.get('description','')[:100]}"
            )
        lines.append("")

    return "\n".join(lines)


# ── Master index ───────────────────────────────────────────────────────────────

def export_index(characters: list[dict], places: list[dict]) -> str:
    lines = ["# RP Lore — Master Index", ""]

    lines.append(f"## Characters ({len(characters)})")
    for c in sorted(characters, key=lambda x: x.get("name", "")):
        name  = c.get("name", "?")
        job   = c.get("job") or "?"
        apps  = len(c.get("appearances") or [])
        lines.append(f"- [{name.title()}](characters/{_slug(name)}.md) — {job} ({apps} scenes)")
    lines.append("")

    lines.append(f"## Places ({len(places)})")
    for p in sorted(places, key=lambda x: x.get("name", "")):
        name = p.get("name", "?")
        apps = len(p.get("appearances") or [])
        lines.append(f"- [{name.title()}](places/{_slug(name)}.md) ({apps} scenes)")
    lines.append("")

    lines.append("## Documents")
    lines.append("- [Story so far](story_so_far.md)")
    return "\n".join(lines)


# ── Main export ────────────────────────────────────────────────────────────────

def run_export(only_character: str | None = None, only_story: bool = False):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "characters").mkdir(exist_ok=True)
    (OUT_DIR / "places").mkdir(exist_ok=True)

    characters = _load_yaml_dir(LORE_DIR / "characters")
    places     = _load_yaml_dir(LORE_DIR / "places")

    if only_story:
        out = OUT_DIR / "story_so_far.md"
        out.write_text(export_story(), encoding="utf-8")
        print(f"Exported story → {out}")
        return

    if only_character:
        target = next(
            (c for c in characters if only_character.lower() in (c.get("name") or "").lower()),
            None,
        )
        if not target:
            print(f"[error] Character '{only_character}' not found.")
            return
        out = OUT_DIR / "characters" / f"{_slug(target['name'])}.md"
        out.write_text(export_character(target), encoding="utf-8")
        print(f"Exported → {out}")
        return

    # Full export
    for char in characters:
        slug = _slug(char.get("name", char["_file"]))
        out  = OUT_DIR / "characters" / f"{slug}.md"
        out.write_text(export_character(char), encoding="utf-8")
        print(f"  character: {out.name}")

    for place in places:
        slug = _slug(place.get("name", place["_file"]))
        out  = OUT_DIR / "places" / f"{slug}.md"
        out.write_text(export_place(place), encoding="utf-8")
        print(f"  place: {out.name}")

    story_out = OUT_DIR / "story_so_far.md"
    story_out.write_text(export_story(), encoding="utf-8")
    print(f"  story: {story_out.name}")

    index_out = OUT_DIR / "index.md"
    index_out.write_text(export_index(characters, places), encoding="utf-8")
    print(f"  index: {index_out.name}")

    print(f"\nExport complete → {OUT_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--all",       action="store_true",  help="Export everything")
    parser.add_argument("--character", type=str, default=None, help="Export one character")
    parser.add_argument("--story",     action="store_true",  help="Export story summary only")
    args = parser.parse_args()

    if args.story:
        run_export(only_story=True)
    elif args.character:
        run_export(only_character=args.character)
    else:
        run_export()
