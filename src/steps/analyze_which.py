"""Step 4c — Which: named concepts present in the scene (full scene sweep, dedicated call)."""

import json
import re
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from llm import call_llm_json
from steps.manual_lore import load_manual_concept, load_all_manual_concepts, merge_manual_into_concept
from steps.synthesis import current_scene_synthesis
from steps.scene_patch import chunk_messages
from lore_summary import update_summary
try:
    from store import upsert as _store_upsert
except Exception:
    def _store_upsert(*a, **kw): pass


def _is_valid_json(path: Path) -> bool:
    try:
        json.loads(path.read_text(encoding="utf-8"))
        return True
    except Exception:
        return False


_PROMPT = """\
Read this roleplay scene and list every named concept that appears in it.

SCENE OVERVIEW:
{scene_synthesis}

A concept is any named thing that is NOT a character and NOT a place:
- factions, guilds, orders, organisations (e.g. "la guilde des ombres", "l'ordre du crépuscule")
- named artifacts, weapons, objects of significance (e.g. "l'épée runique", "le cristal de mémoire")
- ideologies, religions, doctrines (e.g. "le culte du vide", "la loi du sang")
- magic systems, technologies, rituals (e.g. "la magie runique", "le rituel d'ancrage")
- named laws, pacts, treaties (e.g. "le traité de cendres")

Do NOT include: character names, place names, common objects (a sword, a door), pronouns.
Characters already identified (exclude): {known_chars}
Locations already identified (exclude): {known_places}

Known concepts from previous scenes (use these canonical names if the same concept reappears):
{known_yaml}

For each concept found:
- canonical_name: specific name in lowercase
- type: object | faction | ideology | system | artifact | other
- appellations: all names/variants used in this scene
- description: what it is and what role it plays here
- related_characters: characters linked to it in this scene
- significance: why it matters (one sentence)
- status: active | defunct | contested | legendary | unknown
- location: where this faction/artifact is based (place name or "unknown")
- allies: allied factions or characters (for factions)
- enemies: opposed factions or characters (for factions)
- access: free | earned | restricted | forbidden | unknown

If no named concepts appear in this scene, return an empty list.

JSON: {{"concepts": [{{"canonical_name": "", "type": "", "appellations": [], "description": "", "related_characters": [], "significance": "", "status": "unknown", "location": "unknown", "allies": [], "enemies": [], "access": "unknown"}}]}}

Scene:
---
{text}
---"""


def _slug(name: str) -> str:
    return re.sub(r'\s+', '_', name.lower().strip())


def _scene_text(messages: list[dict]) -> str:
    return "\n".join(
        f"{m.get('content_en') or m.get('content', '')}"
        for m in messages
    )


def _load_concept_yaml(concepts_dir: Path, canonical_name: str) -> dict:
    path = concepts_dir / f"{_slug(canonical_name)}.yaml"
    if path.exists():
        with open(path, encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return {}


def _save_concept_yaml(concepts_dir: Path, canonical_name: str, data: dict):
    concepts_dir.mkdir(parents=True, exist_ok=True)
    path = concepts_dir / f"{_slug(canonical_name)}.yaml"
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(data, f, allow_unicode=True, sort_keys=False)


def _merge_concept(existing: dict, extracted: dict, scene_id: str) -> dict:
    merged = dict(existing)
    merged.setdefault("name", extracted.get("canonical_name", ""))
    merged.setdefault("type", extracted.get("type", "other"))
    for lst in ("appellations", "related_characters", "allies", "enemies", "appearances"):
        merged.setdefault(lst, [])
    merged.setdefault("description", "")
    merged.setdefault("significance", "")
    for field in ("status", "location", "access"):
        merged.setdefault(field, "")

    for app in (extracted.get("appellations") or []):
        if app.lower() not in [a.lower() for a in merged["appellations"]]:
            merged["appellations"].append(app.lower())

    for field in ("description", "significance"):
        new_val = extracted.get(field) or ""
        if new_val and len(new_val) > len(merged[field]):
            merged[field] = new_val.lower()

    for lst in ("related_characters", "allies", "enemies"):
        for item in (extracted.get(lst) or []):
            if item.lower() not in [x.lower() for x in merged[lst]]:
                merged[lst].append(item.lower())

    for field in ("status", "location", "access"):
        new_val = (extracted.get(field) or "").strip().lower()
        if new_val and new_val != "unknown":
            merged[field] = new_val

    if scene_id not in merged["appearances"]:
        merged["appearances"].append(scene_id)
    merged.setdefault("first_appearance", scene_id)
    return merged


def run_which(
    scene_file: Path,
    analysis_dir: Path,
    concepts_dir: Path,
    known_chars: list[str] | None = None,
    known_places: list[str] | None = None,
    lore_dir: Path | None = None,
) -> dict:
    out_path = analysis_dir / "which.json"
    if out_path.exists() and _is_valid_json(out_path):
        print("    [skip] which already done")
        return json.loads(out_path.read_text(encoding="utf-8"))
    if out_path.exists():
        print("    [corrupt] which.json malformed, re-analyzing...")

    if not _is_valid_json(scene_file):
        print(f"    [error] scene file {scene_file.name} is malformed")
        return {}

    with open(scene_file, encoding="utf-8") as f:
        scene = json.load(f)

    scene_id = scene["scene_id"]
    messages = scene["messages"]

    known = {}
    if concepts_dir.exists():
        for yf in concepts_dir.glob("*.yaml"):
            with open(yf, encoding="utf-8") as f:
                d = yaml.safe_load(f) or {}
                if d.get("name"):
                    known[d["name"]] = d
    for name, mc in load_all_manual_concepts().items():
        known[name] = merge_manual_into_concept(known.get(name, {}), mc) if name in known else mc

    known_yaml       = "\n".join(f"- {n}: {d.get('_summary', '')}" for n, d in known.items()) or "none"
    scene_synthesis  = current_scene_synthesis(lore_dir, scene_id) if lore_dir else "none"
    known_chars_str  = ", ".join(known_chars or []) or "none"
    known_places_str = ", ".join(known_places or []) or "none"

    scene_chars_lower = {n.lower() for n in (known_chars or [])}
    exclude = scene_chars_lower | {n.lower() for n in (known_places or [])}

    def _call_chunk(chunk: list[dict]) -> list[dict]:
        r = call_llm_json(
            _PROMPT.format(
                scene_synthesis=scene_synthesis,
                known_chars=known_chars_str,
                known_places=known_places_str,
                known_yaml=known_yaml,
                text=_scene_text(chunk),
            ),
            num_predict=2048,
            num_ctx=12288,
        )
        return r.get("concepts") or []

    chunks = chunk_messages(messages)
    if len(chunks) > 1:
        print(f"    which: {len(chunks)} chunks")

    # Collect all concepts across chunks, deduplicate by canonical_name
    concept_map: dict[str, dict] = {}
    for chunk in chunks:
        for c in _call_chunk(chunk):
            if not isinstance(c, dict) or not c.get("canonical_name"):
                continue
            cname = c["canonical_name"].lower().strip()
            if not cname or cname.startswith("unknown"):
                continue
            if cname in exclude:
                continue
            if cname not in concept_map:
                concept_map[cname] = c
            else:
                # Merge: take longer description/significance, union lists
                ex = concept_map[cname]
                for field in ("description", "significance"):
                    if len(c.get(field, "")) > len(ex.get(field, "")):
                        ex[field] = c[field]
                for lst in ("appellations", "related_characters", "allies", "enemies"):
                    seen = {x.lower() for x in (ex.get(lst) or [])}
                    for item in (c.get(lst) or []):
                        if item and item.lower() not in seen:
                            ex.setdefault(lst, []).append(item)
                            seen.add(item.lower())

    # Filter related_characters: only keep characters present in this scene
    for c in concept_map.values():
        if scene_chars_lower and c.get("related_characters"):
            c["related_characters"] = [
                r for r in c["related_characters"]
                if r.lower() in scene_chars_lower
            ]

    concepts = list(concept_map.values())

    for concept in concepts:
        existing = _load_concept_yaml(concepts_dir, concept["canonical_name"])
        merged   = _merge_concept(existing, concept, scene_id)
        merged   = merge_manual_into_concept(merged, load_manual_concept(concept["canonical_name"]))
        _save_concept_yaml(concepts_dir, concept["canonical_name"], merged)
        summary  = update_summary(concepts_dir / f"{_slug(concept['canonical_name'])}.yaml", "concepts")
        _store_upsert("concepts", merged["name"], summary)
        print(f"    concept updated: {concept['canonical_name']}")

    output = {
        "concepts": [c.get("canonical_name") for c in concepts],
        "details":  concepts,
    }

    analysis_dir.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"    which: {len(concepts)} concept(s)")
    return output
