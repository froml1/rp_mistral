"""Step 4d — Which: named concepts present in the scene (objects, factions, ideologies, systems).
Runs before What, feeds it as thematic context. Updates data/lore/concepts/*.yaml."""

import json
import re
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from llm import call_llm_json

_PROMPT = """\
Analyze the CONCEPTS present in this RP scene.
Concepts are named, specific elements that are neither characters nor locations nor events.
They include: named objects of significance, factions/organizations, ideologies/beliefs, laws/systems, technologies, rituals, named artifacts.

IMPORTANT rules:
- EXCLUDE generic/common items (a door, a table, a weapon) — only include named or narratively significant ones
- EXCLUDE pronouns and vague references
- INCLUDE only concepts that are discussed, debated, or meaningfully referenced in the scene
- If a concept appears trivial or purely incidental, omit it

Known concepts (may be incomplete):
{known_yaml}

For each qualifying concept:
- canonical_name: specific name in lowercase (e.g. "the iron covenant", "ley crystal", "doctrine of silence")
- type: object | faction | ideology | system | artifact | other
- appellations: all names/references used for this concept in the scene
- description: what it is, what role it plays
- related_characters: characters associated with it in this scene
- significance: why it matters narratively (one sentence)

JSON: {{"concepts": [{{"canonical_name": "", "type": "", "appellations": [], "description": "", "related_characters": [], "significance": ""}}]}}

Scene:
---
{text}
---"""


def _slug(name: str) -> str:
    return re.sub(r'\s+', '_', name.lower().strip())


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
    merged.setdefault("appellations", [])
    merged.setdefault("description", "")
    merged.setdefault("significance", "")
    merged.setdefault("related_characters", [])
    merged.setdefault("appearances", [])

    for app in (extracted.get("appellations") or []):
        if app.lower() not in [a.lower() for a in merged["appellations"]]:
            merged["appellations"].append(app.lower())

    new_desc = extracted.get("description") or ""
    if new_desc and len(new_desc) > len(merged["description"]):
        merged["description"] = new_desc.lower()

    new_sig = extracted.get("significance") or ""
    if new_sig and len(new_sig) > len(merged["significance"]):
        merged["significance"] = new_sig.lower()

    for char in (extracted.get("related_characters") or []):
        if char.lower() not in [c.lower() for c in merged["related_characters"]]:
            merged["related_characters"].append(char.lower())

    if scene_id not in merged["appearances"]:
        merged["appearances"].append(scene_id)

    merged.setdefault("first_appearance", scene_id)
    return merged


def _scene_text(messages: list[dict]) -> str:
    return "\n".join(
        f"{(m.get('author') or {}).get('name', '?') if isinstance(m.get('author'), dict) else m.get('author', '?')}: "
        f"{m.get('content_en') or m.get('content', '')}"
        for m in messages
    )


def run_which(scene_file: Path, analysis_dir: Path, concepts_dir: Path) -> dict:
    out_path = analysis_dir / "which.json"
    if out_path.exists():
        print(f"    [skip] which already done")
        return json.loads(out_path.read_text(encoding="utf-8"))

    with open(scene_file, encoding="utf-8") as f:
        scene = json.load(f)

    scene_id = scene["scene_id"]
    text = _scene_text(scene["messages"])

    known = {}
    if concepts_dir.exists():
        for yf in concepts_dir.glob("*.yaml"):
            with open(yf, encoding="utf-8") as f:
                d = yaml.safe_load(f) or {}
                if d.get("name"):
                    known[d["name"]] = d

    known_yaml = yaml.dump(known, allow_unicode=True) if known else "none"
    result = call_llm_json(
        _PROMPT.format(known_yaml=known_yaml, text=text),
        num_predict=2048,
    )

    concepts = [
        c for c in (result.get("concepts") or [])
        if isinstance(c, dict) and c.get("canonical_name")
    ]

    for concept in concepts:
        existing = _load_concept_yaml(concepts_dir, concept["canonical_name"])
        merged = _merge_concept(existing, concept, scene_id)
        _save_concept_yaml(concepts_dir, concept["canonical_name"], merged)
        print(f"    concept updated: {concept['canonical_name']}")

    output = {
        "concepts": [c.get("canonical_name") for c in concepts],
        "details":  concepts,
    }

    analysis_dir.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"    which: {len(concepts)} concepts")
    return output
