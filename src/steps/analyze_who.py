"""Step 4c — Who: character analysis, updates character YAMLs."""

import json
import re
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from llm import call_llm_json

_PROMPT = """\
Analyze the CHARACTERS in this RP scene.
IMPORTANT: authors (Discord users who write the scene) are NOT characters. Authors to ignore: {authors}

Known information about characters (may be incomplete):
{known_yaml}

For each character (not author) present or active in this scene, extract:
- canonical_name: full name in lowercase (e.g. "lena marchal")
- appellations: all names/references used for this character, excluding pronouns (e.g. ["lena", "miss marchal", "the chrome mask"])
- description_physical: physical appearance details found in this scene
- description_psychological: personality, behavior, emotional state
- job: occupation or social role
- beliefs: beliefs, values, convictions
- main_locations: places associated with this character

Update known info, correct inconsistencies. Only add new details explicitly found in this scene.

JSON: {{"characters": [{{"canonical_name": "", "appellations": [], "description_physical": "", "description_psychological": "", "job": "", "beliefs": [], "main_locations": []}}]}}

Scene:
---
{text}
---"""


def _slug(name: str) -> str:
    return re.sub(r'\s+', '_', name.lower().strip())


def _load_char_yaml(chars_dir: Path, canonical_name: str) -> dict:
    path = chars_dir / f"{_slug(canonical_name)}.yaml"
    if path.exists():
        with open(path, encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return {}


def _save_char_yaml(chars_dir: Path, canonical_name: str, data: dict):
    chars_dir.mkdir(parents=True, exist_ok=True)
    path = chars_dir / f"{_slug(canonical_name)}.yaml"
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(data, f, allow_unicode=True, sort_keys=False)


def _merge_char(existing: dict, extracted: dict, scene_id: str) -> dict:
    merged = dict(existing)
    merged.setdefault("name", extracted.get("canonical_name", ""))
    merged.setdefault("appellations", [])
    merged.setdefault("description_physical", "")
    merged.setdefault("description_psychological", "")
    merged.setdefault("job", "")
    merged.setdefault("beliefs", [])
    merged.setdefault("main_locations", [])
    merged.setdefault("appearances", [])

    for app in (extracted.get("appellations") or []):
        if app.lower() not in [a.lower() for a in merged["appellations"]]:
            merged["appellations"].append(app.lower())

    for field in ("description_physical", "description_psychological", "job"):
        new_val = extracted.get(field) or ""
        if new_val and len(new_val) > len(merged[field]):
            merged[field] = new_val.lower()

    for b in (extracted.get("beliefs") or []):
        if b.lower() not in [x.lower() for x in merged["beliefs"]]:
            merged["beliefs"].append(b.lower())

    for loc in (extracted.get("main_locations") or []):
        if loc.lower() not in [x.lower() for x in merged["main_locations"]]:
            merged["main_locations"].append(loc.lower())

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


def run_who(scene_file: Path, analysis_dir: Path, chars_dir: Path) -> dict:
    out_path = analysis_dir / "who.json"
    if out_path.exists():
        print(f"    [skip] who already done")
        return json.loads(out_path.read_text(encoding="utf-8"))

    with open(scene_file, encoding="utf-8") as f:
        scene = json.load(f)

    scene_id = scene["scene_id"]
    messages = scene["messages"]
    text = _scene_text(messages)

    # Authors = Discord usernames (not characters)
    authors = list({
        (m.get("author") or {}).get("name", "") if isinstance(m.get("author"), dict) else str(m.get("author", ""))
        for m in messages
    } - {""})

    # Load known characters
    known = {}
    if chars_dir.exists():
        for yf in chars_dir.glob("*.yaml"):
            with open(yf, encoding="utf-8") as f:
                d = yaml.safe_load(f) or {}
                if d.get("name"):
                    known[d["name"]] = d

    known_yaml = yaml.dump(known, allow_unicode=True) if known else "none"
    result = call_llm_json(
        _PROMPT.format(authors=", ".join(authors), known_yaml=known_yaml, text=text),
        num_predict=2048,
    )

    characters = [c for c in (result.get("characters") or []) if isinstance(c, dict) and c.get("canonical_name")]

    for char in characters:
        existing = _load_char_yaml(chars_dir, char["canonical_name"])
        merged = _merge_char(existing, char, scene_id)
        _save_char_yaml(chars_dir, char["canonical_name"], merged)
        print(f"    character updated: {char['canonical_name']}")

    output = {
        "characters": [c.get("canonical_name") for c in characters],
        "details":    characters,
    }

    analysis_dir.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    return output
