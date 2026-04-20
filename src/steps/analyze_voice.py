"""Step 4f — Voice: per-character speech pattern fingerprint.

Extracts linguistic signature from character lines in a scene.
Updates data/lore/voices/{character_slug}.yaml
"""

import json
import re
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))
from llm import call_llm_json

LORE_DIR = Path(__file__).parent.parent.parent / "data" / "lore"

_PROMPT = """\
Analyze the SPEECH PATTERNS of character "{name}" based on their lines in this RP scene.

Character's lines (actions marked with * and dialogue):
---
{char_lines}
---

Extract ONLY what is clearly visible in these lines. Use null if insufficient data.

- avg_sentence_length: "short" | "medium" | "long"
- dialogue_vs_action_ratio: float 0.0 (all action) to 1.0 (all dialogue)
- formality_level: "very_formal" | "formal" | "neutral" | "informal" | "very_informal"
- vocabulary_register: "academic" | "poetic" | "blunt" | "flowery" | "technical" | "crude" | "archaic" | "mixed"
- communication_style: brief description of HOW they express themselves (direct/indirect, verbose/terse, etc.)
- speech_quirks: recurring expressions, verbal tics, signature phrases (list, empty if none)
- recurring_themes_in_speech: topics or obsessions that come up in their lines (list)
- roleplay_prompt: 2-3 sentences describing how to write in this character's voice, concretely

JSON:
{{
  "avg_sentence_length": null,
  "dialogue_vs_action_ratio": null,
  "formality_level": null,
  "vocabulary_register": null,
  "communication_style": "",
  "speech_quirks": [],
  "recurring_themes_in_speech": [],
  "roleplay_prompt": ""
}}"""

_STAR_RE   = re.compile(r'\*([^*]+)\*')
_DIALOG_RE = re.compile(r'"([^"]+)"')


def _extract_char_lines(messages: list[dict], char_name: str) -> str:
    """Extract lines where the character is the subject of *action* or speaks dialogue."""
    name_low  = char_name.lower()
    name_parts = name_low.split()
    lines = []

    for msg in messages:
        content = msg.get("content_en") or msg.get("content", "")
        content_low = content.lower()

        # Check if the character is mentioned in this message
        if not any(part in content_low for part in name_parts if len(part) > 2):
            continue

        # Extract *action* blocks and dialogue lines mentioning the character
        for action in _STAR_RE.findall(content):
            if any(part in action.lower() for part in name_parts if len(part) > 2):
                lines.append(f"*{action}*")

        for dialog in _DIALOG_RE.findall(content):
            # Include dialogue if the surrounding context names the character
            lines.append(f'"{dialog}"')

        # Also include plain lines (non-action, non-dialogue) if character is named
        plain = content
        plain = _STAR_RE.sub("", plain)
        plain = _DIALOG_RE.sub("", plain)
        plain = plain.strip()
        if plain and any(part in plain.lower() for part in name_parts if len(part) > 2):
            lines.append(plain[:200])

    return "\n".join(lines[:60])  # cap to avoid prompt overflow


def _load_voice_yaml(slug: str) -> dict:
    path = LORE_DIR / "voices" / f"{slug}.yaml"
    if path.exists():
        with open(path, encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return {}


def _save_voice_yaml(slug: str, data: dict):
    voices_dir = LORE_DIR / "voices"
    voices_dir.mkdir(parents=True, exist_ok=True)
    with open(voices_dir / f"{slug}.yaml", "w", encoding="utf-8") as f:
        yaml.dump(data, f, allow_unicode=True, sort_keys=False)


def _slug(name: str) -> str:
    return re.sub(r'\s+', '_', name.lower().strip())


def _merge_voice(existing: dict, new: dict, scene_id: str) -> dict:
    merged = dict(existing)
    merged.setdefault("name", "")
    merged.setdefault("scenes_analyzed", [])
    merged.setdefault("speech_quirks", [])
    merged.setdefault("recurring_themes_in_speech", [])

    # Scalar fields: keep most recent non-null
    for field in ("avg_sentence_length", "formality_level", "vocabulary_register",
                  "communication_style", "dialogue_vs_action_ratio", "roleplay_prompt"):
        val = new.get(field)
        if val is not None and val != "":
            merged[field] = val

    # List fields: accumulate unique
    for item in (new.get("speech_quirks") or []):
        if item.lower() not in [x.lower() for x in merged["speech_quirks"]]:
            merged["speech_quirks"].append(item.lower())

    for item in (new.get("recurring_themes_in_speech") or []):
        if item.lower() not in [x.lower() for x in merged["recurring_themes_in_speech"]]:
            merged["recurring_themes_in_speech"].append(item.lower())

    # Keep roleplay_prompt from most verbose source
    new_rp = (new.get("roleplay_prompt") or "")
    cur_rp = (merged.get("roleplay_prompt") or "")
    if len(new_rp) > len(cur_rp):
        merged["roleplay_prompt"] = new_rp

    if scene_id not in merged["scenes_analyzed"]:
        merged["scenes_analyzed"].append(scene_id)

    return merged


def run_voice(scene_file: Path, analysis_dir: Path, who: dict) -> dict:
    out_path = analysis_dir / "voice.json"

    # Load scene
    try:
        with open(scene_file, encoding="utf-8") as f:
            scene = json.load(f)
    except Exception:
        return {}

    scene_id = scene["scene_id"]
    messages = scene["messages"]

    results = {}
    for char_name in (who.get("characters") or []):
        if not char_name:
            continue
        slug = _slug(char_name)

        existing = _load_voice_yaml(slug)
        # Skip if already analyzed this scene
        if scene_id in (existing.get("scenes_analyzed") or []):
            results[char_name] = existing
            continue

        char_lines = _extract_char_lines(messages, char_name)
        if not char_lines or len(char_lines) < 30:
            continue  # not enough data

        result = call_llm_json(
            _PROMPT.format(name=char_name, char_lines=char_lines),
            num_predict=512,
            num_ctx=4096,
        )

        merged = _merge_voice(existing, result, scene_id)
        merged["name"] = char_name
        _save_voice_yaml(slug, merged)
        results[char_name] = merged
        print(f"    voice updated: {char_name}")

    if results:
        analysis_dir.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps({k: v.get("roleplay_prompt", "") for k, v in results.items()},
                       ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    return results
