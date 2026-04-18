"""
Phase 0 — Auto-extraction from Discord exports.
Produces:
  config/personnages_draft.yaml  — players and characters
  config/lore_draft.yaml         — relations, places, objects detected in text
Review both drafts, correct, then promote to personnages.yaml / lore.yaml.
Usage: python src/bootstrap.py data/exports/
"""

import json
import re
import sys
from collections import defaultdict
from pathlib import Path
import yaml


# ── patterns ──────────────────────────────────────────────────────────────────

OOC_RE = re.compile(r'\[OOC[^\]]*\]|\(\(.*?\)\)|<OOC>.*?</OOC>', re.IGNORECASE)
ACTION_RE = re.compile(r'\*([^*]+)\*')
NAME_IN_ACTION = re.compile(r'\b([A-ZÀ-Ÿ][a-zà-ÿ]{2,})\b')
FRENCH_FUNCTION_WORDS = {
    "Elle", "Il", "Ils", "Elles", "Lui", "Nous", "Vous", "Eux", "On",
    "Le", "La", "Les", "Un", "Une", "Des", "Du", "Au", "Aux", "Leur", "Leurs",
    "Ce", "Cette", "Ces", "Mon", "Ton", "Son", "Mes", "Tes", "Ses",
    "Mais", "Donc", "Car", "Puis", "Alors", "Enfin", "Ainsi",
}

# Character-character relation: "[A] est le/la [term] de [B]"
RELATION_DECL_RE = re.compile(
    r'\b([A-ZÀ-Ÿ][a-zà-ÿ]{2,})\b[^.]{0,30}?\best(?:ait)?\s+(?:le|la|son|sa|leur)\s+'
    r'(frère|sœur|père|mère|fils|fille|oncle|tante|cousin[e]?|ami[e]?|'
    r'ennemi[e]?|rival[e]?|allié[e]?|amant[e]?|maître|élève)\s+(?:de\s+)?'
    r'\b([A-ZÀ-Ÿ][a-zà-ÿ]{2,})\b',
    re.IGNORECASE,
)

RELATION_TERM_MAP = {
    "frère": "frère_de", "sœur": "sœur_de",
    "père": "parent_de", "mère": "parent_de",
    "fils": "enfant_de", "fille": "enfant_de",
    "oncle": "parent_de", "tante": "parent_de",
    "cousin": "cousin_de", "cousine": "cousin_de",
    "ami": "ami_de", "amie": "ami_de",
    "ennemi": "ennemi_de", "ennemie": "ennemi_de",
    "rival": "rival_de", "rivale": "rival_de",
    "allié": "allié_de", "alliée": "allié_de",
    "amant": "amant_de", "amante": "amant_de",
    "maître": "maître_de", "élève": "élève_de",
}

# Place hints: common French place introductors
PLACE_CONTEXT_RE = re.compile(
    r'(?:entra(?:it)?|sortit|arriva(?:it)?|se trouvait|était|se dirigeait|'
    r'vint|vient|alla|allait|resta|demeura)\s+(?:dans|à|au|aux|chez|vers|devant|derrière)\s+'
    r'(?:la|le|les|une?|l\')?\s*([A-ZÀ-Ÿ][a-zà-ÿ\s]{2,20})',
    re.IGNORECASE,
)

# Object hints: "sortit X", "prit X", "tenait X", "portait X"
OBJECT_CONTEXT_RE = re.compile(
    r'(?:sortit|prit|tenait|portait|déposa|posa|jeta|lança|saisit|brandit)\s+'
    r'(?:un[e]?|son|sa|le|la|l\')\s+([a-zà-ÿ]+(?:\s+[a-zà-ÿ]+)?)',
    re.IGNORECASE,
)


# ── helpers ───────────────────────────────────────────────────────────────────

def load_export(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return data.get("messages", data) if isinstance(data, dict) else data


def extract_channel_name(path: Path) -> str:
    return path.stem.split("_")[0] if "_" in path.stem else path.stem


def is_ooc_only(content: str) -> bool:
    return bool(OOC_RE.fullmatch(content.strip()))


def extract_names_from_actions(content: str) -> list[str]:
    names = []
    seen: set[str] = set()
    for action_match in ACTION_RE.finditer(content):
        for name_match in NAME_IN_ACTION.finditer(action_match.group(1)):
            candidate = name_match.group(1)
            if candidate not in FRENCH_FUNCTION_WORDS and candidate not in seen:
                seen.add(candidate)
                names.append(candidate)
    return names


def extract_relations(content: str, source: str) -> list[dict]:
    relations = []
    for m in RELATION_DECL_RE.finditer(content):
        subject, term, target = m.group(1), m.group(2).lower(), m.group(3)
        rel_type = RELATION_TERM_MAP.get(term)
        if rel_type and subject != target:
            relations.append({
                "from": subject, "from_type": "character",
                "rel": rel_type,
                "to": target, "to_type": "character",
                "source": source, "confidence": "low",
            })
    return relations


def extract_places(content: str) -> list[str]:
    places = []
    for action_match in ACTION_RE.finditer(content):
        for m in PLACE_CONTEXT_RE.finditer(action_match.group(1)):
            candidate = m.group(1).strip().rstrip(".,;")
            if len(candidate) > 3:
                places.append(candidate)
    return places


def extract_objects(content: str) -> list[str]:
    objects = []
    for action_match in ACTION_RE.finditer(content):
        for m in OBJECT_CONTEXT_RE.finditer(action_match.group(1)):
            candidate = m.group(1).strip().rstrip(".,;")
            if len(candidate) > 2:
                objects.append(candidate)
    return objects


# ── main ──────────────────────────────────────────────────────────────────────

def bootstrap(exports_dir: str):
    exports_path = Path(exports_dir)
    json_files = list(exports_path.glob("**/*.json"))

    if not json_files:
        print(f"No JSON files found in {exports_dir}")
        return

    author_characters: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    channels: set[str] = set()
    all_relations: list[dict] = []
    place_counts: dict[str, int] = defaultdict(int)
    object_counts: dict[str, int] = defaultdict(int)

    for filepath in json_files:
        channel = extract_channel_name(filepath)
        channels.add(channel)
        messages = load_export(filepath)

        for msg in messages:
            author = msg.get("author", {}).get("name") or msg.get("author", "unknown")
            content = msg.get("content", "")
            timestamp = msg.get("timestamp", "")

            if not content or len(content) < 3 or is_ooc_only(content):
                continue

            source = f"{filepath.stem}:{timestamp}"

            for name in extract_names_from_actions(content):
                author_characters[author][name] += 1

            all_relations.extend(extract_relations(content, source))

            for place in extract_places(content):
                place_counts[place] += 1

            for obj in extract_objects(content):
                object_counts[obj] += 1

    # ── personnages_draft.yaml ─────────────────────────────────────────────
    players = {}
    for author, characters in sorted(author_characters.items()):
        sorted_chars = sorted(characters.items(), key=lambda x: x[1], reverse=True)
        detected = [
            {"name": name, "occurrences": count, "confidence": "high" if count >= 5 else "low"}
            for name, count in sorted_chars
        ]
        players[author] = {"detected_characters": detected}

    personnages_draft = {
        "players": players,
        "detected_channels": sorted(channels),
        "note": "Auto-generated. Review and rename to personnages.yaml",
    }

    out_personnages = Path("config/personnages_draft.yaml")
    out_personnages.parent.mkdir(parents=True, exist_ok=True)
    with open(out_personnages, "w", encoding="utf-8") as f:
        yaml.dump(personnages_draft, f, allow_unicode=True, sort_keys=False)

    # ── lore_draft.yaml ───────────────────────────────────────────────────
    # Deduplicate relations by (from, rel, to)
    seen_relations: set[tuple] = set()
    unique_relations = []
    for r in all_relations:
        key = (r["from"], r["rel"], r["to"])
        if key not in seen_relations:
            seen_relations.add(key)
            unique_relations.append(r)

    lore_draft = {
        "note": (
            "Auto-generated by bootstrap.py. Confidence is always 'low' — review before "
            "promoting to lore.yaml. Add missing entities (places, objects, cultures, "
            "intentions, rules) manually."
        ),
        "detected_relations": unique_relations,
        "detected_places": [
            {"name": name, "occurrences": count}
            for name, count in sorted(place_counts.items(), key=lambda x: x[1], reverse=True)
            if count >= 2
        ],
        "detected_objects": [
            {"name": name, "occurrences": count}
            for name, count in sorted(object_counts.items(), key=lambda x: x[1], reverse=True)
            if count >= 2
        ],
    }

    out_lore = Path("config/lore_draft.yaml")
    with open(out_lore, "w", encoding="utf-8") as f:
        yaml.dump(lore_draft, f, allow_unicode=True, sort_keys=False)

    print(f"Personnages draft : {out_personnages}  ({len(players)} players)")
    print(f"Lore draft        : {out_lore}")
    print(f"  {len(unique_relations)} relations detected")
    print(f"  {len(lore_draft['detected_places'])} recurring places")
    print(f"  {len(lore_draft['detected_objects'])} recurring objects")
    print("→ Review drafts, then promote to personnages.yaml / lore.yaml")


if __name__ == "__main__":
    exports_dir = sys.argv[1] if len(sys.argv) > 1 else "data/exports"
    with_llm = "--with-llm" in sys.argv
    bootstrap(exports_dir)

    if with_llm:
        print("\nStarting LLM lore extraction pass…")
        sys.path.insert(0, str(Path(__file__).parent))
        from lore_extractor import extract_from_file
        from preprocessing import load_config, build_alias_map, process_export, group_into_scenes, scene_to_text
        from lore import load_lore

        config = load_config()
        lore = load_lore()
        alias_map = build_alias_map(config)
        alias_map.update(lore.character_aliases())

        for filepath in Path(exports_dir).glob("**/*.json"):
            messages = process_export(filepath, config, lore=lore)
            scenes = group_into_scenes(messages)
            pairs = [
                (text, f"{filepath.stem}:{meta.get('start', '')}")
                for scene in scenes
                for text, meta in [scene_to_text(scene, alias_map)]
            ]
            extract_from_file(filepath, pairs, verbose=True)
