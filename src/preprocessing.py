"""
Cleaning and structuring of Discord exports for the RAG pipeline.
Produces enriched documents with RP metadata.
"""

import json
import re
from pathlib import Path
from dataclasses import dataclass, field
import yaml

from lore import LoreGraph, load_lore, RELATION_TERM_TO_TYPE
from tagger import tag_message


OOC_RE = re.compile(r'\[OOC[^\]]*\]|\(\(.*?\)\)', re.IGNORECASE | re.DOTALL)
ACTION_RE = re.compile(r'\*([^*]+)\*')
MENTION_RE = re.compile(r'<@!?\d+>')
EMOJI_RE = re.compile(r'<:\w+:\d+>')
URL_RE = re.compile(r'https?://\S+')

# `-` speaker switch: follows a sentence/description closing marker (* . ! ? " »)
SPEAKER_SPLIT_RE = re.compile(r'(?<=[*\.!?\"»])\s*-(?=\s)')

# Capitalized proper name (3+ chars), not a French function word
NAME_RE = re.compile(r'\b([A-ZÀ-Ÿ][a-zà-ÿ]{2,})\b')
FRENCH_FUNCTION_WORDS = {
    "Elle", "Il", "Ils", "Elles", "Lui", "Nous", "Vous", "Eux", "On",
    "Le", "La", "Les", "Un", "Une", "Des", "Du", "Au", "Aux", "Leur", "Leurs",
    "Ce", "Cette", "Ces", "Mon", "Ton", "Son", "Mes", "Tes", "Ses",
    "Mais", "Donc", "Car", "Puis", "Alors", "Enfin", "Ainsi",
}

# Coordinate conjunctions that extend a subject ("Antoine et Gaulthier")
COORD_CONJUNCTIONS = {"et", "&"}


@dataclass
class MessageRP:
    author: str
    timestamp: str
    channel: str
    raw_content: str
    clean_content: str = ""
    # Characters actively voiced or acting in this message
    characters: list[str] = field(default_factory=list)
    # Characters evoked/mentioned but not present in the scene
    referenced: list[str] = field(default_factory=list)
    # Structural/content tags (heuristic): action, dialogue, descriptif, pensée, nsfw_hint, ooc
    tags: list[str] = field(default_factory=list)
    is_ooc: bool = False
    arc: str = ""
    metadata: dict = field(default_factory=dict)


# Matches "son frère", "sa sœur", "ton père", etc. + optional following name
RELATION_TERM_RE = re.compile(
    r'\b(?:son|sa|ton|ta|votre|leur)\s+(' + '|'.join(RELATION_TERM_TO_TYPE.keys()) + r')\b',
    re.IGNORECASE,
)


def load_config(config_path: str = "config/personnages.yaml") -> dict:
    path = Path(config_path)
    if not path.exists():
        return {"joueurs": {}, "aliases": {}, "arcs": {}}
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def build_alias_map(config: dict) -> dict[str, str]:
    """Returns a flat {alias: canonical_name} dict from the config."""
    return {alias: canonical for alias, canonical in config.get("aliases", {}).items()}


def _is_candidate_name(word: str) -> bool:
    clean = word.strip('.,;:!?*"\'()')
    return bool(NAME_RE.fullmatch(clean)) and clean not in FRENCH_FUNCTION_WORDS


def classify_action_names(action_text: str) -> tuple[list[str], list[str]]:
    """
    Splits names in an action block into (subjects, complements).

    Subject heuristic: a capitalized name at position 0, or at position 2 after
    a coordinate conjunction ('et', '&'), is treated as a grammatical subject.
    All other names are complements (may be physically present or merely evoked).
    """
    words = action_text.split()
    subjects: list[str] = []
    complements: list[str] = []
    seen: set[str] = set()

    for i, word in enumerate(words):
        clean = word.strip('.,;:!?*"\'()')
        if not _is_candidate_name(clean) or clean in seen:
            continue
        seen.add(clean)
        is_subject = i == 0 or (i == 2 and words[1].lower() in COORD_CONJUNCTIONS)
        if is_subject:
            subjects.append(clean)
        else:
            complements.append(clean)

    return subjects, complements


def extract_classified_characters(
    content: str,
    alias_map: dict[str, str],
    lore: LoreGraph | None = None,
    context_characters: list[str] | None = None,
) -> tuple[list[str], list[str]]:
    """
    Returns (active_characters, referenced_characters) for the full message content.

    Active = subjects of at least one action block.
    Referenced = names that only appear as complements, plus characters resolved
                 from relational terms ("son frère") via the lore graph.
    Aliases are resolved to their canonical names.
    """
    all_subjects: set[str] = set()
    all_complements: set[str] = set()

    for action_match in ACTION_RE.finditer(content):
        subjects, complements = classify_action_names(action_match.group(1))
        for name in subjects:
            resolved = alias_map.get(name, name)
            all_subjects.add(resolved)
        for name in complements:
            resolved = alias_map.get(name, name)
            all_complements.add(resolved)

    # Resolve relational terms ("son frère", "sa sœur") via lore graph
    if lore and context_characters:
        for m in RELATION_TERM_RE.finditer(content):
            term = m.group(1).lower()
            for ctx_char in context_characters:
                resolved = lore.resolve_relation_term(term, ctx_char)
                if resolved and resolved not in all_subjects:
                    all_complements.add(resolved)
                    break

    referenced = sorted(all_complements - all_subjects)
    active = sorted(all_subjects)
    return active, referenced


def resolve_characters(
    author: str,
    content: str,
    config: dict,
    alias_map: dict[str, str],
    author_chars: dict[str, list[str]],
    lore: LoreGraph | None = None,
) -> tuple[list[str], list[str]]:
    """
    Returns (active_characters, referenced_characters) for this message.

    Resolution order for active characters:
    1. Subjects of *action* blocks (with alias expansion)
    2. Config override
    3. No action blocks → inherit last known characters (dialogue continuation)
    4. Action blocks with no subject names → pure scene description → empty
    """
    last_known = author_chars.get(author, [])
    active, referenced = extract_classified_characters(
        content, alias_map, lore=lore, context_characters=last_known
    )

    if active:
        author_chars[author] = active
        return active, referenced

    # Config fallback: look for known character names mentioned anywhere in the message
    config_chars = config.get("joueurs", {}).get(author, {}).get("personnages", [])
    if config_chars:
        matched = [c for c in config_chars if c.lower() in content.lower()]
        if matched:
            author_chars[author] = matched
            return matched, referenced

    has_action_block = bool(ACTION_RE.search(content))

    if not has_action_block:
        # Pure dialogue or continuation — inherit last known active characters
        return list(last_known), referenced

    # Action blocks exist but yielded no subject names = scene description
    return [], referenced


def split_speakers(content: str) -> list[str]:
    """Splits a message on speaker-switch markers (' - ' after closing punctuation)."""
    segments = SPEAKER_SPLIT_RE.split(content)
    return [s.strip() for s in segments if s.strip()]


def clean_message(content: str) -> tuple[str, bool]:
    is_ooc = bool(OOC_RE.search(content))
    clean = OOC_RE.sub("", content)
    clean = MENTION_RE.sub("", clean)
    clean = EMOJI_RE.sub("", clean)
    clean = URL_RE.sub("", clean)
    clean = re.sub(r'\s{2,}', ' ', clean).strip()
    return clean, is_ooc


def process_export(
    filepath: Path, config: dict, lore: LoreGraph | None = None
) -> list[MessageRP]:
    with open(filepath, encoding="utf-8") as f:
        data = json.load(f)

    raw_messages = data.get("messages", data) if isinstance(data, dict) else data
    channel = filepath.stem
    arc = config.get("arcs", {}).get(channel, channel)
    alias_map = build_alias_map(config)
    if lore:
        alias_map.update(lore.character_aliases())

    results = []
    author_chars: dict[str, list[str]] = {}

    for msg in raw_messages:
        author = msg.get("author", {}).get("name") or msg.get("author", "unknown")
        raw_content = msg.get("content", "").strip()
        timestamp = msg.get("timestamp", "")

        if not raw_content:
            continue

        clean_content, is_ooc = clean_message(raw_content)
        characters, referenced = resolve_characters(
            author, raw_content, config, alias_map, author_chars, lore=lore
        )
        tags = tag_message(raw_content, is_ooc)

        results.append(MessageRP(
            author=author,
            timestamp=timestamp,
            channel=channel,
            raw_content=raw_content,
            clean_content=clean_content,
            characters=characters,
            referenced=referenced,
            tags=tags,
            is_ooc=is_ooc,
            arc=arc,
            metadata={
                "author": author,
                "characters": characters,
                "referenced": referenced,
                "tags": tags,
                "channel": channel,
                "arc": arc,
                "timestamp": timestamp,
                "ooc": is_ooc,
            }
        ))

    return results


def group_into_scenes(messages: list[MessageRP], max_size: int = 30) -> list[list[MessageRP]]:
    scenes, current = [], []

    for msg in messages:
        current.append(msg)
        if len(current) >= max_size:
            scenes.append(current)
            current = []

    if current:
        scenes.append(current)

    return scenes


def _attribute_segment(segment: str, known_characters: list[str], alias_map: dict[str, str]) -> str | None:
    """Returns a character name if the segment has an unambiguous action-block subject."""
    subjects, _ = extract_classified_characters(segment, alias_map)
    matched = [s for s in subjects if s in known_characters]
    return matched[0] if len(matched) == 1 else None


def scene_to_text(
    scene: list[MessageRP],
    alias_map: dict[str, str] | None = None,
    scene_tags: list[str] | None = None,
) -> tuple[str, dict]:
    if alias_map is None:
        alias_map = {}
    lines = []

    for msg in scene:
        if msg.is_ooc:
            continue

        chars = msg.characters
        if not chars:
            prefix = f"[{msg.author}:descriptif]"
        elif len(chars) == 1:
            prefix = f"[{chars[0]}]"
        else:
            prefix = f"[{', '.join(chars)}]"

        segments = split_speakers(msg.clean_content)
        if len(segments) <= 1:
            lines.append(f"{prefix} {msg.clean_content}")
        else:
            lines.append(f"{prefix} {segments[0]}")
            for seg in segments[1:]:
                attributed = _attribute_segment(seg, chars, alias_map)
                tag = f"[{attributed}]" if attributed else "[-]"
                lines.append(f"{tag} {seg}")

    text = "\n".join(lines)
    all_characters = list({c for m in scene for c in m.characters})
    all_referenced = list({c for m in scene for c in m.referenced} - set(all_characters))

    # Aggregate message-level structural tags (union, excluding ooc)
    msg_tags = list({t for m in scene for t in m.tags if t != "ooc"})

    metadata = {
        "arc": scene[0].arc,
        "channel": scene[0].channel,
        "start": scene[0].timestamp,
        "end": scene[-1].timestamp,
        "authors": ",".join(sorted({m.author for m in scene})),
        "characters": ",".join(all_characters),
        "referenced": ",".join(all_referenced),
        "message_tags": ",".join(msg_tags),
        "scene_tags": ",".join(scene_tags or []),
    }
    return text, metadata
