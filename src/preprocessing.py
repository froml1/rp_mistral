"""
Cleaning and structuring of Discord exports for the RAG pipeline.
Produces enriched documents with RP metadata.
"""

import csv
import json
import re
from pathlib import Path
from dataclasses import dataclass, field
import yaml

from lore import LoreGraph, load_lore
from analyzer import analyze_messages


# Minimal Discord artifact cleanup — not content analysis
_MENTION_RE = re.compile(r'<@!?\d+>')
_CUSTOM_EMOJI_RE = re.compile(r'<:\w+:\d+>')
_URL_RE = re.compile(r'https?://\S+')


@dataclass
class MessageRP:
    author: str
    timestamp: str
    channel: str
    raw_content: str
    clean_content: str = ""
    characters: list[str] = field(default_factory=list)
    referenced: list[str] = field(default_factory=list)
    speaker_segments: list[dict] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    is_ooc: bool = False
    arc: str = ""
    metadata: dict = field(default_factory=dict)


def load_messages(filepath: Path) -> list[dict]:
    """Loads a JSON or CSV Discord export and returns normalized message dicts."""
    if filepath.suffix.lower() == ".csv":
        messages = []
        with open(filepath, encoding="utf-8", newline="") as f:
            sample = f.read(4096)
            f.seek(0)
            dialect = csv.Sniffer().sniff(sample, delimiters=",;\t|")
            for row in csv.DictReader(f, dialect=dialect):
                author = (row.get("author") or row.get("Author") or "").strip()
                content = (row.get("content") or row.get("Content") or "").strip()
                timestamp = (row.get("timestamp") or row.get("Timestamp") or "").strip()
                if content:
                    messages.append({
                        "author": {"name": author},
                        "content": content,
                        "timestamp": timestamp,
                    })
        return messages
    else:
        with open(filepath, encoding="utf-8") as f:
            data = json.load(f)
        return data.get("messages", data) if isinstance(data, dict) else data


def load_config(config_path: str = "config/personnages.yaml") -> dict:
    path = Path(config_path)
    if not path.exists():
        return {"joueurs": {}, "aliases": {}, "arcs": {}}
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def build_alias_map(config: dict) -> dict[str, str]:
    return {alias: canonical for alias, canonical in config.get("aliases", {}).items()}


def _clean_discord(content: str) -> str:
    """Strips Discord-specific formatting artifacts that carry no narrative meaning."""
    content = _MENTION_RE.sub("", content)
    content = _CUSTOM_EMOJI_RE.sub("", content)
    content = _URL_RE.sub("", content)
    return re.sub(r' {2,}', ' ', content).strip()


def process_export(
    filepath: Path, config: dict, lore: LoreGraph | None = None
) -> list[MessageRP]:
    raw_messages = load_messages(filepath)
    channel = filepath.stem
    arc = config.get("arcs", {}).get(channel, channel)

    alias_map = build_alias_map(config)
    if lore:
        alias_map.update(lore.character_aliases())

    analyses = analyze_messages(raw_messages, alias_map=alias_map)

    results = []
    for msg, analysis in zip(raw_messages, analyses):
        author_field = msg.get("author", {})
        author = (
            author_field.get("name") if isinstance(author_field, dict) else str(author_field)
        ) or "unknown"
        raw_content = msg.get("content", "").strip()
        timestamp = msg.get("timestamp", "")

        if not raw_content:
            continue

        # Skip pure HRP messages (not OOC — OOC stays for context tracking)
        if not analysis["is_rp"] and not analysis["is_ooc"]:
            continue

        clean_content = _clean_discord(raw_content)

        results.append(MessageRP(
            author=author,
            timestamp=timestamp,
            channel=channel,
            raw_content=raw_content,
            clean_content=clean_content,
            characters=analysis["characters"],
            referenced=analysis["referenced"],
            speaker_segments=analysis["speaker_segments"],
            tags=analysis["tags"],
            is_ooc=analysis["is_ooc"],
            arc=arc,
            metadata={
                "author": author,
                "characters": analysis["characters"],
                "referenced": analysis["referenced"],
                "tags": analysis["tags"],
                "channel": channel,
                "arc": arc,
                "timestamp": timestamp,
                "ooc": analysis["is_ooc"],
            },
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


def scene_to_text(
    scene: list[MessageRP],
    alias_map: dict[str, str] | None = None,
    scene_tags: list[str] | None = None,
) -> tuple[str, dict]:
    lines = []

    for msg in scene:
        if msg.is_ooc:
            continue

        segments = msg.speaker_segments
        if segments:
            for seg in segments:
                char = seg.get("character")
                text = seg.get("text", "").strip()
                if not text:
                    continue
                tag = f"[{char}]" if char else "[-]"
                lines.append(f"{tag} {text}")
        else:
            # Fallback: prefix whole message with character list
            chars = msg.characters
            if not chars:
                prefix = f"[{msg.author}:descriptif]"
            elif len(chars) == 1:
                prefix = f"[{chars[0]}]"
            else:
                prefix = f"[{', '.join(chars)}]"
            lines.append(f"{prefix} {msg.clean_content}")

    text = "\n".join(lines)
    all_characters = list({c for m in scene for c in m.characters})
    all_referenced = list({c for m in scene for c in m.referenced} - set(all_characters))
    msg_tags = list({t for m in scene for t in m.tags if t != "ooc"})

    metadata = {
        "arc":          scene[0].arc,
        "channel":      scene[0].channel,
        "start":        scene[0].timestamp,
        "end":          scene[-1].timestamp,
        "authors":      ",".join(sorted({m.author for m in scene})),
        "characters":   ",".join(all_characters),
        "referenced":   ",".join(all_referenced),
        "message_tags": ",".join(msg_tags),
        "scene_tags":   ",".join(scene_tags or []),
    }
    return text, metadata
