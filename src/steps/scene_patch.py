"""
Shared utility: write analysis enrichments back into a scene file.

Each analyze step can call write_enrichment() to store what it detected
(speaker attribution, inconsistencies, etc.) under _enrichments.<step>.
This never modifies messages[].content — it only adds/replaces top-level
_enrichments.<step> in the scene JSON.
"""

import json
from pathlib import Path


def read_enrichments(scene_file: Path) -> dict:
    """Return the full _enrichments dict from a scene file, or {} on failure."""
    try:
        data = json.loads(scene_file.read_text(encoding="utf-8"))
        return data.get("_enrichments", {})
    except Exception:
        return {}


def write_enrichment(scene_file: Path, step: str, data: dict) -> None:
    """
    Merge data into _enrichments[step] in the scene file.
    Only called when data is non-empty (caller responsibility).
    """
    try:
        scene = json.loads(scene_file.read_text(encoding="utf-8"))
    except Exception:
        return
    scene.setdefault("_enrichments", {})[step] = data
    scene_file.write_text(json.dumps(scene, ensure_ascii=False, indent=2), encoding="utf-8")


CHUNK_SIZE    = 40   # messages per analysis chunk
CHUNK_OVERLAP = 5    # messages shared between adjacent chunks


def chunk_messages(messages: list[dict]) -> list[list[dict]]:
    """Split messages into overlapping chunks for long-scene analysis."""
    if len(messages) <= CHUNK_SIZE:
        return [messages]
    chunks: list[list[dict]] = []
    start = 0
    while start < len(messages):
        end = min(start + CHUNK_SIZE, len(messages))
        chunks.append(messages[start:end])
        if end == len(messages):
            break
        start = end - CHUNK_OVERLAP
    return chunks


def format_inconsistencies(enrichments: dict) -> str:
    """Format all known inconsistencies from prior steps for injection into a prompt."""
    lines = []
    for step, data in enrichments.items():
        for inc in (data.get("inconsistencies") or []):
            idx  = inc.get("message_idx")
            loc  = f" [msg {idx}]" if idx is not None else ""
            lines.append(f"- [{step}]{loc} {inc.get('type', '')}: {inc.get('description', '')}")
    return "\n".join(lines) or "none"
