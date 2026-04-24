"""
Summary card generation for lore entities.

Each YAML gets a compact `_summary` field (~150 chars, rule-based, no LLM).
Used as the primary injection surface in prompts and as the ChromaDB document.
"""

from pathlib import Path
import yaml


def make_char_summary(data: dict) -> str:
    parts = []
    name = data.get("name", "?")
    parts.append(name)
    if data.get("job"):
        parts.append(data["job"])
    if data.get("author"):
        parts.append(data["author"])
    psych = (data.get("description_psychological") or "").strip()
    if psych:
        parts.append(psych[:120])
    n = len(data.get("appearances") or [])
    if n:
        parts.append(f"{n} scene(s)")
    locs = (data.get("main_locations") or [])[:2]
    if locs:
        parts.append("locs: " + ", ".join(locs))
    rels = (data.get("relations") or [])[:2]
    if rels:
        rel_strs = [
            f"{r['character']} : {r.get('relation', '')}" if isinstance(r, dict) else str(r)
            for r in rels
        ]
        parts.append("rels: " + ", ".join(rel_strs))
    ep = data.get("emotional_polarity") or {}
    emo = (ep.get("dominant_emotions") or [])[:2]
    if emo:
        parts.append("emo: " + ", ".join(emo))
    return " | ".join(parts)[:200]


def make_place_summary(data: dict) -> str:
    parts = []
    name = data.get("name", "?")
    parts.append(name)
    desc = (data.get("description") or "").strip()
    if desc:
        parts.append(desc[:120])
    attrs = (data.get("attributes") or [])[:3]
    if attrs:
        parts.append(", ".join(attrs))
    n = len(data.get("appearances") or [])
    if n:
        parts.append(f"{n} scene(s)")
    return " | ".join(parts)[:200]


def make_concept_summary(data: dict) -> str:
    parts = []
    name = data.get("name", "?")
    ctype = data.get("type", "")
    parts.append(f"{name} ({ctype})" if ctype else name)
    sig = (data.get("significance") or data.get("description") or "").strip()
    if sig:
        parts.append(sig[:120])
    chars = (data.get("related_characters") or [])[:3]
    if chars:
        parts.append("chars: " + ", ".join(chars))
    return " | ".join(parts)[:200]


_MAKERS = {
    "characters": make_char_summary,
    "places":     make_place_summary,
    "concepts":   make_concept_summary,
}


def update_summary(path: Path, entity_type: str) -> str:
    """Read a lore YAML, compute its _summary, write it back. Returns the summary."""
    maker = _MAKERS.get(entity_type)
    if not maker or not path.exists():
        return ""
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        summary = maker(data)
        data["_summary"] = summary
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(data, f, allow_unicode=True, sort_keys=False)
        return summary
    except Exception:
        return ""


def summaries_for_dir(lore_dir: Path, entity_type: str) -> dict[str, str]:
    """Return {name: summary} for all YAMLs in a lore subdirectory."""
    d = lore_dir / entity_type
    if not d.exists():
        return {}
    result = {}
    for f in d.glob("*.yaml"):
        try:
            data = yaml.safe_load(f.read_text(encoding="utf-8")) or {}
            name = data.get("name")
            if not name:
                continue
            s = data.get("_summary") or _MAKERS[entity_type](data)
            result[name] = s
        except Exception:
            pass
    return result
