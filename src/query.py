"""
Query engine: answers natural language questions from pipeline lore + analysis outputs.

Usage:
  python src/query.py "What is the relationship between Lena and the Iron Covenant?"
  python src/query.py  (interactive mode)
"""

import json
import re
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent))

from llm import call_llm

DATA_DIR      = Path("data")
LORE_DIR      = DATA_DIR / "lore"
ANALYSIS_DIR  = DATA_DIR / "analysis"
HOW_CTX_FILE  = LORE_DIR / "how_context.yaml"

MAX_SCENES    = 6
MAX_CTX_CHARS = 12000

_EXTRACT_PROMPT = """\
Given this question, list the entity names (characters, places, concepts, factions) it references.
Return ONLY a JSON array of lowercase strings.

Question: {question}

JSON: []"""

_ANSWER_PROMPT = """\
You are an expert analyst of a narrative roleplay universe.
Answer the question using ONLY the provided context. Be precise and cite characters/scenes when relevant.
If the context does not contain enough information, say so clearly.

CONTEXT:
{context}

QUESTION: {question}

ANSWER:"""


# ── Lore loading ──────────────────────────────────────────────────────────────

def _load_yaml_dir(path: Path) -> list[dict]:
    if not path.exists():
        return []
    results = []
    for f in path.glob("*.yaml"):
        try:
            d = yaml.safe_load(f.read_text(encoding="utf-8")) or {}
            d["_file"] = f.stem
            results.append(d)
        except Exception:
            pass
    return results


def load_all_lore() -> dict:
    return {
        "characters": _load_yaml_dir(LORE_DIR / "characters"),
        "places":     _load_yaml_dir(LORE_DIR / "places"),
        "concepts":   _load_yaml_dir(LORE_DIR / "concepts"),
        "how_context": (
            yaml.safe_load(HOW_CTX_FILE.read_text(encoding="utf-8"))
            if HOW_CTX_FILE.exists() else {}
        ),
    }


# ── Entity matching ───────────────────────────────────────────────────────────

def _names_for_entity(entity: dict) -> list[str]:
    names = []
    if entity.get("name"):
        names.append(entity["name"].lower())
    for app in entity.get("appellations") or []:
        names.append(str(app).lower())
    return names


def _entity_matches(entity: dict, terms: list[str]) -> bool:
    entity_names = _names_for_entity(entity)
    for term in terms:
        term = term.lower().strip()
        if not term:
            continue
        for name in entity_names:
            if term in name or name in term:
                return True
    return False


def find_relevant_entities(lore: dict, terms: list[str]) -> dict:
    matched = {"characters": [], "places": [], "concepts": []}
    for cat in ("characters", "places", "concepts"):
        for entity in lore[cat]:
            if _entity_matches(entity, terms):
                matched[cat].append(entity)
    return matched


# ── Scene retrieval ───────────────────────────────────────────────────────────

def _collect_scene_ids(matched_entities: dict) -> list[str]:
    scene_ids: set[str] = set()
    for cat in ("characters", "places", "concepts"):
        for entity in matched_entities[cat]:
            for sid in entity.get("appearances") or []:
                scene_ids.add(str(sid))
    return sorted(scene_ids)


def load_scene_context(scene_id: str) -> str | None:
    scene_dir = ANALYSIS_DIR / scene_id
    if not scene_dir.exists():
        return None

    parts = []

    what_path = scene_dir / "what.json"
    if what_path.exists():
        what = json.loads(what_path.read_text(encoding="utf-8"))
        if what.get("summary"):
            parts.append(f"[{scene_id}] SUMMARY: {what['summary']}")
        for ev in (what.get("events") or [])[:8]:
            desc = ev.get("description", "")
            chars = ", ".join(ev.get("characters") or [])
            if desc:
                parts.append(f"  • {desc}" + (f" ({chars})" if chars else ""))

    how_path = scene_dir / "how.json"
    if how_path.exists():
        how = json.loads(how_path.read_text(encoding="utf-8"))
        if how.get("context_synthesis"):
            parts.append(f"  [narrative context] {how['context_synthesis']}")

    return "\n".join(parts) if parts else None


# ── Context builder ───────────────────────────────────────────────────────────

def _fmt_entity(entity: dict, cat: str) -> str:
    lines = [f"[{cat.upper()}] {entity.get('name', '?')}"]
    if entity.get("appellations"):
        lines.append(f"  also known as: {', '.join(entity['appellations'])}")
    for field in ("description", "description_physical", "description_psychological",
                  "job", "significance", "type"):
        val = entity.get(field)
        if val:
            lines.append(f"  {field}: {val}")
    for field in ("beliefs", "attributes", "related_characters", "main_locations"):
        items = entity.get(field)
        if items:
            lines.append(f"  {field}: {', '.join(str(x) for x in items)}")
    return "\n".join(lines)


def build_context(matched_entities: dict, lore: dict, question: str) -> str:
    sections = []

    # Entity profiles
    for cat in ("characters", "places", "concepts"):
        for entity in matched_entities[cat]:
            sections.append(_fmt_entity(entity, cat))

    # If no specific entities matched, include a short how_context overview
    if not any(matched_entities[c] for c in ("characters", "places", "concepts")):
        how_ctx = lore.get("how_context") or {}
        if how_ctx:
            overview = "\n".join(
                f"- {sid}: {synth}" for sid, synth in list(how_ctx.items())[-10:]
            )
            sections.append(f"[NARRATIVE CONTEXT - recent scenes]\n{overview}")

    # Scene details for matched entities
    scene_ids = _collect_scene_ids(matched_entities)
    scene_texts = []
    for sid in scene_ids[:MAX_SCENES]:
        ctx = load_scene_context(sid)
        if ctx:
            scene_texts.append(ctx)

    if scene_texts:
        sections.append("[RELEVANT SCENES]\n" + "\n\n".join(scene_texts))

    context = "\n\n".join(sections)
    if len(context) > MAX_CTX_CHARS:
        context = context[:MAX_CTX_CHARS] + "\n[... context truncated ...]"
    return context


# ── Entity extraction from question ──────────────────────────────────────────

def extract_terms_from_question(question: str, lore: dict) -> list[str]:
    # Fast path: directly scan question words against all known names/appellations
    question_lower = question.lower()
    matched_terms = []

    for cat in ("characters", "places", "concepts"):
        for entity in lore[cat]:
            for name in _names_for_entity(entity):
                if len(name) >= 3 and name in question_lower:
                    matched_terms.append(name)

    if matched_terms:
        return matched_terms

    # Fallback: use LLM to extract entity names
    raw = call_llm(
        _EXTRACT_PROMPT.format(question=question),
        fmt="json",
        num_predict=128,
    )
    try:
        terms = json.loads(raw)
        if isinstance(terms, list):
            return [str(t) for t in terms]
    except Exception:
        pass

    # Last resort: capitalized words from the question
    return [w.lower() for w in re.findall(r'\b[A-ZÀ-Ÿ][a-zà-ÿ]{2,}\b', question)]


# ── Public API ────────────────────────────────────────────────────────────────

def answer(question: str, verbose: bool = False) -> str:
    lore = load_all_lore()

    terms = extract_terms_from_question(question, lore)
    if verbose:
        print(f"  terms: {terms}")

    matched = find_relevant_entities(lore, terms)
    if verbose:
        for cat, entities in matched.items():
            if entities:
                print(f"  {cat}: {[e.get('name') for e in entities]}")

    context = build_context(matched, lore, question)
    if verbose:
        print(f"  context: {len(context)} chars")

    if not context.strip():
        return "No relevant information found in the lore. Run the pipeline first."

    return call_llm(
        _ANSWER_PROMPT.format(context=context, question=question),
        num_predict=-1,
        num_ctx=16384,
    )


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if len(sys.argv) > 1:
        q = " ".join(sys.argv[1:])
        print(answer(q, verbose=True))
    else:
        print("RP_IA Query — type your question (empty line to quit)\n")
        while True:
            try:
                q = input("? ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not q:
                break
            print()
            print(answer(q, verbose=False))
            print()
