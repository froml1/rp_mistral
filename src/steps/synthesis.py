"""
Step 4b — Synthesis: build a rich narrative synthesis from sweep data.
Output frozen in data/lore/lore_how.yaml.

Two-phase hierarchical approach (Mistral context-window safe):
  Phase A — done by lore_sweep: 1 narrative line per scene
  Phase B — here: batch compression (N scenes → summary) then global synthesis

lore_how.yaml contains:
  overall_context  — 2-3 sentence global description of the story
  narrative_axes   — main story threads with characters and tension
  characters       — role, arc, key relations per character

Replaces general_how.yaml as the narrative context injected into step 6.
Incremental: already-computed batch summaries are reused on re-run.
"""

import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from llm import call_llm_json
from steps.lore_sweep import load_sweep

LORE_HOW_FILE = "lore_how.yaml"
_BATCH_SIZE = 40

_BATCH_PROMPT = """\
Summarize the events and character dynamics from these RP scene descriptions in 3-4 sentences.
Focus on: what happened, who was involved, key tensions, revelations, or turning points.

Scenes:
{narratives}

JSON: {{"summary": ""}}"""

_FINAL_PROMPT = """\
You are synthesizing a complete roleplay story. Build a rich narrative synthesis from the material below.

STORY SUMMARY (chronological batches):
{batch_summaries}

KNOWN CHARACTERS:
{characters}

KNOWN PLACES:
{places}

Produce:

1. CHARACTERS — for each main character (those appearing most): name, role in the story, narrative arc (how they evolve), key_relations (list of "other_character : relation_type").
2. NARRATIVE AXES — 2 to 6 main story threads. Each: name, summary (2 sentences), characters involved, core tension.
3. OVERALL CONTEXT — 2-3 sentences: setting, mood, central conflict of this story.

JSON:
{{
  "characters": [{{"name": "", "role": "", "arc": "", "key_relations": []}}],
  "narrative_axes": [{{"name": "", "summary": "", "characters": [], "tension": ""}}],
  "overall_context": ""
}}"""


def run_synthesis(lore_dir: Path) -> Path:
    """
    Build narrative synthesis from sweep narratives.
    Saves to data/lore/lore_how.yaml. Incremental: skips if already done.
    """
    lore_how_path = lore_dir / LORE_HOW_FILE
    sweep = load_sweep(lore_dir)
    narratives: dict[str, str] = sweep.get("narratives") or {}

    if not narratives:
        print("  [lore_how] no sweep narratives found — run step 4 (sweep) first")
        return lore_how_path

    sorted_scenes = sorted(narratives.items())

    # Load existing file to reuse cached batch summaries
    existing: dict = {}
    if lore_how_path.exists():
        existing = yaml.safe_load(lore_how_path.read_text(encoding="utf-8")) or {}

    if existing.get("_done"):
        n_axes = len(existing.get("narrative_axes") or [])
        n_chars = len(existing.get("characters") or [])
        print(f"  [lore_how] already done — {n_axes} axes, {n_chars} characters")
        return lore_how_path

    cached_batches: list[str] = existing.get("_batch_summaries") or []
    batches = [sorted_scenes[i:i + _BATCH_SIZE] for i in range(0, len(sorted_scenes), _BATCH_SIZE)]
    total = len(batches)

    # Phase B-1: compress each batch (skip already cached)
    batch_summaries: list[str] = list(cached_batches)
    for i, batch in enumerate(batches[len(cached_batches):], start=len(cached_batches)):
        lines = "\n".join(f"- {sid}: {narr}" for sid, narr in batch)
        result = call_llm_json(_BATCH_PROMPT.format(narratives=lines), num_predict=400, num_ctx=4096)
        summary = (result.get("summary") or "").strip()
        batch_summaries.append(summary or "(no summary)")
        print(f"  [lore_how] batch {i + 1}/{total} compressed")

        # Save progress after each batch
        existing["_batch_summaries"] = batch_summaries
        lore_dir.mkdir(parents=True, exist_ok=True)
        lore_how_path.write_text(yaml.dump(existing, allow_unicode=True, sort_keys=False), encoding="utf-8")

    # Phase B-2: final synthesis
    chars_catalog  = sweep.get("characters") or {}
    places_catalog = sweep.get("places") or {}
    chars_lines  = "\n".join(f"- {n}: {d.get('description', '')}" for n, d in list(chars_catalog.items())[:25])
    places_lines = "\n".join(f"- {n}: {d.get('description', '')}" for n, d in list(places_catalog.items())[:12])
    all_summaries = "\n\n".join(f"[batch {i+1}] {s}" for i, s in enumerate(batch_summaries))

    print(f"  [lore_how] final synthesis call ({len(batch_summaries)} batches)…")
    final = call_llm_json(
        _FINAL_PROMPT.format(
            batch_summaries=all_summaries,
            characters=chars_lines or "none",
            places=places_lines or "none",
        ),
        num_predict=2048,
        num_ctx=8192,
    )

    lore_how = {
        "_done": True,
        "_batch_summaries": batch_summaries,
        "overall_context": str(final.get("overall_context") or ""),
        "characters":      [c for c in (final.get("characters") or []) if isinstance(c, dict) and c.get("name")],
        "narrative_axes":  [a for a in (final.get("narrative_axes") or []) if isinstance(a, dict) and a.get("name")],
    }

    lore_how_path.write_text(yaml.dump(lore_how, allow_unicode=True, sort_keys=False), encoding="utf-8")
    print(
        f"  [lore_how] done — {len(lore_how['narrative_axes'])} axes, "
        f"{len(lore_how['characters'])} characters"
    )
    return lore_how_path


def load_lore_how(lore_dir: Path) -> dict:
    path = lore_dir / LORE_HOW_FILE
    if not path.exists():
        return {}
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def synthesis_context_block(lore_dir: Path) -> str:
    """
    Format lore_how.yaml as a prompt injection block.
    Used by all step-6 analyze functions as their primary narrative anchor.
    """
    s = load_lore_how(lore_dir)
    if not s or not s.get("_done"):
        return "none"

    lines = []

    ctx = s.get("overall_context") or ""
    if ctx:
        lines.append(f"Overall: {ctx}")

    axes = s.get("narrative_axes") or []
    if axes:
        lines.append("\nNarrative axes:")
        for a in axes[:6]:
            chars = ", ".join((a.get("characters") or [])[:4])
            lines.append(f"  - {a.get('name', '')}: {a.get('summary', '')} [{chars}]")

    chars = s.get("characters") or []
    if chars:
        lines.append("\nCharacter arcs:")
        for c in chars[:15]:
            rels = ", ".join((c.get("key_relations") or [])[:3])
            line = f"  - {c.get('name', '')}: {c.get('role', '')}. Arc: {c.get('arc', '')}"
            if rels:
                line += f" | {rels}"
            lines.append(line)

    return "\n".join(lines) or "none"
