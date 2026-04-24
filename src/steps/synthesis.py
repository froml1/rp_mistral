"""
Step 4 — Synthesis: build a rich narrative synthesis directly from scene files.
Output frozen in data/lore/lore_how.yaml.

Two-phase hierarchical approach (Mistral context-window safe):
  Phase A — batch compression: N scenes → one summary per batch
  Phase B — global synthesis: all batch summaries → lore_how.yaml

lore_how.yaml contains:
  overall_context  — 2-3 sentence global description of the story
  narrative_axes   — main story threads with characters and tension
  characters       — role, arc, key relations per character

Incremental: already-computed batch summaries are reused on re-run.
"""

import json
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from llm import call_llm_json

LORE_HOW_FILE = "lore_how.yaml"
_BATCH_SIZE = 40

_BATCH_PROMPT = """\
Summarize the events and character dynamics from these RP scenes in 3-4 sentences.
Focus on: what happened, who was involved, key tensions, revelations, or turning points.

Scenes:
{scenes}

JSON: {{"summary": ""}}"""

_FINAL_PROMPT = """\
You are synthesizing a complete roleplay story. Build a rich narrative synthesis from the material below.

STORY SUMMARY (chronological batches):
{batch_summaries}

Produce:

1. CHARACTERS — for each main character: name, role in the story, narrative arc (how they evolve), key_relations (list of "other_character : relation_type").
2. NARRATIVE AXES — 2 to 6 main story threads. Each: name, summary (2 sentences), characters involved, core tension.
3. OVERALL CONTEXT — 2-3 sentences: setting, mood, central conflict of this story.

JSON:
{{
  "characters": [{{"name": "", "role": "", "arc": "", "key_relations": []}}],
  "narrative_axes": [{{"name": "", "summary": "", "characters": [], "tension": ""}}],
  "overall_context": ""
}}"""


def _scene_snippet(scene: dict, max_msgs: int = 8, max_chars: int = 120) -> str:
    msgs = scene.get("messages", [])[:max_msgs]
    lines = []
    for m in msgs:
        author = (m.get("author") or {}).get("name", "?") if isinstance(m.get("author"), dict) else str(m.get("author", "?"))
        content = (m.get("content_en") or m.get("content", ""))[:max_chars]
        lines.append(f"[{author}]: {content}")
    return "\n".join(lines)


def run_synthesis(scenes_dir: Path, lore_dir: Path) -> Path:
    """
    Build narrative synthesis from all scene files.
    Saves to data/lore/lore_how.yaml. Incremental: skips if already done.
    """
    lore_how_path = lore_dir / LORE_HOW_FILE

    scene_files = sorted(scenes_dir.glob("**/*.json"))
    if not scene_files:
        print("  [lore_how] no scene files found")
        return lore_how_path

    # Load existing to reuse cached batch summaries
    existing: dict = {}
    if lore_how_path.exists():
        existing = yaml.safe_load(lore_how_path.read_text(encoding="utf-8")) or {}

    if existing.get("_done"):
        n_axes  = len(existing.get("narrative_axes") or [])
        n_chars = len(existing.get("characters") or [])
        print(f"  [lore_how] already done — {n_axes} axes, {n_chars} characters")
        return lore_how_path

    # Build scene batches
    batches = [scene_files[i:i + _BATCH_SIZE] for i in range(0, len(scene_files), _BATCH_SIZE)]
    total = len(batches)
    cached: list[str] = existing.get("_batch_summaries") or []

    # Phase A: compress each batch
    batch_summaries: list[str] = list(cached)
    for i, batch in enumerate(batches[len(cached):], start=len(cached)):
        scenes_text_parts = []
        for sf in batch:
            try:
                scene = json.loads(sf.read_text(encoding="utf-8"))
                sid = scene.get("scene_id", sf.stem)
                snippet = _scene_snippet(scene)
                if snippet.strip():
                    scenes_text_parts.append(f"--- {sid} ---\n{snippet}")
            except Exception:
                continue

        if not scenes_text_parts:
            batch_summaries.append("(empty batch)")
            continue

        result = call_llm_json(
            _BATCH_PROMPT.format(scenes="\n\n".join(scenes_text_parts)),
            num_predict=400,
            num_ctx=6144,
        )
        summary = (result.get("summary") or "").strip()
        batch_summaries.append(summary or "(no summary)")
        print(f"  [lore_how] batch {i + 1}/{total} done")

        # Save progress after each batch
        existing["_batch_summaries"] = batch_summaries
        lore_dir.mkdir(parents=True, exist_ok=True)
        lore_how_path.write_text(yaml.dump(existing, allow_unicode=True, sort_keys=False), encoding="utf-8")

    # Phase B: global synthesis
    all_summaries = "\n\n".join(f"[batch {i+1}] {s}" for i, s in enumerate(batch_summaries))
    print(f"  [lore_how] global synthesis ({len(batch_summaries)} batches)…")

    final = call_llm_json(
        _FINAL_PROMPT.format(batch_summaries=all_summaries),
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
    """Format lore_how.yaml as a prompt injection block for step-6 analyze functions."""
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
