"""
Build fine-tuning dataset from confirmed_scenes/*.yaml

Each confirmed scene produces one training example per analysis step.
Format: ChatML (system / user / assistant)

Output: data/finetune/dataset.jsonl
Usage:  python src/dataset_builder.py [--step who|where|which|what|how|all]
"""

import argparse
import json
import sys
from pathlib import Path

import yaml

ROOT          = Path(__file__).resolve().parent.parent
CONFIRMED_DIR = ROOT / "data" / "confirmed_scenes"
SCENES_DIR    = ROOT / "data" / "scenes"
OUT_DIR       = ROOT / "data" / "finetune"

SYSTEM_BASE = (
    "You are a narrative analyst specialized in French roleplay transcripts. "
    "Extract structured information from RP scenes. "
    "Scenes follow these conventions: *...* = action/description, "
    '\"...\" = dialogue or reported speech, '
    "a `-` after punctuation marks a speaker change within a message. "
    "Authors (Discord usernames) are NOT characters. "
    "Return only valid JSON, lowercase values, no pronouns in name lists."
)

SYSTEM_PER_STEP = {
    "when": SYSTEM_BASE + " Your task: extract the TEMPORAL context of the scene.",
    "where": SYSTEM_BASE + " Your task: extract LOCATIONS present in the scene.",
    "who": SYSTEM_BASE + " Your task: identify CHARACTERS present or active in the scene.",
    "which": SYSTEM_BASE + " Your task: identify named CONCEPTS (objects, factions, ideologies, systems) in the scene.",
    "what": SYSTEM_BASE + " Your task: produce an EXHAUSTIVE list of events in the scene.",
    "how": SYSTEM_BASE + " Your task: identify causal LINKS, character RELATIONS/SENTIMENTS, and narrative connections.",
}

USER_TEMPLATE = "Analyze this RP scene:\n---\n{scene_text}\n---"

# Context fields injected into how/what user prompts
HOW_CONTEXT_FIELDS  = ("when", "where", "who", "which", "what")
WHAT_CONTEXT_FIELDS = ("when", "where", "who", "which")


def _scene_text(scene_id: str) -> str:
    for sf in SCENES_DIR.glob(f"**/{scene_id}.json"):
        try:
            data = json.loads(sf.read_text(encoding="utf-8"))
            msgs = data.get("messages", [])
            return "\n".join(
                f"{(m.get('author') or {}).get('name', '?') if isinstance(m.get('author'), dict) else m.get('author', '?')}: "
                f"{m.get('content_en') or m.get('content', '')}"
                for m in msgs
            )
        except Exception:
            pass
    return ""


def _build_user(step: str, scene_text: str, confirmed: dict) -> str:
    base = USER_TEMPLATE.format(scene_text=scene_text)

    if step == "what" and any(confirmed.get(f) for f in WHAT_CONTEXT_FIELDS):
        ctx_lines = []
        for f in WHAT_CONTEXT_FIELDS:
            if confirmed.get(f):
                ctx_lines.append(f"{f.upper()}:\n{json.dumps(confirmed[f], ensure_ascii=False)}")
        return base + "\n\nContext from previous steps:\n" + "\n\n".join(ctx_lines)

    if step == "how" and any(confirmed.get(f) for f in HOW_CONTEXT_FIELDS):
        ctx_lines = []
        for f in HOW_CONTEXT_FIELDS:
            if confirmed.get(f):
                ctx_lines.append(f"{f.upper()}:\n{json.dumps(confirmed[f], ensure_ascii=False)}")
        return base + "\n\nContext from previous steps:\n" + "\n\n".join(ctx_lines)

    return base


def _build_assistant(step: str, confirmed: dict) -> str:
    data = confirmed.get(step)
    if not data:
        return ""
    return json.dumps(data, ensure_ascii=False, indent=2)


def build_dataset(steps: list[str]) -> list[dict]:
    if not CONFIRMED_DIR.exists():
        print(f"[error] {CONFIRMED_DIR} does not exist — annotate scenes first.")
        return []

    confirmed_files = sorted(CONFIRMED_DIR.glob("*.yaml"))
    print(f"Found {len(confirmed_files)} confirmed scenes")

    examples = []
    skipped = 0

    for yf in confirmed_files:
        try:
            confirmed = yaml.safe_load(yf.read_text(encoding="utf-8")) or {}
        except Exception as e:
            print(f"  [skip] {yf.name}: {e}")
            skipped += 1
            continue

        if confirmed.get("status") != "confirmed":
            continue

        scene_id   = confirmed.get("scene_id", yf.stem)
        scene_text = _scene_text(scene_id)
        if not scene_text:
            print(f"  [warn] no scene text found for {scene_id}")
            skipped += 1
            continue

        for step in steps:
            assistant_content = _build_assistant(step, confirmed)
            if not assistant_content or assistant_content == "{}":
                continue

            examples.append({
                "messages": [
                    {"role": "system",    "content": SYSTEM_PER_STEP[step]},
                    {"role": "user",      "content": _build_user(step, scene_text, confirmed)},
                    {"role": "assistant", "content": assistant_content},
                ]
            })

    print(f"Built {len(examples)} training examples ({skipped} scenes skipped)")
    return examples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--step",
        default="all",
        choices=["all", "when", "where", "who", "which", "what", "how"],
        help="Which analysis step to build examples for (default: all)",
    )
    args = parser.parse_args()

    steps = list(SYSTEM_PER_STEP.keys()) if args.step == "all" else [args.step]

    examples = build_dataset(steps)
    if not examples:
        sys.exit(1)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "dataset.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print(f"Saved {len(examples)} examples → {out_path}")

    # Stats
    step_counts: dict[str, int] = {}
    for ex in examples:
        sys_msg = ex["messages"][0]["content"]
        for s in SYSTEM_PER_STEP:
            if f"task: {s}" in sys_msg.lower() or f"task: extract" in sys_msg.lower():
                step_counts[s] = step_counts.get(s, 0) + 1
                break
    if step_counts:
        print("Examples per step:", step_counts)


if __name__ == "__main__":
    main()
