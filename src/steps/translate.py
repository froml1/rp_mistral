"""Step 2 - Translate: add content_en field to each message."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from llm import call_llm_json

BATCH_SIZE = 20

_PROMPT = """\
Translate the following Discord RP messages from French to English.
Preserve narrative style, names, and formatting (*actions*, "dialogue").
Return JSON: {{"messages": [{{"index": 0, "content_en": "..."}}]}}

Messages:
{messages}"""


def _translate_batch(messages: list[dict]) -> dict[int, str]:
    formatted = "\n".join(
        f"[{i}] {m.get('content', '')}"
        for i, m in enumerate(messages)
    )
    data = call_llm_json(_PROMPT.format(messages=formatted), num_predict=2048, num_ctx=8192)
    return {
        item["index"]: item.get("content_en", "")
        for item in (data.get("messages") or [])
        if isinstance(item, dict) and "index" in item
    }


def run_translate(purged_dir: Path, out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    files = list(purged_dir.glob("**/*.json"))
    if not files:
        print(f"  No purged files found in {purged_dir}")
        return []

    produced = []
    for fp in files:
        out_path = out_dir / fp.name
        if out_path.exists():
            print(f"  [skip] {fp.name} -> already translated")
            produced.append(out_path)
            continue

        print(f"  Translating {fp.name}...")
        with open(fp, encoding="utf-8") as f:
            data = json.load(f)
        messages = data.get("messages", data) if isinstance(data, dict) else data

        for start in range(0, len(messages), BATCH_SIZE):
            batch = messages[start:start + BATCH_SIZE]
            translations = _translate_batch(batch)
            for j, msg in enumerate(batch):
                msg["content_en"] = translations.get(j, msg.get("content", ""))
            print(f"    {min(start + BATCH_SIZE, len(messages))}/{len(messages)} messages translated")

        out_path.write_text(
            json.dumps({"messages": messages}, ensure_ascii=False, indent=2),
            encoding="utf-8"
        )
        produced.append(out_path)

    return produced
