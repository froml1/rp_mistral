"""
Gradio interface — query + pipeline control.
Usage: .venv/bin/python src/interface.py
"""

import json
import subprocess
import sys
import threading
from pathlib import Path

import gradio as gr
import yaml

ROOT    = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from query import answer, load_all_lore
from graph  import build_social_graph, build_relation_evolution, build_pcnpc_heuristic, build_tension_curve, build_power_map
from export import run_export
from steps.manual_lore import (
    load_manual_char, save_manual_char, load_all_manual_chars,
    load_manual_place, save_manual_place, load_all_manual_places,
    load_manual_concept, save_manual_concept, load_all_manual_concepts,
    load_manual_events, save_manual_events,
    manual_char_path, manual_place_path, manual_concept_path, manual_events_path,
    MANUAL_DIR,
)

DATA_DIR       = ROOT / "data"
LORE_DIR       = DATA_DIR / "lore"
ANALYSIS_DIR   = DATA_DIR / "analysis"
SCENES_DIR     = DATA_DIR / "scenes"
CONFIRMED_DIR  = DATA_DIR / "confirmed_scenes"
PYTHON         = sys.executable

# ── Pipeline ──────────────────────────────────────────────────────────────────

_pipeline_log: list[str] = []
_pipeline_proc: subprocess.Popen | None = None
_pipeline_running = False
_pipeline_exit_code: int | None = None


def _resolve_exports(exports_dir: str) -> str:
    p = Path(exports_dir.strip())
    if not p.is_absolute():
        p = ROOT / p
    return str(p)


def _run_pipeline(from_step, only_step, scene_id, exports_dir):
    global _pipeline_running, _pipeline_proc, _pipeline_exit_code
    _pipeline_running = True
    _pipeline_exit_code = None
    _pipeline_log.clear()

    cmd = [PYTHON, str(SRC_DIR / "pipeline.py"), _resolve_exports(exports_dir)]
    if only_step:
        cmd += ["--only-step", str(only_step)]
    elif from_step > 1:
        cmd += ["--from-step", str(from_step)]
    if scene_id.strip():
        cmd += ["--scene", scene_id.strip()]

    _pipeline_log.append(f"[cmd] {' '.join(cmd)}")
    _pipeline_log.append(f"[python exists] {Path(PYTHON).exists()}")
    _pipeline_log.append(f"[pipeline exists] {(SRC_DIR / 'pipeline.py').exists()}")

    try:
        env = {**__import__("os").environ, "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"}
        _pipeline_proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            env=env,
            cwd=str(ROOT),
        )
        for line in _pipeline_proc.stdout:
            _pipeline_log.append(line.rstrip())
        _pipeline_proc.wait()
        _pipeline_exit_code = _pipeline_proc.returncode
        _pipeline_log.append(f"\n[exit code {_pipeline_proc.returncode}]")
    except Exception as e:
        _pipeline_log.append(f"[error] {e}")
    finally:
        _pipeline_running = False
        _pipeline_proc = None


def start_pipeline(from_step, only_step_str, scene_id, exports_dir):
    import time
    if _pipeline_running:
        yield "Pipeline already running…", get_pipeline_log()
        return
    only_step = int(only_step_str) if only_step_str.strip() else None
    threading.Thread(
        target=_run_pipeline,
        args=(from_step, only_step, scene_id, exports_dir),
        daemon=True,
    ).start()
    time.sleep(0.2)
    while _pipeline_running:
        yield "Running…", "\n".join(_pipeline_log[-300:])
        time.sleep(0.5)
    code = _pipeline_exit_code
    status = f"Done ✓ (exit {code})" if code == 0 else f"Error ✗ (exit {code})"
    yield status, "\n".join(_pipeline_log[-300:])


def stop_pipeline():
    global _pipeline_proc
    if _pipeline_proc and _pipeline_proc.poll() is None:
        _pipeline_proc.terminate()
        _pipeline_log.append("\n[stopped by user]")
        return "Pipeline stopped."
    return "No pipeline running."


def get_pipeline_log():
    return "\n".join(_pipeline_log[-200:])


# ── Lore stats ────────────────────────────────────────────────────────────────

def lore_stats() -> str:
    lore = load_all_lore()
    lines = []
    for cat in ("characters", "places", "concepts"):
        entities = lore[cat]
        if entities:
            names = ", ".join(e.get("name", e.get("_file", "?")) for e in entities[:12])
            suffix = f"… +{len(entities)-12}" if len(entities) > 12 else ""
            lines.append(f"**{cat}** ({len(entities)}): {names}{suffix}")
        else:
            lines.append(f"**{cat}**: —")

    how_ctx = lore.get("how_context") or {}
    lines.append(f"**scenes indexed**: {len(how_ctx)}")

    scene_dirs = list(ANALYSIS_DIR.glob("*/what.json")) if ANALYSIS_DIR.exists() else []
    lines.append(f"**scenes analyzed**: {len(scene_dirs)}")

    return "\n".join(lines) if lines else "No lore found. Run the pipeline first."


# ── Lore browser ─────────────────────────────────────────────────────────────

def list_entities(category: str) -> list[str]:
    path = LORE_DIR / category
    if not path.exists():
        return []
    return sorted(f.stem for f in path.glob("*.yaml"))


def show_entity(category: str, slug: str) -> str:
    if not slug:
        return ""
    path = LORE_DIR / category / f"{slug}.yaml"
    if not path.exists():
        return "Not found."
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    data.pop("_file", None)
    return yaml.dump(data, allow_unicode=True, sort_keys=False)


def refresh_entity_list(category: str):
    items = list_entities(category)
    return gr.Dropdown(choices=items, value=None)


# ── Annotation ────────────────────────────────────────────────────────────────

def _ann_queue() -> list[str]:
    """Scenes with a full analysis (what.json) that are not yet confirmed/rejected."""
    confirmed = set()
    if CONFIRMED_DIR.exists():
        confirmed = {f.stem for f in CONFIRMED_DIR.glob("*.yaml")}
    queue = []
    if ANALYSIS_DIR.exists():
        for d in sorted(ANALYSIS_DIR.iterdir()):
            if d.is_dir() and (d / "what.json").exists() and d.name not in confirmed:
                queue.append(d.name)
    return queue


def _ann_load(scene_id: str) -> tuple:
    """Return (raw_text, when_yaml, where_yaml, who_yaml, which_yaml, what_yaml, how_yaml)."""
    scene_dir = ANALYSIS_DIR / scene_id

    # Raw scene text (prefer translated content)
    raw_text = ""
    for sf in SCENES_DIR.glob(f"**/{scene_id}.json"):
        try:
            data = json.loads(sf.read_text(encoding="utf-8"))
            msgs = data.get("messages", [])
            raw_text = "\n".join(
                f"{(m.get('author') or {}).get('name', '?') if isinstance(m.get('author'), dict) else m.get('author', '?')}: "
                f"{m.get('content_en') or m.get('content', '')}"
                for m in msgs
            )
        except Exception:
            pass
        break

    def _j(name):
        p = scene_dir / name
        try:
            return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
        except Exception:
            return {}

    def _y(d):
        return yaml.dump(d, allow_unicode=True, sort_keys=False) if d else ""

    return (
        raw_text,
        _y(_j("when.json")),
        _y(_j("where.json")),
        _y(_j("who.json")),
        _y(_j("which.json")),
        _y(_j("what.json")),
        _y(_j("how.json")),
    )


def _ann_save(scene_id: str, status: str, when_y, where_y, who_y, which_y, what_y, how_y):
    CONFIRMED_DIR.mkdir(parents=True, exist_ok=True)
    record = {"scene_id": scene_id, "status": status}
    if status == "confirmed":
        def _parse(s):
            try:
                return yaml.safe_load(s) or {}
            except Exception:
                return {}
        record.update({
            "when":  _parse(when_y),
            "where": _parse(where_y),
            "who":   _parse(who_y),
            "which": _parse(which_y),
            "what":  _parse(what_y),
            "how":   _parse(how_y),
        })
    out = CONFIRMED_DIR / f"{scene_id}.yaml"
    with open(out, "w", encoding="utf-8") as f:
        yaml.dump(record, f, allow_unicode=True, sort_keys=False)


def ann_goto(idx: int):
    """Load scene at queue[idx], return all UI values."""
    queue = _ann_queue()
    total = len(queue)
    if not queue:
        empty = ("", "", "", "", "", "", "")
        return (0, "No scenes to annotate — run the pipeline first.", *empty)
    idx = max(0, min(idx, total - 1))
    scene_id = queue[idx]
    raw, when_y, where_y, who_y, which_y, what_y, how_y = _ann_load(scene_id)
    confirmed_count = len(list(CONFIRMED_DIR.glob("*.yaml"))) if CONFIRMED_DIR.exists() else 0
    progress = f"**{idx + 1} / {total}** pending — {confirmed_count} confirmed — `{scene_id}`"
    return (idx, progress, raw, when_y, where_y, who_y, which_y, what_y, how_y)


def ann_validate(idx, when_y, where_y, who_y, which_y, what_y, how_y):
    queue = _ann_queue()
    if queue and 0 <= idx < len(queue):
        _ann_save(queue[idx], "confirmed", when_y, where_y, who_y, which_y, what_y, how_y)
    return ann_goto(idx)  # queue shrinks → same idx = next scene


def ann_reject(idx, when_y, where_y, who_y, which_y, what_y, how_y):
    queue = _ann_queue()
    if queue and 0 <= idx < len(queue):
        _ann_save(queue[idx], "rejected", when_y, where_y, who_y, which_y, what_y, how_y)
    return ann_goto(idx)


def ann_skip(idx, *_):
    return ann_goto(idx + 1)


def ann_prev(idx, *_):
    return ann_goto(idx - 1)


def ann_refresh(*_):
    return ann_goto(0)


# ── Manual Lore ───────────────────────────────────────────────────────────────

def _ml_list_entities(category: str) -> list[str]:
    cat_map = {"Character": "characters", "Place": "places", "Concept": "concepts", "Event": "events"}
    d = MANUAL_DIR / cat_map.get(category, "characters")
    if not d.exists():
        return []
    return sorted(f.stem for f in d.glob("*.yaml"))


def _ml_load(category: str, entity_name: str):
    """Load manual entry → return (desc1, desc2_or_type, lists_text, advanced_yaml, status)."""
    name = entity_name.strip()
    if not name:
        return "", "", "", "", "Enter an entity name."
    if category == "Character":
        d = load_manual_char(name)
        desc1 = d.get("description_physical", "")
        desc2 = d.get("description_psychological", "")
        job   = d.get("job", "")
        lists = (
            "appellations:\n" + "\n".join(d.get("appellations") or []) + "\n\n"
            "beliefs:\n"      + "\n".join(d.get("beliefs") or [])      + "\n\n"
            "likes:\n"        + "\n".join(d.get("likes") or [])        + "\n\n"
            "dislikes:\n"     + "\n".join(d.get("dislikes") or [])     + "\n\n"
            "main_locations:\n" + "\n".join(d.get("main_locations") or []) + "\n\n"
            "misc:\n"         + "\n".join(d.get("misc") or [])
        )
        adv = yaml.dump({
            "personality_axes": d.get("personality_axes") or {},
            "competency_axes":  d.get("competency_axes")  or {},
            "emotional_polarity": d.get("emotional_polarity") or {},
        }, allow_unicode=True, sort_keys=False)
        return desc1, desc2, job, lists, adv, f"Loaded: {name}" if d else f"New entry: {name}"
    elif category == "Place":
        d = load_manual_place(name)
        desc1 = d.get("description", "")
        lists = (
            "appellations:\n" + "\n".join(d.get("appellations") or []) + "\n\n"
            "attributes:\n"   + "\n".join(d.get("attributes") or [])
        )
        return desc1, "", "", lists, "", f"Loaded: {name}" if d else f"New entry: {name}"
    elif category == "Concept":
        d = load_manual_concept(name)
        desc1 = d.get("description", "")
        desc2 = d.get("significance", "")
        ctype = d.get("type", "")
        lists = (
            "appellations:\n"      + "\n".join(d.get("appellations") or [])      + "\n\n"
            "related_characters:\n"+ "\n".join(d.get("related_characters") or [])
        )
        return desc1, desc2, ctype, lists, "", f"Loaded: {name}" if d else f"New entry: {name}"
    elif category == "Event":
        d = load_manual_events(name)
        summary = d.get("summary", "")
        evts = yaml.dump({"events": d.get("events") or []}, allow_unicode=True, sort_keys=False)
        return summary, "", "", "", evts, f"Loaded scene: {name}" if d else f"New scene: {name}"
    return "", "", "", "", "", ""


def _ml_save(category: str, entity_name: str, desc1: str, desc2: str, field3: str, lists_text: str, adv_yaml: str):
    name = entity_name.strip()
    if not name:
        return "Error: name is required.", gr.Dropdown(choices=_ml_list_entities(category))

    def _parse_section(text: str, key: str) -> list[str]:
        """Extract items under a 'key:\n' header in lists_text."""
        lines = text.split("\n")
        capture = False
        result = []
        for line in lines:
            stripped = line.strip()
            if stripped.lower().startswith(f"{key}:"):
                capture = True
                continue
            if capture:
                if stripped and not stripped.endswith(":"):
                    result.append(stripped)
                elif stripped == "" and result:
                    # blank line = end of section only if next non-blank starts a header
                    pass
                elif stripped.endswith(":") and stripped != f"{key}:":
                    capture = False
        return [x for x in result if x]

    def _adv() -> dict:
        try:
            return yaml.safe_load(adv_yaml) or {}
        except Exception:
            return {}

    try:
        if category == "Character":
            data = {
                "description_physical":    desc1.strip(),
                "description_psychological": desc2.strip(),
                "job":          field3.strip(),
                "appellations": _parse_section(lists_text, "appellations"),
                "beliefs":      _parse_section(lists_text, "beliefs"),
                "likes":        _parse_section(lists_text, "likes"),
                "dislikes":     _parse_section(lists_text, "dislikes"),
                "main_locations": _parse_section(lists_text, "main_locations"),
                "misc":         _parse_section(lists_text, "misc"),
            }
            adv = _adv()
            if adv.get("personality_axes"):
                data["personality_axes"] = adv["personality_axes"]
            if adv.get("competency_axes"):
                data["competency_axes"] = adv["competency_axes"]
            if adv.get("emotional_polarity"):
                data["emotional_polarity"] = adv["emotional_polarity"]
            save_manual_char(name, data)

        elif category == "Place":
            data = {
                "description":  desc1.strip(),
                "appellations": _parse_section(lists_text, "appellations"),
                "attributes":   _parse_section(lists_text, "attributes"),
            }
            save_manual_place(name, data)

        elif category == "Concept":
            data = {
                "description":        desc1.strip(),
                "significance":       desc2.strip(),
                "type":               field3.strip(),
                "appellations":       _parse_section(lists_text, "appellations"),
                "related_characters": _parse_section(lists_text, "related_characters"),
            }
            save_manual_concept(name, data)

        elif category == "Event":
            adv = _adv()
            data = {
                "summary": desc1.strip(),
                "events":  adv.get("events") or [],
            }
            save_manual_events(name, data)

        return f"Saved: {name}", gr.Dropdown(choices=_ml_list_entities(category), value=name)
    except Exception as e:
        return f"Error: {e}", gr.Dropdown(choices=_ml_list_entities(category))


def _ml_delete(category: str, entity_name: str):
    name = entity_name.strip()
    if not name:
        return "Error: name is required.", gr.Dropdown(choices=_ml_list_entities(category))
    cat_map = {"Character": "characters", "Place": "places", "Concept": "concepts", "Event": "events"}
    import re as _re
    slug = _re.sub(r'[^\w]', '_', name.lower().strip())
    path = MANUAL_DIR / cat_map.get(category, "characters") / f"{slug}.yaml"
    if path.exists():
        path.unlink()
        return f"Deleted: {name}", gr.Dropdown(choices=_ml_list_entities(category), value=None)
    return f"Not found: {name}", gr.Dropdown(choices=_ml_list_entities(category))


# ── Chat ──────────────────────────────────────────────────────────────────────

def chat(question: str, history: list):
    if not question.strip():
        return history, ""
    history = history or []
    history.append({"role": "user", "content": question})
    try:
        response = answer(question)
    except Exception as e:
        response = f"Error: {e}"
    history.append({"role": "assistant", "content": response})
    return history, ""


# ── UI ────────────────────────────────────────────────────────────────────────

with gr.Blocks(title="RP_IA") as app:
    gr.Markdown("# RP_IA — Roleplay Analysis")

    with gr.Tabs():

        # ── Tab: Query ───────────────────────────────────────────────────────
        with gr.Tab("Query"):
            chatbot = gr.Chatbot(label="", height=520)
            with gr.Row():
                question_box = gr.Textbox(
                    placeholder="Ask anything about the RP universe…",
                    label="Question",
                    scale=5,
                )
                send_btn = gr.Button("Send", variant="primary", scale=1)

            send_btn.click(chat, [question_box, chatbot], [chatbot, question_box])
            question_box.submit(chat, [question_box, chatbot], [chatbot, question_box])

        # ── Tab: Pipeline ────────────────────────────────────────────────────
        with gr.Tab("Pipeline"):
            with gr.Row():
                exports_input = gr.Textbox(
                    value="data/exports",
                    label="Exports directory",
                    scale=3,
                )
                from_step_slider = gr.Slider(
                    minimum=1, maximum=6, step=1, value=1,
                    label="From step  (1=purge 2=translate 3=subdivide 4=synthesis+rp 5=analyze 6=post)",
                    scale=1,
                )
                only_step_input = gr.Textbox(
                    value="",
                    label="Only step (blank = all, 1-7)",
                    scale=1,
                )
                scene_input = gr.Textbox(
                    value="",
                    label="Scene ID (optional)",
                    scale=2,
                )

            with gr.Row():
                run_btn  = gr.Button("Run Pipeline", variant="primary", scale=2)
                stop_btn = gr.Button("Stop", variant="stop", scale=1)
                status   = gr.Textbox(label="Status", interactive=False, scale=3)

            log_box = gr.Textbox(
                label="Log",
                lines=20,
                interactive=False,
                autoscroll=True,
            )

            run_btn.click(
                start_pipeline,
                [from_step_slider, only_step_input, scene_input, exports_input],
                [status, log_box],
            )
            stop_btn.click(stop_pipeline, outputs=status)

        # ── Tab: Lore ────────────────────────────────────────────────────────
        with gr.Tab("Lore"):
            stats_box = gr.Markdown(lore_stats())
            refresh_stats_btn = gr.Button("Refresh", size="sm")

            gr.Markdown("---")

            with gr.Row():
                cat_radio = gr.Radio(
                    ["characters", "places", "concepts"],
                    value="characters",
                    label="Category",
                )
                entity_dd = gr.Dropdown(
                    choices=list_entities("characters"),
                    label="Entity",
                    scale=2,
                )
                refresh_list_btn = gr.Button("Refresh list", size="sm", scale=1)

            entity_view = gr.Code(label="YAML", language="yaml", lines=25)

            with gr.Row():
                export_all_btn   = gr.Button("Export all to Markdown", size="sm")
                export_story_btn = gr.Button("Export story summary", size="sm")
                reindex_btn      = gr.Button("Rebuild vector index", size="sm")
                export_status    = gr.Textbox(label="Status", interactive=False, scale=3)

            def do_export_all():
                try:
                    run_export()
                    return f"Export done → data/export_md/"
                except Exception as e:
                    return f"Error: {e}"

            def do_export_story():
                try:
                    run_export(only_story=True)
                    return "Story exported → data/export_md/story_so_far.md"
                except Exception as e:
                    return f"Error: {e}"

            def do_reindex():
                try:
                    from store import reindex
                    n = reindex()
                    return f"Index rebuilt — {n} entities"
                except Exception as e:
                    return f"Error: {e}"

            refresh_stats_btn.click(lore_stats, outputs=stats_box)
            refresh_list_btn.click(refresh_entity_list, cat_radio, entity_dd)
            cat_radio.change(refresh_entity_list, cat_radio, entity_dd)
            entity_dd.change(show_entity, [cat_radio, entity_dd], entity_view)
            export_all_btn.click(do_export_all, outputs=export_status)
            export_story_btn.click(do_export_story, outputs=export_status)
            reindex_btn.click(do_reindex, outputs=export_status)


        # ── Tab: Manual Lore ─────────────────────────────────────────────────
        with gr.Tab("Manual Lore"):
            gr.Markdown(
                "Add ground-truth data that will **override LLM extractions** during the pipeline. "
                "Data is stored in `data/lore/manual/` — separate from LLM data, never overwritten."
            )

            with gr.Row():
                ml_cat = gr.Radio(
                    ["Character", "Place", "Concept", "Event"],
                    value="Character",
                    label="Category",
                    scale=2,
                )
                ml_existing = gr.Dropdown(
                    choices=_ml_list_entities("Character"),
                    label="Existing entries (click to load)",
                    scale=3,
                )
                ml_refresh_btn = gr.Button("Refresh", size="sm", scale=1)

            ml_name = gr.Textbox(label="Name / Scene ID", placeholder="e.g. lena marchal  or  scene_007")

            with gr.Row():
                ml_desc1  = gr.Textbox(label="Description / Summary", lines=3, scale=3)
                ml_desc2  = gr.Textbox(label="Desc. Psychological / Significance", lines=3, scale=3)
                ml_field3 = gr.Textbox(label="Job / Type", lines=1, scale=1)

            ml_lists = gr.Textbox(
                label="Lists  (header: then one item per line — appellations / beliefs / likes / dislikes / main_locations / misc  or  attributes / related_characters)",
                lines=10,
                placeholder="appellations:\nlena\nmiss marchal\n\nbeliefs:\nhonor above all\n\nlikes:\nfighting",
            )

            ml_adv = gr.Code(
                label="Advanced YAML  (personality_axes / competency_axes / emotional_polarity  or  events list)",
                language="yaml",
                lines=12,
                value="personality_axes: {}\ncompetency_axes: {}\nemotional_polarity: {}",
            )

            with gr.Row():
                ml_save_btn   = gr.Button("Save", variant="primary", scale=2)
                ml_delete_btn = gr.Button("Delete", variant="stop", scale=1)
                ml_status     = gr.Textbox(label="Status", interactive=False, scale=4)

            # helpers
            def _ml_load_wrap(cat, name):
                res = _ml_load(cat, name)
                # res = (desc1, desc2, field3, lists, adv, status)  (6 values)
                return res[0], res[1], res[2], res[3], res[4], res[5]

            def _ml_load_from_dd(cat, slug):
                # slug is e.g. "lena_marchal" — convert back to name for display
                return _ml_load_wrap(cat, slug.replace("_", " "))

            def _ml_refresh_dd(cat):
                return gr.Dropdown(choices=_ml_list_entities(cat), value=None)

            _ml_form_outs = [ml_desc1, ml_desc2, ml_field3, ml_lists, ml_adv, ml_status]

            ml_existing.change(_ml_load_from_dd, [ml_cat, ml_existing], _ml_form_outs)
            ml_refresh_btn.click(_ml_refresh_dd, ml_cat, ml_existing)
            ml_cat.change(_ml_refresh_dd, ml_cat, ml_existing)

            ml_save_btn.click(
                _ml_save,
                [ml_cat, ml_name, ml_desc1, ml_desc2, ml_field3, ml_lists, ml_adv],
                [ml_status, ml_existing],
            )
            ml_delete_btn.click(
                _ml_delete,
                [ml_cat, ml_name],
                [ml_status, ml_existing],
            )

        # ── Tab: Annotate ────────────────────────────────────────────────────
        with gr.Tab("Annotate"):
            ann_idx   = gr.State(0)
            ann_progress = gr.Markdown("Load a scene to start.")

            with gr.Row():
                ann_prev_btn    = gr.Button("← Prev", size="sm", scale=1)
                ann_skip_btn    = gr.Button("Skip →", size="sm", scale=1)
                ann_refresh_btn = gr.Button("Refresh queue", size="sm", scale=1)
                ann_reject_btn  = gr.Button("Reject ✗", variant="stop", scale=2)
                ann_val_btn     = gr.Button("Validate ✓", variant="primary", scale=2)

            with gr.Row():
                with gr.Column(scale=2):
                    ann_raw = gr.Textbox(
                        label="Raw scene", lines=30, interactive=False
                    )
                with gr.Column(scale=3):
                    ann_when  = gr.Code(label="When",  language="yaml", lines=5)
                    ann_where = gr.Code(label="Where", language="yaml", lines=6)
                    ann_who   = gr.Code(label="Who",   language="yaml", lines=7)
                    ann_which = gr.Code(label="Which", language="yaml", lines=5)
                    ann_what  = gr.Code(label="What",  language="yaml", lines=8)
                    ann_how   = gr.Code(label="How",   language="yaml", lines=8)

            # shared outputs
            _ann_outs = [ann_idx, ann_progress, ann_raw, ann_when, ann_where, ann_who, ann_which, ann_what, ann_how]
            _ann_edit = [ann_when, ann_where, ann_who, ann_which, ann_what, ann_how]

            ann_val_btn.click(ann_validate,     [ann_idx] + _ann_edit, _ann_outs)
            ann_reject_btn.click(ann_reject,    [ann_idx] + _ann_edit, _ann_outs)
            ann_skip_btn.click(ann_skip,        [ann_idx] + _ann_edit, _ann_outs)
            ann_prev_btn.click(ann_prev,        [ann_idx] + _ann_edit, _ann_outs)
            ann_refresh_btn.click(ann_refresh,  [ann_idx] + _ann_edit, _ann_outs)

            # auto-load on tab open
            # l'auto load pause problem, c'est la selection au hasard du modèle... on reprendra ça plus tard. 
            # app.load(ann_goto, ann_idx, _ann_outs) # bug

        # ── Tab: Graph ───────────────────────────────────────────────────────
        with gr.Tab("Graph"):
            gr.Markdown("### Social graph & relationship analysis")

            with gr.Row():
                graph_social_btn   = gr.Button("Social graph",    variant="primary", scale=2)
                graph_tension_btn  = gr.Button("Tension curve",   scale=2)
                graph_power_btn    = gr.Button("Power map",        scale=2)
                graph_pcnpc_btn    = gr.Button("PC / NPC",         scale=1)
                graph_status       = gr.Textbox(label="Status", interactive=False, scale=3)

            graph_image = gr.Image(label="Graph", type="filepath", height=600)

            gr.Markdown("#### Relationship evolution between two characters")
            with gr.Row():
                evo_char_a = gr.Textbox(label="Character A", scale=2)
                evo_char_b = gr.Textbox(label="Character B", scale=2)
                evo_btn    = gr.Button("Generate", variant="primary", scale=1)

            evo_image  = gr.Image(label="Evolution chart", type="filepath", height=350)
            pcnpc_box  = gr.Textbox(label="PC/NPC results", lines=15, interactive=False)

            def do_social():
                try:
                    out = build_social_graph()
                    return str(out) if out else None, "Done."
                except Exception as e:
                    return None, f"Error: {e}"

            def do_evolution(a, b):
                if not a.strip() or not b.strip():
                    return None, "Enter both character names."
                try:
                    out = build_relation_evolution(a.strip(), b.strip())
                    if not out:
                        return None, f"No interactions found between {a} and {b}."
                    return str(out), "Done."
                except Exception as e:
                    return None, f"Error: {e}"

            def do_pcnpc():
                try:
                    results = build_pcnpc_heuristic()
                    if not results:
                        return "No data — run the pipeline first.", "Done."
                    lines = [f"{'Role':22} {'Character':28} Authors"]
                    for r in results:
                        lines.append(
                            f"{r['role']:22} {r['character']:28} "
                            f"({r['authors']}): {', '.join(r['author_list'][:4])}"
                        )
                    return "\n".join(lines), f"{len(results)} characters analyzed."
                except Exception as e:
                    return f"Error: {e}", ""

            def do_tension():
                try:
                    out = build_tension_curve()
                    return str(out) if out else None, "Done."
                except Exception as e:
                    return None, f"Error: {e}"

            def do_power():
                try:
                    out = build_power_map()
                    return str(out) if out else None, "Done."
                except Exception as e:
                    return None, f"Error: {e}"

            graph_social_btn.click(do_social,   outputs=[graph_image, graph_status])
            graph_tension_btn.click(do_tension, outputs=[graph_image, graph_status])
            graph_power_btn.click(do_power,     outputs=[graph_image, graph_status])
            evo_btn.click(do_evolution, [evo_char_a, evo_char_b], [evo_image, graph_status])
            graph_pcnpc_btn.click(do_pcnpc, outputs=[pcnpc_box, graph_status])


if __name__ == "__main__":
    app.launch(server_name="0.0.0.0", server_port=7860, share=False, theme=gr.themes.Soft())
