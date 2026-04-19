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

DATA_DIR     = ROOT / "data"
LORE_DIR     = DATA_DIR / "lore"
ANALYSIS_DIR = DATA_DIR / "analysis"
PYTHON       = sys.executable

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
                    minimum=1, maximum=4, step=1, value=1,
                    label="From step",
                    scale=1,
                )
                only_step_input = gr.Textbox(
                    value="",
                    label="Only step (blank = all)",
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

            refresh_stats_btn.click(lore_stats, outputs=stats_box)
            refresh_list_btn.click(refresh_entity_list, cat_radio, entity_dd)
            cat_radio.change(refresh_entity_list, cat_radio, entity_dd)
            entity_dd.change(show_entity, [cat_radio, entity_dd], entity_view)


if __name__ == "__main__":
    app.launch(server_name="0.0.0.0", server_port=7860, share=False, theme=gr.themes.Soft())
