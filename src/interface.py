"""
Gradio interface — user entry point.
Usage: python src/interface.py

Two modes:
  Q&A    — questions routed through the RAG pipeline
  Lore   — affirmations extracted by Mistral and merged into lore.yaml
"""

import yaml
from pathlib import Path

import gradio as gr
from rag_pipeline import build_query_engine, interroger
from lore_extractor import merge_into_lore, _call_mistral, LORE_PATH


query_engine = None


# ── RAG mode ──────────────────────────────────────────────────────────────────

def initialiser():
    global query_engine
    try:
        query_engine = build_query_engine()
        return "Moteur RAG initialisé. Prêt à répondre."
    except Exception as e:
        return f"Erreur d'initialisation : {e}"


def repondre(question: str, historique: list):
    if not question.strip():
        return historique, "", ""

    if query_engine is None:
        return historique, "Le moteur n'est pas encore initialisé.", ""

    try:
        reponse, sources = interroger(question, query_engine)
    except Exception as e:
        return historique, f"Erreur : {e}", ""

    historique.append({"role": "user", "content": question})
    historique.append({"role": "assistant", "content": reponse})

    lignes_sources = []
    for s in sources:
        chars = ", ".join(s.get("characters") or []) or "?"
        tags = ", ".join(s.get("scene_tags") or []) or "—"
        lignes_sources.append(
            f"• [{s.get('arc','?')}] {s.get('channel','?')} — {str(s.get('start','?'))[:10]}"
            f" | personnages: {chars}"
            f" | tags: {tags}"
            f" | score: {s.get('score','?')}"
        )
    sources_texte = "\n".join(lignes_sources) if lignes_sources else "Aucune source trouvée."

    return historique, "", sources_texte


# ── Lore ingestion mode ───────────────────────────────────────────────────────

LORE_EXTRACTION_SYSTEM = (
    "Tu es un extracteur de lore pour jeux de rôle. "
    "L'utilisateur te donne une affirmation sur l'univers. "
    "Tu retournes UNIQUEMENT du JSON valide extrayant les entités et relations mentionnées."
)

LORE_EXTRACTION_PROMPT = """\
L'utilisateur a fourni cette information sur l'univers :
\"{statement}\"

Extrais les entités et relations dans ce JSON (laisse {{}} ou [] si vide) :

{{
  "characters": {{"Nom": {{"aliases": [], "description": ""}}}},
  "places": {{"Nom": {{"description": ""}}}},
  "events": {{"id": {{"label": "", "description": "", "timestamp": null, "state": "fut"}}}},
  "objects": {{"Nom": {{"description": ""}}}},
  "cultures": {{"Nom": {{"description": ""}}}},
  "intentions": {{}},
  "narrative_axes": {{}},
  "character_knowledge": {{}},
  "universe_rules": [],
  "relations": [
    {{
      "from": "", "from_type": "character|place|event|object|culture",
      "rel": "type", "to": "", "to_type": "character|place|event|object|culture",
      "state": "est", "confidence": "high"
    }}
  ]
}}"""


def _diff_summary(extracted: dict, added: int) -> str:
    """Builds a human-readable summary of what was extracted and added."""
    lines = []
    for section in ("characters", "places", "events", "objects", "cultures",
                    "intentions", "narrative_axes"):
        items = list((extracted.get(section) or {}).keys())
        if items:
            lines.append(f"  {section} : {', '.join(items)}")
    rules = extracted.get("universe_rules") or []
    if rules:
        lines.append(f"  règles : {len(rules)} entrée(s)")
    rels = extracted.get("relations") or []
    if rels:
        rel_strs = [f"{r.get('from')} —{r.get('rel')}→ {r.get('to')}" for r in rels[:5]]
        lines.append(f"  relations : {', '.join(rel_strs)}")
    ck = extracted.get("character_knowledge") or {}
    if ck:
        lines.append(f"  connaissances : {', '.join(ck.keys())}")

    if not lines:
        return "Rien d'extractible dans cet énoncé."

    status = f"✓ {added} élément(s) nouveau(x) ajouté(s) à lore.yaml" if added > 0 \
        else "⚠ Déjà connu — aucun ajout (entrées existantes non écrasées)"
    return status + "\n\nExtrait :\n" + "\n".join(lines)


def ajouter_au_lore(affirmation: str, historique: list):
    if not affirmation.strip():
        return historique, "", "Aucune affirmation fournie."

    historique.append({"role": "user", "content": f"[LORE] {affirmation}"})

    try:
        prompt = LORE_EXTRACTION_SYSTEM + "\n\n" + \
                 LORE_EXTRACTION_PROMPT.format(statement=affirmation)

        # Reuse _call_mistral via a small override
        import requests, json as _json
        from lore_extractor import OLLAMA_URL, LLM_MODEL
        resp = requests.post(
            OLLAMA_URL,
            json={"model": LLM_MODEL, "prompt": prompt, "format": "json", "stream": False},
            timeout=60,
        )
        resp.raise_for_status()
        extracted = _json.loads(resp.json().get("response", "{}"))
        added = merge_into_lore(extracted, source=f"interface:{affirmation[:60]}")
        summary = _diff_summary(extracted, added)

    except Exception as e:
        summary = f"Erreur : {e}"
        extracted = {}

    historique.append({"role": "assistant", "content": summary})
    return historique, "", summary


def lire_lore_resume() -> str:
    """Returns a short human-readable snapshot of the current lore."""
    if not LORE_PATH.exists():
        return "lore.yaml vide ou absent."
    with open(LORE_PATH, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    lines = []
    for section in ("characters", "places", "events", "objects",
                    "cultures", "intentions", "narrative_axes"):
        nodes = data.get(section) or {}
        if nodes:
            lines.append(f"{section} ({len(nodes)}) : {', '.join(list(nodes.keys())[:8])}")
    rules = data.get("universe_rules") or []
    if rules:
        lines.append(f"universe_rules ({len(rules)})")
    rels = data.get("relations") or []
    if rels:
        lines.append(f"relations ({len(rels)})")
    ck = data.get("character_knowledge") or {}
    if ck:
        lines.append(f"character_knowledge : {', '.join(ck.keys())}")

    return "\n".join(lines) if lines else "Lore vide."


# ── UI ────────────────────────────────────────────────────────────────────────

with gr.Blocks(title="IA Roleplay Discord", theme=gr.themes.Soft()) as app:
    gr.Markdown("# IA Analyse Roleplay Discord")

    with gr.Row():
        # ── Left: chat ──────────────────────────────────────────────────────
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(label="Conversation", height=520)

            with gr.Tabs():
                with gr.Tab("❓ Question"):
                    with gr.Row():
                        question_input = gr.Textbox(
                            placeholder="Quel est le lien entre Garrance et Gaulthier ?",
                            label="Question sur le RP",
                            scale=5,
                        )
                        envoyer_btn = gr.Button("Envoyer", variant="primary", scale=1)

                with gr.Tab("📖 Ajouter au lore"):
                    with gr.Row():
                        lore_input = gr.Textbox(
                            placeholder='Ex : "Antoine est le fils de Marc Duverne, un ancien militaire décédé"',
                            label="Affirmation sur l'univers",
                            scale=5,
                        )
                        lore_btn = gr.Button("Ajouter", variant="primary", scale=1)

        # ── Right: panel ────────────────────────────────────────────────────
        with gr.Column(scale=1):
            with gr.Tabs():
                with gr.Tab("Sources"):
                    sources_output = gr.Textbox(
                        label="Scènes consultées", lines=12, interactive=False
                    )

                with gr.Tab("Lore actuel"):
                    lore_snapshot = gr.Textbox(
                        label="État du lore.yaml", lines=12, interactive=False
                    )
                    refresh_btn = gr.Button("Rafraîchir", size="sm")

            init_btn = gr.Button("Initialiser le moteur RAG", variant="secondary")
            init_status = gr.Textbox(label="Statut", interactive=False, lines=1)

    # ── Events ──────────────────────────────────────────────────────────────
    init_btn.click(initialiser, outputs=init_status)

    envoyer_btn.click(
        repondre,
        inputs=[question_input, chatbot],
        outputs=[chatbot, question_input, sources_output],
    )
    question_input.submit(
        repondre,
        inputs=[question_input, chatbot],
        outputs=[chatbot, question_input, sources_output],
    )

    lore_btn.click(
        ajouter_au_lore,
        inputs=[lore_input, chatbot],
        outputs=[chatbot, lore_input, sources_output],
    )
    lore_input.submit(
        ajouter_au_lore,
        inputs=[lore_input, chatbot],
        outputs=[chatbot, lore_input, sources_output],
    )

    refresh_btn.click(lire_lore_resume, outputs=lore_snapshot)
    app.load(lire_lore_resume, outputs=lore_snapshot)


if __name__ == "__main__":
    app.launch(server_name="0.0.0.0", server_port=7860, share=False)
