"""
Gradio interface — user entry point.
Usage: python src/interface.py
"""

import gradio as gr
from rag_pipeline import build_query_engine, interroger


query_engine = None


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

    historique.append((question, reponse))

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


with gr.Blocks(title="IA Roleplay Discord", theme=gr.themes.Soft()) as app:
    gr.Markdown("# IA Analyse Roleplay Discord\nPostez vos questions sur vos sessions de RP.")

    with gr.Row():
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(label="Conversation", height=500)
            with gr.Row():
                question_input = gr.Textbox(
                    placeholder="Ex : Qu'est-ce qu'Antoine cherche à faire ? Quel est le lien entre Garrance et Gaulthier ?",
                    label="Votre question",
                    scale=5,
                )
                envoyer_btn = gr.Button("Envoyer", variant="primary", scale=1)

        with gr.Column(scale=1):
            gr.Markdown("### Sources utilisées")
            sources_output = gr.Textbox(label="Scènes consultées", lines=14, interactive=False)
            init_btn = gr.Button("Initialiser le moteur", variant="secondary")
            init_status = gr.Textbox(label="Statut", interactive=False)

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


if __name__ == "__main__":
    app.launch(server_name="0.0.0.0", server_port=7860, share=False)
