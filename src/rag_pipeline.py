"""
Pipeline RAG : interroge la base vectorielle avec Mistral via Ollama.
"""

import chromadb
from llama_index.core import VectorStoreIndex, Settings
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.vector_stores.chroma import ChromaVectorStore


CHROMA_PATH = "data/index"
COLLECTION_NAME = "rp_discord"
EMBED_MODEL = "nomic-embed-text"
LLM_MODEL = "mistral"

SYSTEM_PROMPT = """Tu es un assistant spécialisé dans l'analyse de roleplay narratif écrit.
Les textes qui te sont fournis proviennent d'exports Discord.
Conventions importantes :
- [Personnage] : ce personnage est actif dans le message (parole ou action)
- [Perso1, Perso2] : le joueur interprète plusieurs personnages dans le même message
- [auteur:descriptif] : message purement narratif, aucun personnage directement exprimé
- [-] : prise de parole secondaire dont le locuteur n'a pas pu être résolu automatiquement ; déduis-le du contexte narratif (par élimination ou cohérence de la scène)
- Les descriptions et actions sont encadrées par des astérisques : *description*
- Les guillemets ("...") signalent une citation dans un dialogue, une parole rapportée dans une description, ou la traduction d'une autre langue
- Les pronoms (il, elle, ils…) renvoient au personnage le plus récemment actif dans la conversation
- Un personnage peut être évoqué (pensée, souvenir, dialogue indirect) sans être physiquement présent dans la scène ; les métadonnées distinguent "characters" (présents actifs) et "referenced" (évoqués)
- Un personnage peut être désigné par un surnom ou un terme de relation ("son frère", "Go") ; utilise le contexte pour résoudre ces références
- Chaque scène porte des tags thématiques (scene_tags) comme : combat, romance, intrigue, révélation, nsfw, deuil, trahison, etc. — utilise-les pour contextualiser le ton et le contenu
- Le lore de l'univers est organisé en : personnages, lieux, événements, objets, cultures, intentions, règles d'univers et axes narratifs
- Un axe narratif est un thème fédérateur (ex : "Affaire Vincourt") qui regroupe des personnages, lieux et événements autour d'un même sujet ; plusieurs axes peuvent se croiser
- Ces conventions sont des guides et peuvent contenir des erreurs ou des cas ambigus
- Les annotations OOC (hors-personnage) ont été retirées
Réponds en français. Sois précis, cite les personnages et les scènes quand c'est pertinent.
Si tu ne trouves pas l'information dans les extraits fournis, dis-le clairement."""


def build_query_engine(top_k: int = 5) -> RetrieverQueryEngine:
    Settings.embed_model = OllamaEmbedding(model_name=EMBED_MODEL)
    Settings.llm = Ollama(model=LLM_MODEL, request_timeout=120.0, system_prompt=SYSTEM_PROMPT)

    client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = client.get_or_create_collection(COLLECTION_NAME)
    vector_store = ChromaVectorStore(chroma_collection=collection)

    index = VectorStoreIndex.from_vector_store(vector_store)
    retriever = VectorIndexRetriever(index=index, similarity_top_k=top_k)

    return RetrieverQueryEngine(
        retriever=retriever,
        node_postprocessors=[SimilarityPostprocessor(similarity_cutoff=0.3)],
    )


def interroger(question: str, query_engine: RetrieverQueryEngine) -> tuple[str, list[dict]]:
    response = query_engine.query(question)

    sources = []
    for node in response.source_nodes:
        meta = node.metadata
        sources.append({
            "arc": meta.get("arc", "?"),
            "channel": meta.get("channel", "?"),
            "start": meta.get("start", "?"),
            "characters": [c for c in meta.get("characters", "").split(",") if c],
            "scene_tags": [t for t in meta.get("scene_tags", "").split(",") if t],
            "score": round(node.score or 0, 3),
        })

    return str(response), sources
