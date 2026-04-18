"""
Indexing Discord exports into ChromaDB via LlamaIndex.
Usage: python src/indexer.py data/exports_filtered/
"""

import sys
from pathlib import Path

import chromadb
from llama_index.core import Document, VectorStoreIndex, StorageContext, Settings
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.vector_stores.chroma import ChromaVectorStore

from preprocessing import load_config, load_messages, build_alias_map, group_raw_into_scenes, raw_scene_to_text
from lore import load_lore
from lore_extractor import merge_into_lore
from scene_synthesizer import synthesize_scene


CHROMA_PATH     = "data/index"
COLLECTION_NAME = "rp_discord"
EMBED_MODEL     = "nomic-embed-text"
LLM_MODEL       = "mistral"


def setup_settings():
    Settings.embed_model = OllamaEmbedding(model_name=EMBED_MODEL)
    Settings.llm = Ollama(model=LLM_MODEL, request_timeout=120.0)


def get_vector_store():
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = client.get_or_create_collection(COLLECTION_NAME)
    return ChromaVectorStore(chroma_collection=collection)


def index_exports(exports_dir: str, extract_lore: bool = False):
    setup_settings()
    config   = load_config()
    lore     = load_lore()
    alias_map = build_alias_map(config)
    alias_map.update(lore.character_aliases())

    exports_path = Path(exports_dir)
    json_files   = list(exports_path.glob("**/*.json")) + list(exports_path.glob("**/*.csv"))

    if not json_files:
        print(f"No exports found in {exports_dir}")
        return

    documents    = []
    total_scenes = 0

    for f_idx, filepath in enumerate(json_files, 1):
        print(f"[{f_idx}/{len(json_files)}] {filepath.name}")
        raw_messages = load_messages(filepath)
        if not raw_messages:
            continue

        channel  = filepath.stem
        arc      = config.get("arcs", {}).get(channel, channel)
        scenes   = group_raw_into_scenes(raw_messages)
        n_scenes = len(scenes)

        for s_idx, scene in enumerate(scenes, 1):
            print(f"  scène {s_idx}/{n_scenes}", end="\r")
            scene_text = raw_scene_to_text(scene, alias_map)
            if len(scene_text.strip()) < 50:
                continue

            synthesis = synthesize_scene(scene_text, alias_map=alias_map, extract_lore=extract_lore)

            if not synthesis["summary"]:
                continue

            start_ts = scene[0].get("timestamp", "")
            end_ts   = scene[-1].get("timestamp", "")

            metadata = {
                "arc":        arc,
                "channel":    channel,
                "start":      start_ts,
                "end":        end_ts,
                "characters": ",".join(synthesis["characters"]),
                "referenced": ",".join(synthesis["referenced"]),
                "location":   str(synthesis["location"] or ""),
                "themes":     ",".join(synthesis["themes"]),
                "scene_text": scene_text[:2000],
            }

            # Document indexé = summary (dense, sémantique) + scène brute pour le contexte RAG
            indexed_text = synthesis["summary"] + "\n\n" + scene_text
            documents.append(Document(text=indexed_text, metadata=metadata))
            total_scenes += 1

            if extract_lore and synthesis["lore"]:
                source    = f"{filepath.stem}:{start_ts}"
                lore_data = _synthesis_lore_to_extracted(synthesis["lore"])
                merge_into_lore(lore_data, source)

        print(f"  {n_scenes} scènes — {len(raw_messages)} messages      ")

    print(f"→ {total_scenes} scènes à indexer depuis {len(json_files)} fichier(s)")

    vector_store    = get_vector_store()
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    VectorStoreIndex.from_documents(documents, storage_context=storage_context, show_progress=True)
    print(f"Indexing complete. Store saved to {CHROMA_PATH}/")


def _synthesis_lore_to_extracted(lore: dict) -> dict:
    """Adapte le lore de la synthèse au format attendu par merge_into_lore."""
    knowledge = {}
    for char, buckets in (lore.get("knowledge") or {}).items():
        knowledge[char] = {
            "sait":        [{"description": d} for d in (buckets.get("sait") or []) if isinstance(d, str)],
            "croit":       [{"description": d} for d in (buckets.get("croit") or []) if isinstance(d, str)],
            "ne_sait_pas": [{"description": d} for d in (buckets.get("ne_sait_pas") or []) if isinstance(d, str)],
        }
    return {
        "relations":          lore.get("relations") or [],
        "universe_rules":     lore.get("facts") or [],
        "character_knowledge": knowledge,
        "characters": {}, "places": {}, "events": {}, "objects": {},
        "cultures": {}, "intentions": {}, "narrative_axes": {},
    }


if __name__ == "__main__":
    exports_dir  = sys.argv[1] if len(sys.argv) > 1 else "data/exports_filtered"
    extract_lore = "--with-lore" in sys.argv
    index_exports(exports_dir, extract_lore=extract_lore)
