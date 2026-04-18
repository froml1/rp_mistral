"""
Indexing Discord exports into ChromaDB via LlamaIndex.
Usage: python src/indexer.py data/exports/
"""

import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import chromadb
from llama_index.core import Document, VectorStoreIndex, StorageContext, Settings
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.vector_stores.chroma import ChromaVectorStore

from preprocessing import load_config, build_alias_map, process_export, group_into_scenes, scene_to_text
from lore import load_lore
from lore_extractor import extract_and_merge
from tagger import tag_scene


CHROMA_PATH = "data/index"
COLLECTION_NAME = "rp_discord"
EMBED_MODEL = "nomic-embed-text"
LLM_MODEL = "mistral"


def setup_settings():
    Settings.embed_model = OllamaEmbedding(model_name=EMBED_MODEL)
    Settings.llm = Ollama(model=LLM_MODEL, request_timeout=120.0)


def scene_text_preview(scene) -> str:
    """Returns a short text preview of a scene for tagging (first 800 chars)."""
    lines = []
    for msg in scene:
        if not msg.is_ooc:
            lines.append(msg.clean_content)
        if sum(len(l) for l in lines) > 800:
            break
    return " ".join(lines)[:800]


def get_vector_store():
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = client.get_or_create_collection(COLLECTION_NAME)
    return ChromaVectorStore(chroma_collection=collection)


def index_exports(exports_dir: str, extract_lore: bool = False, tag_scenes: bool = False):
    setup_settings()
    config = load_config()
    exports_path = Path(exports_dir)
    json_files = list(exports_path.glob("**/*.json")) + list(exports_path.glob("**/*.csv"))

    if not json_files:
        print(f"No exports found in {exports_dir}")
        return

    alias_map = build_alias_map(config)
    lore = load_lore()
    documents = []
    total_scenes = 0

    for f_idx, filepath in enumerate(json_files, 1):
        print(f"[{f_idx}/{len(json_files)}] {filepath.name}")
        messages = process_export(filepath, config, lore=lore)
        scenes   = group_into_scenes(messages)
        n_scenes = len(scenes)

        for s_idx, scene in enumerate(scenes, 1):
            print(f"  scène {s_idx}/{n_scenes}", end="\r")
            stags = tag_scene(scene_text_preview(scene)) if tag_scenes else []
            text, metadata = scene_to_text(scene, alias_map, scene_tags=stags)
            if len(text.strip()) < 50:
                continue
            documents.append(Document(text=text, metadata=metadata))
            total_scenes += 1
            if extract_lore:
                source = f"{filepath.stem}:{metadata.get('start', '')}"
                extract_and_merge(text, source, verbose=True)

        print(f"  {n_scenes} scènes — {len(messages)} messages      ")

    print(f"→ {total_scenes} scènes à indexer depuis {len(json_files)} fichier(s)")

    vector_store = get_vector_store()
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        show_progress=True,
    )

    print(f"Indexing complete. Store saved to {CHROMA_PATH}/")


if __name__ == "__main__":
    exports_dir = sys.argv[1] if len(sys.argv) > 1 else "data/exports"
    extract_lore = "--with-lore" in sys.argv
    tag_scenes = "--with-tags" in sys.argv
    index_exports(exports_dir, extract_lore=extract_lore, tag_scenes=tag_scenes)
