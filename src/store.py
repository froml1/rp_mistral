"""
ChromaDB vector store for lore entities.

- Embeds entity summary cards via Ollama nomic-embed-text
- Enables semantic retrieval: top-K entities relevant to a natural-language query
- Auto-updated when the pipeline saves/updates a YAML
- Full reindex available (step 5 / manual)
"""

import re
import sys
from pathlib import Path

import requests
import yaml
import chromadb
from chromadb.config import Settings

ROOT     = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
LORE_DIR = DATA_DIR / "lore"
DB_DIR   = DATA_DIR / "chroma"

EMBED_URL   = "http://localhost:11434/api/embeddings"
EMBED_MODEL = "nomic-embed-text"

_CATEGORIES = ("characters", "places", "concepts")

# ── Ollama embedding ───────────────────────────────────────────────────────────

def _embed(text: str) -> list[float]:
    try:
        resp = requests.post(
            EMBED_URL,
            json={"model": EMBED_MODEL, "prompt": text},
            timeout=60,
        )
        resp.raise_for_status()
        return resp.json().get("embedding", [])
    except Exception as e:
        print(f"  [store] embed error: {e}", file=sys.stderr)
        return []


# ── ChromaDB client ───────────────────────────────────────────────────────────

def _client() -> chromadb.Client:
    DB_DIR.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(DB_DIR))


def _collection(client: chromadb.Client, category: str):
    return client.get_or_create_collection(
        name=f"lore_{category}",
        metadata={"hnsw:space": "cosine"},
    )


# ── Public API ─────────────────────────────────────────────────────────────────

def upsert(category: str, name: str, summary: str, extra_meta: dict | None = None):
    """Add or update an entity in the vector store."""
    if not summary or not name:
        return
    embedding = _embed(summary)
    if not embedding:
        return
    slug = re.sub(r'[^\w]', '_', name.lower().strip())
    meta = {"name": name, "category": category, **(extra_meta or {})}
    client = _client()
    col = _collection(client, category)
    col.upsert(
        ids=[slug],
        embeddings=[embedding],
        documents=[summary],
        metadatas=[meta],
    )


def search(query: str, categories: list[str] | None = None, n: int = 8) -> list[dict]:
    """
    Semantic search across lore categories.
    Returns list of {name, category, summary, score} sorted by relevance.
    """
    q_vec = _embed(query)
    if not q_vec:
        return []

    cats = categories or list(_CATEGORIES)
    client = _client()
    results = []

    for cat in cats:
        col = _collection(client, cat)
        try:
            res = col.query(
                query_embeddings=[q_vec],
                n_results=min(n, col.count() or 1),
                include=["documents", "metadatas", "distances"],
            )
            for doc, meta, dist in zip(
                res["documents"][0],
                res["metadatas"][0],
                res["distances"][0],
            ):
                results.append({
                    "name":     meta.get("name", "?"),
                    "category": cat,
                    "summary":  doc,
                    "score":    1.0 - dist,  # cosine similarity
                })
        except Exception:
            pass

    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:n]


def search_chars(query: str, n: int = 6) -> list[dict]:
    return search(query, categories=["characters"], n=n)


def search_places(query: str, n: int = 4) -> list[dict]:
    return search(query, categories=["places"], n=n)


def search_concepts(query: str, n: int = 4) -> list[dict]:
    return search(query, categories=["concepts"], n=n)


def reindex(lore_dir: Path | None = None):
    """Rebuild the full index from all lore YAMLs. Safe to run anytime."""
    from lore_summary import _MAKERS, make_char_summary, make_place_summary, make_concept_summary
    ld = lore_dir or LORE_DIR
    total = 0
    for cat in _CATEGORIES:
        d = ld / cat
        if not d.exists():
            continue
        for yf in d.glob("*.yaml"):
            try:
                data = yaml.safe_load(yf.read_text(encoding="utf-8")) or {}
                name = data.get("name")
                if not name:
                    continue
                summary = data.get("_summary") or _MAKERS[cat](data)
                upsert(cat, name, summary, extra_meta={"slug": yf.stem})
                total += 1
                print(f"  [store] indexed {cat}/{name}", flush=True)
            except Exception as e:
                print(f"  [store] error {yf.name}: {e}", file=sys.stderr)
    print(f"  [store] reindex done — {total} entities")
    return total


def count() -> dict[str, int]:
    client = _client()
    return {cat: _collection(client, cat).count() for cat in _CATEGORIES}
