"""
Resets the ChromaDB vector index.

Usage:
  python src/reset_index.py              # asks for confirmation
  python src/reset_index.py --yes        # skips confirmation
  python src/reset_index.py --lore       # also resets config/lore.yaml
"""

import sys
from pathlib import Path

import chromadb

CHROMA_PATH = "data/index"
COLLECTION_NAME = "rp_discord"
LORE_PATH = Path("config/lore.yaml")


def reset_index(skip_confirm: bool = False, reset_lore: bool = False):
    targets = [f"ChromaDB collection '{COLLECTION_NAME}' ({CHROMA_PATH}/)"]
    if reset_lore:
        targets.append(f"lore.yaml ({LORE_PATH})")

    print("The following will be permanently deleted:")
    for t in targets:
        print(f"  • {t}")

    if not skip_confirm:
        answer = input("Confirm? [y/N] ").strip().lower()
        if answer != "y":
            print("Aborted.")
            return

    # Reset ChromaDB collection
    chroma_path = Path(CHROMA_PATH)
    if chroma_path.exists():
        client = chromadb.PersistentClient(path=CHROMA_PATH)
        existing = [c.name for c in client.list_collections()]
        if COLLECTION_NAME in existing:
            client.delete_collection(COLLECTION_NAME)
            print(f"✓ Collection '{COLLECTION_NAME}' deleted.")
        else:
            print(f"  Collection '{COLLECTION_NAME}' not found — nothing to delete.")
    else:
        print(f"  {CHROMA_PATH}/ does not exist — nothing to delete.")

    # Optionally reset lore
    if reset_lore:
        if LORE_PATH.exists():
            LORE_PATH.unlink()
            print(f"✓ {LORE_PATH} deleted.")
        else:
            print(f"  {LORE_PATH} not found — nothing to delete.")

    print("Done. Run indexer to rebuild:")
    print("  python3 src/indexer.py data/exports/")


if __name__ == "__main__":
    reset_index(
        skip_confirm="--yes" in sys.argv,
        reset_lore="--lore" in sys.argv,
    )
