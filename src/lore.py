"""
World knowledge graph loader and query interface.
Reads config/lore.yaml and provides resolution helpers for preprocessing.
"""

from pathlib import Path
from collections import defaultdict
import yaml


# French relational terms → relation type
RELATION_PATTERNS = [
    ("frère",   "frère_de"),
    ("sœur",    "sœur_de"),
    ("père",    "parent_de"),
    ("mère",    "parent_de"),
    ("fils",    "enfant_de"),
    ("fille",   "enfant_de"),
    ("oncle",   "parent_de"),
    ("tante",   "parent_de"),
    ("ami",     "ami_de"),
    ("amie",    "ami_de"),
    ("ennemi",  "ennemi_de"),
    ("ennemie", "ennemi_de"),
    ("rival",   "rival_de"),
    ("rivale",  "rival_de"),
    ("allié",   "allié_de"),
    ("alliée",  "allié_de"),
    ("amant",   "amant_de"),
    ("amante",  "amant_de"),
    ("maître",  "maître_de"),
    ("élève",   "élève_de"),
]

RELATION_TERM_TO_TYPE: dict[str, str] = {term: rel for term, rel in RELATION_PATTERNS}

# Canonical temporal state ordering (for degradation logic)
STATE_ORDER = {"sera": 0, "est": 1, "fut": 2}


class LoreGraph:
    def __init__(self, lore_path: str = "config/lore.yaml"):
        path = Path(lore_path)
        if not path.exists():
            self._data: dict = {}
            self._edges: list[dict] = []
        else:
            with open(path, encoding="utf-8") as f:
                self._data = yaml.safe_load(f) or {}
            self._edges = self._data.get("relations", []) or []

        self._from_index: dict[str, list[dict]] = defaultdict(list)
        self._to_index: dict[str, list[dict]] = defaultdict(list)
        for edge in self._edges:
            self._from_index[edge["from"]].append(edge)
            self._to_index[edge["to"]].append(edge)

    # ── node accessors ──────────────────────────────────────────────────────

    def characters(self, state: str | None = None) -> dict:
        return self._filter_by_state(self._data.get("characters") or {}, state)

    def places(self, state: str | None = None) -> dict:
        return self._filter_by_state(self._data.get("places") or {}, state)

    def events(self, state: str | None = None) -> dict:
        return self._filter_by_state(self._data.get("events") or {}, state)

    def objects(self, state: str | None = None) -> dict:
        return self._filter_by_state(self._data.get("objects") or {}, state)

    def cultures(self, state: str | None = None) -> dict:
        return self._filter_by_state(self._data.get("cultures") or {}, state)

    def intentions(self, state: str | None = None) -> dict:
        return self._filter_by_state(self._data.get("intentions") or {}, state)

    def narrative_axes(self, state: str | None = None) -> dict:
        return self._filter_by_state(self._data.get("narrative_axes") or {}, state)

    def universe_rules(self) -> list:
        return self._data.get("universe_rules") or []

    def character_knowledge(self, character: str) -> dict:
        """Returns {sait: [...], ne_sait_pas: [...], croit: [...]} for a character."""
        ck = self._data.get("character_knowledge") or {}
        return ck.get(character) or {"sait": [], "ne_sait_pas": [], "croit": []}

    def all_entity_names(self, state: str | None = None) -> set[str]:
        result: set[str] = set()
        for section in ("characters", "places", "events", "objects", "cultures",
                        "intentions", "narrative_axes"):
            nodes = self._filter_by_state(self._data.get(section) or {}, state)
            result.update(nodes.keys())
        return result

    def axes_for_entity(self, entity_name: str, state: str | None = None) -> list[str]:
        """Returns axis IDs that include the given entity, optionally filtered by state."""
        result = []
        for axis_id, axis_data in self.narrative_axes(state).items():
            if not isinstance(axis_data, dict):
                continue
            for field in ("characters", "places", "events", "objects", "cultures", "intentions"):
                if entity_name in (axis_data.get(field) or []):
                    result.append(axis_id)
                    break
        return result

    # ── edge queries ────────────────────────────────────────────────────────

    def edges_from(self, entity: str, rel: str | None = None,
                   state: str | None = None) -> list[dict]:
        edges = self._from_index.get(entity, [])
        return [
            e for e in edges
            if (rel is None or e["rel"] == rel)
            and (state is None or e.get("state", "est") == state)
        ]

    def edges_to(self, entity: str, rel: str | None = None,
                 state: str | None = None) -> list[dict]:
        edges = self._to_index.get(entity, [])
        return [
            e for e in edges
            if (rel is None or e["rel"] == rel)
            and (state is None or e.get("state", "est") == state)
        ]

    def neighbors(self, entity: str, rel: str | None = None,
                  state: str | None = None) -> list[str]:
        return [e["to"] for e in self.edges_from(entity, rel, state)]

    def reverse_neighbors(self, entity: str, rel: str | None = None,
                          state: str | None = None) -> list[str]:
        return [e["from"] for e in self.edges_to(entity, rel, state)]

    # ── resolution helpers ─────────────────────────────────────────────────

    def resolve_relation_term(self, term: str, context_character: str,
                              state: str = "est") -> str | None:
        """
        Resolves a relational noun to a character name given a context character.
        Only considers relations in the given state (default: currently active).

        Example: term="frère", context_character="Garrance", state="est"
        → returns Gaulthier if an est-state frère_de edge exists.
        """
        rel_type = RELATION_TERM_TO_TYPE.get(term.lower().rstrip("se"))
        if not rel_type:
            return None

        candidates = self.reverse_neighbors(context_character, rel_type, state)
        if len(candidates) == 1:
            return candidates[0]

        candidates = self.neighbors(context_character, rel_type, state)
        if len(candidates) == 1:
            return candidates[0]

        return None

    def character_aliases(self) -> dict[str, str]:
        result: dict[str, str] = {}
        for name, data in self.characters().items():
            if isinstance(data, dict):
                for alias in data.get("aliases", []):
                    result[alias] = name
        return result

    def what_character_knows(self, character: str) -> list[str]:
        """Returns descriptions from the 'sait' bucket."""
        return [item.get("description", "") for item in self.character_knowledge(character).get("sait", [])]

    def what_character_believes(self, character: str) -> list[dict]:
        """Returns belief entries from the 'croit' bucket."""
        return self.character_knowledge(character).get("croit", [])

    def what_character_ignores(self, character: str) -> list[str]:
        """Returns descriptions from the 'ne_sait_pas' bucket."""
        return [item.get("description", "") for item in self.character_knowledge(character).get("ne_sait_pas", [])]

    # ── helpers ─────────────────────────────────────────────────────────────

    @staticmethod
    def _filter_by_state(nodes: dict, state: str | None) -> dict:
        if state is None:
            return nodes
        return {k: v for k, v in nodes.items()
                if (v or {}).get("state", "est") == state}


def load_lore(lore_path: str = "config/lore.yaml") -> LoreGraph:
    return LoreGraph(lore_path)
