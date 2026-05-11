"""
Step 8 — Lore Synthesis: global entity resolution after all scenes are analyzed.

Reads all analysis/{scene_id}/who.json, where.json, which.json.
Clusters variant names → canonical entities (heuristic + optional LLM pass).
Merges all scene extractions per canonical entity.
Writes final YAMLs to lore/characters/, lore/places/, lore/concepts/.
Updates general_who/where/which + ChromaDB index.

Run AFTER all scenes have been analyzed (step 6).
Can be re-run without re-analyzing scenes.
"""

import json
import re
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from collections import Counter
from llm import call_llm_json
from steps.analyze_entities import _merge_char, _save_char_yaml, _PERSONALITY_AXES, _COMPETENCY_AXES
from steps.analyze_context  import _merge_place, _save_place_yaml
from steps.analyze_which    import _merge_concept, _save_concept_yaml, _slug
from steps.manual_lore   import (
    load_manual_char, load_all_manual_chars, merge_manual_into_char,
    load_manual_place, load_all_manual_places, merge_manual_into_place,
    load_manual_concept, load_all_manual_concepts, merge_manual_into_concept,
)
from steps.general_lore  import update_general_who, update_general_where, update_general_which
from lore_summary        import update_summary

try:
    from store import upsert as _store_upsert, reindex as _store_reindex
except Exception:
    def _store_upsert(*a, **kw): pass
    def _store_reindex(*a, **kw): pass


_COMPETENCY_RANK = {"none": 0, "low": 1, "medium": 2, "high": 3, "exceptional": 4}


# ── Name normalisation & matching ─────────────────────────────────────────────

def _norm(name: str) -> str:
    """Strip parenthetical qualifiers, lowercase, collapse spaces."""
    n = re.sub(r'\s*\([^)]*\)', '', name)
    return re.sub(r'\s+', ' ', n).strip().lower()


def _levenshtein(a: str, b: str) -> int:
    if a == b: return 0
    if not a:  return len(b)
    if not b:  return len(a)
    m, n = len(a), len(b)
    prev = list(range(n + 1))
    for i in range(1, m + 1):
        cur = [i] + [0] * n
        for j in range(1, n + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
        prev = cur
    return prev[n]


def _names_match(a: str, b: str) -> bool:
    """Heuristic: do two names refer to the same entity?"""
    na, nb = _norm(a), _norm(b)
    if na == nb:
        return True
    short, long_ = (na, nb) if len(na) <= len(nb) else (nb, na)
    # Substring containment (min 4 chars to avoid noise)
    if len(short) >= 4 and short in long_:
        return True
    # Edit distance proportional to name length
    if _levenshtein(na, nb) <= max(1, len(short) // 5):
        return True
    # Same forename + one family name is a suffix of the other
    # (handles noble prefixes: "kassel" ↔ "herkassel", "de kassel" ↔ "kassel")
    parts_a, parts_b = na.split(), nb.split()
    if len(parts_a) >= 2 and len(parts_b) >= 2 and parts_a[0] == parts_b[0]:
        last_a = " ".join(parts_a[1:])
        last_b = " ".join(parts_b[1:])
        s, l = (last_a, last_b) if len(last_a) <= len(last_b) else (last_b, last_a)
        if len(s) >= 4 and s in l:
            return True
    return False


# ── Union-Find clustering ─────────────────────────────────────────────────────

class _UF:
    def __init__(self, keys):
        self.p = {k: k for k in keys}

    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b, prefer_a: bool = False):
        ra, rb = self.find(a), self.find(b)
        if ra == rb: return
        if prefer_a:
            self.p[rb] = ra
        else:
            self.p[ra] = rb


def _cluster(
    raw_names: list[str],
    manual_anchors: dict[str, list[str]],
) -> dict[str, list[str]]:
    """
    Returns {canonical_name: [all variant names]}.

    Priority for canonical name election:
      1. Manual anchor (from personnages.yaml / manual config)
      2. Variant with fewest parentheticals (most "base" name)
      3. Most frequent occurrence
    """
    # Build alias → canonical reverse map from manual config
    alias_to_manual: dict[str, str] = {}
    for canonical, aliases in manual_anchors.items():
        alias_to_manual[canonical.lower()] = canonical
        for a in aliases:
            alias_to_manual[a.lower()] = canonical

    unique = list(dict.fromkeys(n.lower() for n in raw_names))
    uf = _UF(unique)

    # First pass: merge names that a manual anchor covers
    for name in unique:
        mc = alias_to_manual.get(name) or alias_to_manual.get(_norm(name))
        if mc:
            mc_l = mc.lower()
            if mc_l not in uf.p:
                uf.p[mc_l] = mc_l
            uf.union(mc_l, name, prefer_a=True)

    # Second pass: heuristic pairwise matching
    for i, a in enumerate(unique):
        for b in unique[i + 1:]:
            if uf.find(a) != uf.find(b) and _names_match(a, b):
                # Prefer the one that is a manual anchor root
                a_is_anchor = bool(alias_to_manual.get(uf.find(a)))
                uf.union(uf.find(a), uf.find(b), prefer_a=a_is_anchor)

    # Build raw clusters: root → members
    raw_clusters: dict[str, list[str]] = {}
    for name in unique:
        root = uf.find(name)
        raw_clusters.setdefault(root, []).append(name)

    # Elect canonical name per cluster
    result: dict[str, list[str]] = {}
    for root, members in raw_clusters.items():
        # Any member covered by a manual anchor?
        canonical = None
        for m in members:
            if mc := alias_to_manual.get(m):
                canonical = mc
                break
        if not canonical:
            # Prefer the member whose _norm() is longest (most descriptive base name)
            # but has no parenthetical (most "pure")
            no_paren = [m for m in members if '(' not in m]
            pool = no_paren if no_paren else members
            canonical = max(pool, key=lambda m: len(_norm(m)))
        result[canonical] = members

    return result


def _build_author_anchors(raw_extractions: dict[str, list[tuple[str, dict]]]) -> dict[str, list[str]]:
    """
    Same Discord author + same first forename token → likely same character.
    Returns extra {canonical: [alias, ...]} to inject into manual_anchors.
    """
    author_to_names: dict[str, list[str]] = {}
    for name_l, entries in raw_extractions.items():
        for _, extraction in entries:
            author = (extraction.get("author") or "").lower().strip()
            if author:
                if name_l not in author_to_names.setdefault(author, []):
                    author_to_names[author].append(name_l)

    anchors: dict[str, list[str]] = {}
    for author, names in author_to_names.items():
        # Group by first token of normalised name
        groups: dict[str, list[str]] = {}
        for name in names:
            first = _norm(name).split()[0] if _norm(name).split() else ""
            if first:
                groups.setdefault(first, []).append(name)
        for group in groups.values():
            if len(group) < 2:
                continue
            # Longest normalised name elected canonical
            canonical = max(group, key=lambda n: len(_norm(n)))
            for v in group:
                if v != canonical:
                    anchors.setdefault(canonical, [])
                    if v not in anchors[canonical]:
                        anchors[canonical].append(v)
    return anchors


def _build_appellation_anchors(raw_extractions: dict[str, list[tuple[str, dict]]]) -> dict[str, list[str]]:
    """
    If name A (canonical) appears as an appellation of name B's extractions → A is an alias of B.
    """
    all_canonical = set(raw_extractions.keys())
    name_to_apps: dict[str, set[str]] = {}
    for name_l, entries in raw_extractions.items():
        for _, extraction in entries:
            for app in (extraction.get("appellations") or []):
                a_l = str(app).lower().strip()
                if len(a_l) >= 4:
                    name_to_apps.setdefault(name_l, set()).add(a_l)

    anchors: dict[str, list[str]] = {}
    for name_b, apps in name_to_apps.items():
        for app in apps:
            if app in all_canonical and app != name_b:
                # app is also a standalone canonical name → merge under the longer one
                canonical = max([name_b, app], key=lambda n: len(_norm(n)))
                alias = app if canonical == name_b else name_b
                anchors.setdefault(canonical, [])
                if alias not in anchors[canonical]:
                    anchors[canonical].append(alias)
    return anchors


# ── LLM disambiguation pass (optional) ───────────────────────────────────────

_DISAMBIG_PROMPT = """\
You are resolving entity identity in a roleplay corpus.

These name variants were found across multiple scenes. Decide which ones refer to the \
SAME entity and which are truly distinct.

Names with their scene contexts:
{entries}

Return a list of groups. Each group contains names that refer to the same entity.
Pick the most complete/canonical name as the group label.

JSON: {{"groups": [{{"canonical": "name", "variants": ["name1", "name2"]}}]}}"""


def _llm_disambiguate(
    ambiguous_clusters: dict[str, list[str]],
    contexts: dict[str, str],
) -> dict[str, list[str]]:
    """
    For clusters where heuristic match is uncertain, ask the LLM.
    `contexts`: name → short description/scene context string.
    Returns refined {canonical: [variants]}.
    """
    if not ambiguous_clusters:
        return ambiguous_clusters

    entries_lines = []
    for canonical, variants in ambiguous_clusters.items():
        all_names = [canonical] + [v for v in variants if v != canonical]
        for n in all_names:
            ctx = contexts.get(n, "")
            entries_lines.append(f"- {n}: {ctx[:80]}")

    result = call_llm_json(
        _DISAMBIG_PROMPT.format(entries="\n".join(entries_lines)),
        num_predict=512,
    )

    refined: dict[str, list[str]] = {}
    for group in (result.get("groups") or []):
        c = group.get("canonical", "")
        v = group.get("variants") or []
        if c:
            refined[c] = list({c.lower()} | {x.lower() for x in v})
    return refined or ambiguous_clusters


# ── Collect extractions from analysis JSONs ───────────────────────────────────

def _load_analysis(analysis_dir: Path, filename: str) -> list[tuple[str, dict]]:
    """Return [(scene_id, data)] for all analysis files of given name."""
    results = []
    for scene_dir in sorted(analysis_dir.iterdir()):
        if not scene_dir.is_dir():
            continue
        fp = scene_dir / filename
        if not fp.exists():
            continue
        try:
            data = json.loads(fp.read_text(encoding="utf-8"))
            results.append((scene_dir.name, data))
        except Exception:
            pass
    return results


# ── Majority-vote merge helpers ───────────────────────────────────────────────

def _majority_merge_chars(all_extractions: list[tuple[str, dict]]) -> dict:
    """
    Merge multiple (scene_id, extraction) tuples into one canonical character dict.
    - Categorical scalars (job, gender, age, species, author, family_name): majority vote
    - Text descriptions: longest non-empty value
    - Lists: union
    - Personality axes: mean over all occurrences
    - Competency axes: keep highest
    - Emotional polarity: aggregate
    """
    merged: dict = {}

    # Categorical scalars: majority vote
    for field in ("job", "gender", "age", "species", "author", "family_name"):
        values = [
            (e.get(field) or "").strip().lower()
            for _, e in all_extractions
            if (e.get(field) or "").strip().lower() not in ("", "unknown")
        ]
        merged[field] = Counter(values).most_common(1)[0][0] if values else ""

    # Text descriptions: longest non-empty
    for field in ("description_physical", "description_psychological"):
        best = ""
        for _, e in all_extractions:
            v = (e.get(field) or "").strip().lower()
            if v and len(v) > len(best):
                best = v
        merged[field] = best

    # List fields: union (preserve insertion order)
    for field in ("appellations", "beliefs", "likes", "dislikes", "main_locations",
                  "affiliations", "wounds", "secrets", "misc"):
        seen: set[str] = set()
        result: list[str] = []
        for _, e in all_extractions:
            for item in (e.get(field) or []):
                s = str(item).lower().strip()
                if s and s not in seen:
                    result.append(s)
                    seen.add(s)
        merged[field] = result

    # Relations: union by (character, relation)
    seen_rels: set[tuple[str, str]] = set()
    rels: list[dict] = []
    for _, e in all_extractions:
        for r in (e.get("relations") or []):
            if not isinstance(r, dict) or not r.get("character"):
                continue
            key = (r["character"].lower(), r.get("relation", "").lower())
            if key not in seen_rels:
                rels.append({"character": r["character"].lower(), "relation": r.get("relation", "").lower()})
                seen_rels.add(key)
    merged["relations"] = rels

    # Goals: union
    goals: dict[str, list] = {"short_term": [], "long_term": []}
    for sub in ("short_term", "long_term"):
        seen_g: set[str] = set()
        for _, e in all_extractions:
            for item in ((e.get("goals") or {}).get(sub) or []):
                s = str(item).lower().strip()
                if s and s not in seen_g:
                    goals[sub].append(s)
                    seen_g.add(s)
    merged["goals"] = goals

    # Personality axes: mean
    axis_sums: dict[str, float] = {}
    axis_counts: dict[str, int] = {}
    for _, e in all_extractions:
        for axis in _PERSONALITY_AXES:
            v = (e.get("personality_axes") or {}).get(axis)
            if v is None:
                continue
            try:
                v = int(v)
                axis_sums[axis] = axis_sums.get(axis, 0.0) + v
                axis_counts[axis] = axis_counts.get(axis, 0) + 1
            except (TypeError, ValueError):
                pass
    merged["personality_axes"] = {
        axis: round(axis_sums[axis] / axis_counts[axis])
        for axis in axis_sums
    }

    # Competency axes: keep highest
    comp: dict[str, str] = {}
    for _, e in all_extractions:
        for axis in _COMPETENCY_AXES:
            v = ((e.get("competency_axes") or {}).get(axis) or "").lower()
            if not v:
                continue
            if axis not in comp or _COMPETENCY_RANK.get(v, 0) > _COMPETENCY_RANK.get(comp[axis], 0):
                comp[axis] = v
    merged["competency_axes"] = comp

    # Emotional polarity
    emotions: list[str] = []
    triggers: list[str] = []
    ranges: list[str] = []
    for _, e in all_extractions:
        ep = e.get("emotional_polarity") or {}
        emotions.extend((ep.get("dominant_emotions") or []))
        triggers.extend((ep.get("emotional_triggers") or []))
        if ep.get("emotional_range"):
            ranges.append(ep["emotional_range"])
    merged["emotional_polarity"] = {
        "dominant_emotions": list(dict.fromkeys(str(x).lower() for x in emotions))[-5:],
        "emotional_range":   Counter(ranges).most_common(1)[0][0] if ranges else "moderate",
        "emotional_triggers": list(dict.fromkeys(str(x).lower() for x in triggers)),
    }

    # Appearances
    scene_ids = [sid for sid, _ in all_extractions]
    merged["appearances"] = sorted(set(scene_ids))
    merged["first_appearance"] = min(scene_ids) if scene_ids else ""

    return merged


def _majority_merge_places(all_extractions: list[tuple[str, dict]]) -> dict:
    """
    Merge multiple (scene_id, extraction) tuples into one canonical place dict.
    - Categorical scalars (type, atmosphere, control, access): majority vote
    - description: longest non-empty
    - Lists: union
    """
    merged: dict = {}

    for field in ("type", "atmosphere", "control", "access"):
        values = [
            (e.get(field) or "").strip().lower()
            for _, e in all_extractions
            if (e.get(field) or "").strip().lower() not in ("", "unknown", "other")
        ]
        merged[field] = Counter(values).most_common(1)[0][0] if values else ""

    best_desc = ""
    for _, e in all_extractions:
        v = (e.get("description") or "").strip().lower()
        if v and len(v) > len(best_desc):
            best_desc = v
    merged["description"] = best_desc

    for field in ("appellations", "attributes", "known_inhabitants"):
        seen: set[str] = set()
        result: list[str] = []
        for _, e in all_extractions:
            for item in (e.get(field) or []):
                s = str(item).lower().strip()
                if s and s not in seen:
                    result.append(s)
                    seen.add(s)
        merged[field] = result

    scene_ids = [sid for sid, _ in all_extractions]
    merged["appearances"] = sorted(set(scene_ids))
    merged["first_appearance"] = min(scene_ids) if scene_ids else ""

    return merged


# ── Entity synthesis: chars ───────────────────────────────────────────────────

def _synthesize_chars(
    analysis_dir: Path,
    chars_dir: Path,
    lore_dir: Path,
    with_llm_disambig: bool = False,
) -> int:
    print("  [who] collecting character extractions…")
    all_entries = _load_analysis(analysis_dir, "who.json")
    manual_chars = load_all_manual_chars()

    # Gather all raw names and their extraction data
    # raw_extractions: name_lower → list of (scene_id, extraction_dict)
    raw_extractions: dict[str, list[tuple[str, dict]]] = {}
    for scene_id, data in all_entries:
        for char in (data.get("details") or []):
            if not isinstance(char, dict) or not char.get("canonical_name"):
                continue
            name_l = char["canonical_name"].lower()
            raw_extractions.setdefault(name_l, []).append((scene_id, char))

    if not raw_extractions:
        print("  [who] no extractions found")
        return 0

    # Extend manual anchors with data-derived signals
    manual_anchors = {k: (v.get("appellations") or []) for k, v in manual_chars.items()}
    author_anch = _build_author_anchors(raw_extractions)
    app_anch    = _build_appellation_anchors(raw_extractions)
    for extra in (author_anch, app_anch):
        for k, vs in extra.items():
            if k in manual_anchors:
                manual_anchors[k] = list(dict.fromkeys(manual_anchors[k] + vs))
            else:
                manual_anchors[k] = vs

    clusters = _cluster(list(raw_extractions.keys()), manual_anchors)
    print(f"  [who] {len(raw_extractions)} raw names → {len(clusters)} entities")

    # Optional LLM disambiguation on uncertain clusters
    if with_llm_disambig:
        manual_canonical_set = {k.lower() for k in manual_anchors}
        uncertain = {
            c: vs for c, vs in clusters.items()
            if len(vs) > 1 and c.lower() not in manual_canonical_set
        }
        if uncertain:
            contexts = {
                n: (raw_extractions[n][0][1].get("job") or "")
                for n in raw_extractions
            }
            clusters = {**clusters, **_llm_disambiguate(uncertain, contexts)}

    # Merge all clusters using majority-vote
    final_chars: dict[str, dict] = {}
    for canonical, variants in clusters.items():
        all_extractions = []
        for v in variants:
            all_extractions.extend(raw_extractions.get(v, []))

        if not all_extractions:
            continue

        merged = _majority_merge_chars(all_extractions)
        merged["name"] = canonical
        merged["aliases"] = sorted({v for v in variants if v != canonical.lower()})
        merged = merge_manual_into_char(merged, load_manual_char(canonical))
        final_chars[canonical] = merged

    # Remove appellations that are another character's canonical name
    all_canonicals_lower = {c.lower() for c in final_chars}
    for canonical, char_data in final_chars.items():
        own_lower = canonical.lower()
        cleaned = [
            a for a in (char_data.get("appellations") or [])
            if a.lower() == own_lower or a.lower() not in all_canonicals_lower
        ]
        if len(cleaned) != len(char_data.get("appellations") or []):
            removed = set(char_data.get("appellations", [])) - set(cleaned)
            print(f"    [dedup] {canonical}: removed cross-contaminated appellations {removed}")
        char_data["appellations"] = cleaned

    count = 0
    for canonical, merged in final_chars.items():
        _save_char_yaml(chars_dir, canonical, merged)
        count += 1

    print(f"  [who] {count} canonical character(s) written")
    update_general_who(chars_dir, lore_dir)
    return count


# ── Entity synthesis: places ──────────────────────────────────────────────────

def _synthesize_places(
    analysis_dir: Path,
    places_dir: Path,
    lore_dir: Path,
    with_llm_disambig: bool = False,
) -> int:
    print("  [where] collecting place extractions…")
    all_entries = _load_analysis(analysis_dir, "where.json")
    manual_places = load_all_manual_places()

    raw_extractions: dict[str, list[tuple[str, dict]]] = {}
    for scene_id, data in all_entries:
        for loc in (data.get("details") or []):
            if not isinstance(loc, dict) or not loc.get("canonical_name"):
                continue
            name_l = loc["canonical_name"].lower()
            raw_extractions.setdefault(name_l, []).append((scene_id, loc))

    if not raw_extractions:
        print("  [where] no extractions found")
        return 0

    # Extend manual anchors with appellation-based signals (no author for places)
    manual_anchors = {k: (v.get("appellations") or []) for k, v in manual_places.items()}
    app_anch = _build_appellation_anchors(raw_extractions)
    for k, vs in app_anch.items():
        if k in manual_anchors:
            manual_anchors[k] = list(dict.fromkeys(manual_anchors[k] + vs))
        else:
            manual_anchors[k] = vs

    clusters = _cluster(list(raw_extractions.keys()), manual_anchors)
    print(f"  [where] {len(raw_extractions)} raw names → {len(clusters)} places")

    count = 0
    for canonical, variants in clusters.items():
        all_extractions = []
        for v in variants:
            all_extractions.extend(raw_extractions.get(v, []))
        if not all_extractions:
            continue
        merged = _majority_merge_places(all_extractions)
        merged["name"] = canonical
        merged["aliases"] = sorted({v for v in variants if v != canonical.lower()})
        merged = merge_manual_into_place(merged, load_manual_place(canonical))
        _save_place_yaml(places_dir, canonical, merged)
        count += 1

    print(f"  [where] {count} canonical place(s) written")
    update_general_where(places_dir, lore_dir)
    return count


# ── Entity synthesis: concepts ────────────────────────────────────────────────

def _synthesize_concepts(
    analysis_dir: Path,
    concepts_dir: Path,
    lore_dir: Path,
    with_llm_disambig: bool = False,
) -> int:
    print("  [which] collecting concept extractions…")
    all_entries = _load_analysis(analysis_dir, "which.json")
    manual_concepts = load_all_manual_concepts()

    raw_extractions: dict[str, list[tuple[str, dict]]] = {}
    for scene_id, data in all_entries:
        for concept in (data.get("details") or []):
            if not isinstance(concept, dict) or not concept.get("canonical_name"):
                continue
            name_l = concept["canonical_name"].lower()
            raw_extractions.setdefault(name_l, []).append((scene_id, concept))

    if not raw_extractions:
        print("  [which] no extractions found")
        return 0

    manual_anchors = {k: (v.get("appellations") or []) for k, v in manual_concepts.items()}
    clusters = _cluster(list(raw_extractions.keys()), manual_anchors)
    print(f"  [which] {len(raw_extractions)} raw names → {len(clusters)} concepts")

    count = 0
    for canonical, variants in clusters.items():
        merged: dict = {}
        for v in variants:
            for scene_id, extraction in raw_extractions.get(v, []):
                merged = _merge_concept(merged, extraction, scene_id)

        if not merged:
            continue

        merged["name"] = canonical
        merged["aliases"] = sorted({v for v in variants if v != canonical.lower()})
        merged = merge_manual_into_concept(merged, load_manual_concept(canonical))
        _save_concept_yaml(concepts_dir, canonical, merged)

        summary = update_summary(concepts_dir / f"{_slug(canonical)}.yaml", "concepts")
        _store_upsert("concepts", canonical, summary)
        count += 1

    print(f"  [which] {count} canonical concept(s) written")
    update_general_which(concepts_dir, lore_dir)
    return count


# ── Main entry point ──────────────────────────────────────────────────────────

def run_lore_synthesis(
    analysis_dir: Path,
    lore_dir: Path,
    with_llm_disambig: bool = False,
) -> dict:
    """
    Build canonical lore from all scene analyses.
    Safe to re-run: overwrites existing lore YAMLs with fresher merged data.
    """
    chars_dir    = lore_dir / "characters"
    places_dir   = lore_dir / "places"
    concepts_dir = lore_dir / "concepts"

    chars_dir.mkdir(parents=True, exist_ok=True)
    places_dir.mkdir(parents=True, exist_ok=True)
    concepts_dir.mkdir(parents=True, exist_ok=True)

    n_chars    = _synthesize_chars(analysis_dir, chars_dir, lore_dir, with_llm_disambig)
    n_places   = _synthesize_places(analysis_dir, places_dir, lore_dir, with_llm_disambig)
    n_concepts = _synthesize_concepts(analysis_dir, concepts_dir, lore_dir, with_llm_disambig)

    print("  [store] rebuilding index…")
    try:
        _store_reindex(lore_dir)
    except Exception as e:
        print(f"  [store] warning — {e}")

    stats = {"chars": n_chars, "places": n_places, "concepts": n_concepts}
    print(f"  [synthesis] done — {n_chars} chars, {n_places} places, {n_concepts} concepts")
    return stats


if __name__ == "__main__":
    import argparse

    ROOT = Path(__file__).resolve().parent.parent.parent
    parser = argparse.ArgumentParser(description="Build canonical lore from all scene analyses.")
    parser.add_argument("--analysis-dir", default=str(ROOT / "data" / "analysis"))
    parser.add_argument("--lore-dir",     default=str(ROOT / "data" / "lore"))
    parser.add_argument("--with-llm",     action="store_true",
                        help="Use LLM to resolve ambiguous entity clusters")
    args = parser.parse_args()

    run_lore_synthesis(
        analysis_dir=Path(args.analysis_dir),
        lore_dir=Path(args.lore_dir),
        with_llm_disambig=args.with_llm,
    )
