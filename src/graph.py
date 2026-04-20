"""
Social graph builder from how.json character_relations.

Generates:
  - Social graph image (all characters, weighted edges by relation frequency)
  - Relationship evolution chart (two characters over time)
  - PC/NPC heuristic (author diversity per character)

Usage:
  python src/graph.py --social
  python src/graph.py --evolution "rhys" "lena"
  python src/graph.py --pcnpc
"""

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import yaml

ROOT         = Path(__file__).resolve().parent.parent
ANALYSIS_DIR = ROOT / "data" / "analysis"
LORE_DIR     = ROOT / "data" / "lore"
SCENES_DIR   = ROOT / "data" / "scenes"
OUT_DIR      = ROOT / "data" / "graphs"


def _load_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    except Exception:
        return {}


def _load_yaml(path: Path) -> dict:
    try:
        return yaml.safe_load(path.read_text(encoding="utf-8")) or {} if path.exists() else {}
    except Exception:
        return {}


def _slug(name: str) -> str:
    return re.sub(r'\s+', '_', name.lower().strip())


# ── Social graph ──────────────────────────────────────────────────────────────

_SENTIMENT_COLOR = {
    "trust":       "#2ecc71",
    "admiration":  "#27ae60",
    "love":        "#e74c3c",
    "friendship":  "#3498db",
    "ally":        "#2980b9",
    "indifference":"#95a5a6",
    "neutral":     "#bdc3c7",
    "distrust":    "#e67e22",
    "fear":        "#e74c3c",
    "resentment":  "#c0392b",
    "rivalry":     "#8e44ad",
    "hatred":      "#922b21",
}

_RELATION_STYLE = {
    "ally":       "solid",
    "friend":     "solid",
    "romantic":   "solid",
    "family":     "solid",
    "mentor":     "dashed",
    "subordinate":"dashed",
    "rival":      "dotted",
    "enemy":      "dotted",
    "stranger":   "dotted",
    "neutral":    "dotted",
}


def build_social_graph(output_path: Path | None = None) -> Path:
    try:
        import networkx as nx
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except ImportError:
        print("[error] pip install networkx matplotlib")
        sys.exit(1)

    # Aggregate relations from all how.json
    edges: dict[tuple, dict] = {}  # (from, to) → {count, relation_type, sentiments}

    for scene_dir in sorted(ANALYSIS_DIR.iterdir()):
        if not scene_dir.is_dir():
            continue
        how = _load_json(scene_dir / "how.json")
        for rel in (how.get("character_relations") or []):
            frm  = (rel.get("from_char") or "").lower().strip()
            to   = (rel.get("to_char") or "").lower().strip()
            if not frm or not to or frm == to:
                continue
            key = tuple(sorted([frm, to]))
            if key not in edges:
                edges[key] = {"count": 0, "relation_types": [], "sentiments": []}
            edges[key]["count"] += 1
            if rel.get("relation_type"):
                edges[key]["relation_types"].append(rel["relation_type"].lower())
            if rel.get("sentiment"):
                edges[key]["sentiments"].append(rel["sentiment"].lower())

    if not edges:
        print("[warn] No character relations found in how.json files.")
        return None

    G = nx.Graph()
    for (frm, to), data in edges.items():
        G.add_edge(frm, to, weight=data["count"],
                   relation=data["relation_types"][0] if data["relation_types"] else "neutral",
                   sentiment=data["sentiments"][0] if data["sentiments"] else "neutral")

    # Node sizes: degree (number of relations)
    degrees  = dict(G.degree())
    max_deg  = max(degrees.values()) if degrees else 1
    sizes    = [300 + 1500 * (degrees[n] / max_deg) for n in G.nodes()]

    # Edge colors from sentiment
    edge_colors = [
        _SENTIMENT_COLOR.get(G[u][v].get("sentiment", "neutral"), "#95a5a6")
        for u, v in G.edges()
    ]
    edge_widths = [max(1, G[u][v]["weight"] * 0.8) for u, v in G.edges()]

    fig, ax = plt.subplots(figsize=(14, 10))
    pos = nx.spring_layout(G, seed=42, k=2.0)

    nx.draw_networkx_nodes(G, pos, node_size=sizes, node_color="#3498db", alpha=0.85, ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=8, font_color="white", font_weight="bold", ax=ax)
    nx.draw_networkx_edges(G, pos, edge_color=edge_colors, width=edge_widths, alpha=0.7, ax=ax)

    # Legend
    patches = [
        mpatches.Patch(color=col, label=sent)
        for sent, col in list(_SENTIMENT_COLOR.items())[:8]
    ]
    ax.legend(handles=patches, loc="lower left", fontsize=7, title="Sentiment")
    ax.set_title(f"Character Social Graph — {G.number_of_nodes()} characters, {G.number_of_edges()} relations")
    ax.axis("off")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = output_path or OUT_DIR / "social_graph.png"
    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Social graph saved → {out}")
    return out


# ── Relationship evolution ─────────────────────────────────────────────────────

_SENTIMENT_SCORE = {
    "love": 3, "trust": 2, "admiration": 2, "friendship": 2, "ally": 1,
    "neutral": 0, "indifference": 0,
    "distrust": -1, "rivalry": -1, "fear": -2, "resentment": -2, "hatred": -3,
}


def build_relation_evolution(char_a: str, char_b: str, output_path: Path | None = None) -> Path:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[error] pip install matplotlib")
        sys.exit(1)

    char_a, char_b = char_a.lower().strip(), char_b.lower().strip()
    data_points: list[tuple[str, str, str, float]] = []  # (scene, relation, sentiment, score)

    for scene_dir in sorted(ANALYSIS_DIR.iterdir()):
        if not scene_dir.is_dir():
            continue
        how = _load_json(scene_dir / "how.json")
        for rel in (how.get("character_relations") or []):
            frm  = (rel.get("from_char") or "").lower()
            to   = (rel.get("to_char") or "").lower()
            if not ((char_a in frm or char_a in to) and (char_b in frm or char_b in to)):
                continue
            sentiment = (rel.get("sentiment") or "neutral").lower()
            score     = _SENTIMENT_SCORE.get(sentiment, 0)
            data_points.append((scene_dir.name, rel.get("relation_type", "?"), sentiment, score))

    if not data_points:
        return None

    scenes    = [d[0] for d in data_points]
    scores    = [d[3] for d in data_points]
    relations = [d[1] for d in data_points]
    sentiments= [d[2] for d in data_points]

    fig, ax = plt.subplots(figsize=(12, 5))
    colors = ["#2ecc71" if s >= 1 else "#e74c3c" if s <= -1 else "#95a5a6" for s in scores]

    ax.bar(range(len(scenes)), scores, color=colors, alpha=0.8)
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xticks(range(len(scenes)))
    ax.set_xticklabels([s.split("_")[-1] for s in scenes], rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("Sentiment score")
    ax.set_title(f"Relation evolution: {char_a} ↔ {char_b} ({len(data_points)} interactions)")

    for i, (rel, sent) in enumerate(zip(relations, sentiments)):
        ax.annotate(f"{rel}\n{sent}", (i, scores[i]),
                    textcoords="offset points", xytext=(0, 4),
                    ha="center", fontsize=6, rotation=30)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = output_path or OUT_DIR / f"relation_{_slug(char_a)}_{_slug(char_b)}.png"
    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Relation evolution saved → {out}")
    return out


# ── PC / NPC heuristic ─────────────────────────────────────────────────────────

def build_pcnpc_heuristic() -> list[dict]:
    """
    Heuristic: a character played by many different authors = likely NPC.
    A character always associated with the same author = likely PC.
    """
    char_authors: dict[str, set] = defaultdict(set)

    for sf in SCENES_DIR.glob("**/*.json"):
        try:
            data = json.loads(sf.read_text(encoding="utf-8"))
        except Exception:
            continue
        msgs = data.get("messages", [])
        scene_id = data.get("scene_id", sf.stem)

        # Load who.json for this scene
        who = _load_json(ANALYSIS_DIR / scene_id / "who.json")
        chars_in_scene = [c.lower() for c in (who.get("characters") or [])]

        # Map each author to characters in the same scene
        for msg in msgs:
            author = msg.get("author") or {}
            author_name = (author.get("name", "") if isinstance(author, dict) else str(author)).lower()
            if author_name:
                for char in chars_in_scene:
                    char_authors[char].add(author_name)

    results = []
    for char, authors in sorted(char_authors.items()):
        n_authors = len(authors)
        if n_authors == 0:
            continue
        role = "NPC" if n_authors >= 3 else "PC (likely)" if n_authors == 1 else "PC/NPC (unclear)"
        results.append({
            "character":  char,
            "authors":    n_authors,
            "role":       role,
            "author_list": sorted(authors),
        })

    results.sort(key=lambda x: x["authors"], reverse=True)
    return results


# ── Tension curve ─────────────────────────────────────────────────────────────

_EV_SCORE = {"revelation": 3.0, "decision": 2.0, "emotional": 2.0, "action": 1.0, "conversation": 0.3}
_SENT_NEG  = {"fear", "resentment", "hatred", "distrust", "rivalry"}


def build_tension_curve(output_path: Path | None = None) -> Path:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("[error] pip install matplotlib numpy")
        sys.exit(1)

    scenes_ordered = sorted(
        (d for d in ANALYSIS_DIR.iterdir() if d.is_dir()),
        key=lambda d: d.name,
    )
    if not scenes_ordered:
        print("[warn] No scenes found.")
        return None

    scene_ids, scores = [], []
    for sd in scenes_ordered:
        what = _load_json(sd / "what.json")
        how  = _load_json(sd / "how.json")

        score = sum(_EV_SCORE.get(ev.get("type",""), 0.3) for ev in (what.get("events") or []))
        score += sum(
            1.5 for r in (how.get("character_relations") or [])
            if (r.get("sentiment","").lower() in _SENT_NEG)
        )
        score += len(how.get("links") or []) * 0.4
        scene_ids.append(sd.name.split("_")[-1])  # short label
        scores.append(score)

    # Smoothed curve
    window = max(1, len(scores) // 8)
    smoothed = np.convolve(scores, np.ones(window) / window, mode='same') if len(scores) > window else scores

    fig, ax = plt.subplots(figsize=(14, 5))
    x = range(len(scores))
    ax.fill_between(x, scores, alpha=0.2, color="#e74c3c")
    ax.plot(x, scores,   color="#e74c3c", alpha=0.5, linewidth=0.8, label="raw")
    ax.plot(x, smoothed, color="#c0392b", linewidth=2.5, label="trend")

    # Mark peaks
    if len(scores) > 2:
        peak_idx = int(np.argmax(smoothed))
        ax.axvline(peak_idx, color="#8e44ad", linestyle="--", alpha=0.6)
        ax.annotate(f"peak\n{scenes_ordered[peak_idx].name}",
                    (peak_idx, smoothed[peak_idx]),
                    textcoords="offset points", xytext=(6, 6),
                    fontsize=7, color="#8e44ad")

    ax.set_xticks(list(x)[::max(1, len(scores)//12)])
    ax.set_xticklabels(scene_ids[::max(1, len(scores)//12)], rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("Tension score")
    ax.set_title(f"Narrative tension curve — {len(scores)} scenes")
    ax.legend(fontsize=8)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = output_path or OUT_DIR / "tension_curve.png"
    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Tension curve saved → {out}")
    return out


# ── Power map ─────────────────────────────────────────────────────────────────

_COMP_POWER = {"exceptional": 4, "high": 3, "medium": 2, "low": 1, "none": 0}
_LINK_POWER = {"controls": 3, "commands": 3, "leads_to": 1, "enables": 1,
               "opposes": -1, "prevents": -1}
_REL_POWER  = {"mentor": 2, "leader": 2, "subordinate": -2,
               "ally": 1, "enemy": -1, "rival": -0.5}


def build_power_map(output_path: Path | None = None) -> Path:
    try:
        import networkx as nx
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[error] pip install networkx matplotlib")
        sys.exit(1)

    # Base power from competency axes
    char_power: dict[str, float] = {}
    chars_dir = LORE_DIR / "characters"
    if chars_dir.exists():
        for yf in chars_dir.glob("*.yaml"):
            d = _load_yaml(yf)
            name = d.get("name","")
            if not name:
                continue
            axes = d.get("competency_axes") or {}
            score = sum(
                _COMP_POWER.get(str(v).lower(), 0)
                for k, v in axes.items()
                if v and k in ("combat_melee", "leadership", "intimidation", "magic_power")
            )
            # Personality: arrogance + not humble = more power-seeking
            p_axes = d.get("personality_axes") or {}
            score += p_axes.get("humble_vs_arrogant", 0) * 0.5
            score += p_axes.get("conformist_vs_rebel", 0) * -0.3
            char_power[name.lower()] = max(0, score)

    # Directed influence graph from how.json relations
    G = nx.DiGraph()
    for name in char_power:
        G.add_node(name, power=char_power[name])

    for scene_dir in sorted(ANALYSIS_DIR.iterdir()):
        if not scene_dir.is_dir():
            continue
        how = _load_json(scene_dir / "how.json")
        for rel in (how.get("character_relations") or []):
            frm  = (rel.get("from_char") or "").lower()
            to   = (rel.get("to_char") or "").lower()
            rtype = rel.get("relation_type","").lower()
            w    = _REL_POWER.get(rtype, 0)
            if w > 0 and frm and to and frm != to:
                if G.has_edge(frm, to):
                    G[frm][to]["weight"] = G[frm][to]["weight"] + w
                else:
                    G.add_edge(frm, to, weight=w)

    if not G.nodes():
        print("[warn] No characters found.")
        return None

    # Node sizes from power score
    powers  = {n: G.nodes[n].get("power", 1) for n in G.nodes()}
    max_pow = max(powers.values()) if powers else 1
    sizes   = [200 + 2000 * (powers.get(n, 0) / max(max_pow, 1)) for n in G.nodes()]
    colors  = ["#e74c3c" if powers.get(n, 0) > max_pow * 0.6
               else "#f39c12" if powers.get(n, 0) > max_pow * 0.3
               else "#3498db" for n in G.nodes()]

    fig, ax = plt.subplots(figsize=(13, 9))
    pos = nx.spring_layout(G, seed=42, k=2.5)
    nx.draw_networkx_nodes(G, pos, node_size=sizes, node_color=colors, alpha=0.85, ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=7, font_color="white", font_weight="bold", ax=ax)
    nx.draw_networkx_edges(G, pos, edge_color="#7f8c8d", arrows=True,
                           arrowsize=15, width=1.2, alpha=0.6, ax=ax,
                           connectionstyle="arc3,rad=0.1")

    ax.set_title("Power dynamics map — node size = power score, arrows = influence direction")
    ax.axis("off")

    from matplotlib.patches import Patch
    legend = [Patch(color="#e74c3c", label="High power"),
              Patch(color="#f39c12", label="Medium power"),
              Patch(color="#3498db", label="Low power")]
    ax.legend(handles=legend, loc="lower left", fontsize=8)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = output_path or OUT_DIR / "power_map.png"
    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Power map saved → {out}")
    return out


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--social",    action="store_true", help="Build social graph")
    parser.add_argument("--evolution", nargs=2, metavar=("CHAR_A", "CHAR_B"),
                        help="Relationship evolution between two characters")
    parser.add_argument("--pcnpc",     action="store_true", help="PC/NPC heuristic")
    args = parser.parse_args()

    if args.social:
        build_social_graph()
    elif args.evolution:
        build_relation_evolution(args.evolution[0], args.evolution[1])
    elif args.pcnpc:
        results = build_pcnpc_heuristic()
        print(f"\nPC/NPC heuristic ({len(results)} characters):\n")
        for r in results:
            print(f"  {r['role']:20s} {r['character']:25s} ({r['authors']} author(s): {', '.join(r['author_list'][:4])})")
    else:
        parser.print_help()
