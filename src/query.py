"""
Query engine — routes natural language questions to specialized handlers.

Handlers:
  analytical   → underexploited chars, stats, contradictions, open threads
  events       → full-text search over what.json events
  links        → ownership / location / relationship search over how.json
  combo        → scenes where multiple characters appear together
  entity       → enriched RAG (entity profiles + relations + narrative axes)
  creative     → generative suggestions grounded in existing lore
  arc          → character evolution across scenes
  knows        → what a character knows / doesn't know
  story        → story-so-far narrative summary
  importance   → scene importance ranking
  continuity   → description contradictions across scenes
  foreshadow   → concepts gaining narrative weight over time

Usage:
  python src/query.py "qui a peu de scènes ?"
  python src/query.py  (interactive)
"""

import json
import re
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from llm import call_llm
try:
    from store import search as _store_search, search_chars, search_places, search_concepts
    _STORE_OK = True
except Exception:
    _STORE_OK = False
    def _store_search(*a, **kw): return []
    def search_chars(*a, **kw): return []
    def search_places(*a, **kw): return []
    def search_concepts(*a, **kw): return []

DATA_DIR      = ROOT / "data"
LORE_DIR      = DATA_DIR / "lore"
ANALYSIS_DIR  = DATA_DIR / "analysis"
CONFIRMED_DIR = DATA_DIR / "confirmed_scenes"

MAX_SCENES    = 8
MAX_CTX_CHARS = 14000

# ── Lore loading ───────────────────────────────────────────────────────────────

def _load_yaml(path: Path) -> dict:
    try:
        return yaml.safe_load(path.read_text(encoding="utf-8")) or {} if path.exists() else {}
    except Exception:
        return {}


def _load_yaml_dir(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out = []
    for f in sorted(path.glob("*.yaml")):
        try:
            d = yaml.safe_load(f.read_text(encoding="utf-8")) or {}
            d["_file"] = f.stem
            out.append(d)
        except Exception:
            pass
    return out


def _load_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    except Exception:
        return {}


def load_all_lore() -> dict:
    return {
        "characters": _load_yaml_dir(LORE_DIR / "characters"),
        "places":     _load_yaml_dir(LORE_DIR / "places"),
        "concepts":   _load_yaml_dir(LORE_DIR / "concepts"),
        "how_context": _load_yaml(LORE_DIR / "how_context.yaml"),
    }


# ── Field masking ─────────────────────────────────────────────────────────────

# Per-handler field whitelist — only load what each handler actually needs
_HANDLER_FIELDS = {
    "roleplay":    {"name", "_summary", "description_psychological", "beliefs", "likes",
                   "dislikes", "emotional_polarity", "personality_axes", "appearances"},
    "react":       {"name", "_summary", "description_psychological", "beliefs", "likes",
                   "dislikes", "emotional_polarity", "personality_axes", "emotional_triggers",
                   "competency_axes", "appearances"},
    "arc":         {"name", "_summary", "appearances", "emotional_polarity", "personality_axes",
                   "description_psychological"},
    "continuity":  {"name", "description_physical", "appearances"},
    "knows":       {"name", "appearances", "related_characters"},
    "catchup":     {"name", "_summary", "appearances"},
    "goals":       {"name", "_summary", "description_psychological", "beliefs", "likes",
                   "dislikes", "appearances"},
    "voice":       {"name", "_summary", "appearances"},
    "links":       {"name", "_summary", "appearances"},
    "combo":       {"name", "appearances"},
    "analytical":  None,  # needs everything
    "entity":      None,
    "events":      None,
}

_CHAR_FULL_FIELDS = None  # sentinel = all fields


def _slim(entity: dict, handler: str) -> dict:
    """Return entity with only the fields relevant to this handler."""
    fields = _HANDLER_FIELDS.get(handler)
    if fields is None:
        return entity
    return {k: v for k, v in entity.items() if k in fields}


def _store_ctx(question: str, categories: list[str] | None = None, n: int = 8) -> str:
    """Semantic search → compact context block from summary cards."""
    hits = _store_search(question, categories=categories, n=n)
    if not hits:
        return ""
    return "\n".join(
        f"[{h['category']}] {h['name']}: {h['summary']}"
        for h in hits
    )


# ── Intent detection ──────────────────────────────────────────────────────────

_INTENT_PATTERNS = {
    "analytical": [
        r"\bpeu exploit", r"\bsous.utilis", r"\brarement\b", r"\boubli",
        r"\bpeu de sc[eè]ne", r"\bstats?\b",
        r"\bfil(s)? narratif", r"\bouvert(e|s)?\b", r"\bnon d[eé]velopp",
        r"\bfrequen", r"\bcomptage", r"\bcombien de sc[eè]ne",
    ],
    "events": [
        r"\bqui a\b", r"\bqui est.ce qui\b", r"\bestce que\b", r"\blequel\b",
        r"\bqu[' ]est.il arriv", r"\bqu[' ]a fait\b", r"\ba mang[eé]\b",
        r"\ba dit\b", r"\ba donn[eé]\b", r"\ba trouv[eé]\b", r"\ba tu[eé]\b",
        r"\bme souviens\b", r"\bsouvenir\b", r"\bpass[eé]\b", r"\ba eu lieu\b",
        r"\bque s.est.il pass[eé]", r"\bqu.est.ce qui s.est pass[eé]",
    ],
    "links": [
        r"\bappartient\b", r"\bappartenance\b", r"\bposs[eè]de\b", r"\bpropriet",
        r"\bvit [àa]\b", r"\bhabite\b", r"\bdemeure\b", r"\bdomicile\b",
        r"\alli[eé]", r"\bennemi\b", r"\brival\b", r"\blien(s)?\b",
        r"\brelation(s)?\b", r"\bami(e|s)?\b", r"\bconnect", r"\boppose\b",
        r"\bsoutient\b", r"\bfamille\b", r"\bmen[tors]", r"\bsubordonn",
    ],
    "combo": [
        r"\bet\b.{1,30}\bensemble\b", r"\bavec\b.{1,30}(et|aussi)\b",
        r"\bsc[eè]nes? o[uù]\b.{1,40}\bet\b",
        r"\btous les deux\b", r"\btoutes les deux\b", r"\bquand.{1,30}et\b",
        r"\binteraction(s)?\b.{1,30}entre\b",
    ],
    "creative": [
        r"\bid[eé]e(s)?\b", r"\bsuggestion(s)?\b", r"\binspiration\b",
        r"\bque pourrait\b", r"\bqu.est.ce qu.on pourrait\b",
        r"\bpossibilit[eé](s)?\b", r"\baxe(s)? narratif", r"\bplotline\b",
        r"\bpropose\b", r"\bimagine\b", r"\bcr[eé]er\b", r"\bd[eé]velopper\b",
    ],
    "arc": [
        r"\barc\b", r"\b[eé]volution\b", r"\b[eé]volu[eé]\b", r"\bchang[eé]\b",
        r"\btransform", r"\bau fil\b", r"\bau cours\b", r"\bprogressiv",
        r"\bdevenu\b", r"\b[eé]tait.{1,20}maintenant\b", r"\bavant.{1,20}apr[eè]s\b",
    ],
    "knows": [
        r"\bsait\b", r"\bne sait pas\b", r"\bignore\b", r"\bconscient\b",
        r"\bau courant\b", r"\bconnaissance\b", r"\bsavait\b", r"\bappris\b",
        r"\binform[eé]\b", r"\bau fait\b", r"\bcach[eé]\b", r"\bsecret\b",
    ],
    "story": [
        r"\br[eé]sum[eé]\b", r"\bhistoire\b", r"\bjusqu.ici\b", r"\bso far\b",
        r"\bce qui s.est pass[eé]\b", r"\bglobal\b", r"\bd.ensemble\b",
        r"\bsynth[eè]se\b", r"\bchrono", r"\btimeline\b", r"\bnarratif global\b",
    ],
    "importance": [
        r"\bimportante?(s)?\b", r"\bclef?\b", r"\bpivot\b", r"\bcrucia",
        r"\bsc[eè]ne(s)? majeure", r"\bmeilleure(s)? sc[eè]ne", r"\brang",
        r"\bclass(ement|[eé])", r"\bscor", r"\bplus significat",
    ],
    "continuity": [
        r"\bcontinuit[eé]\b", r"\bcontredit\b", r"\binconsistant\b",
        r"\bcontradict", r"\bdiff[eé]rent(e|s)? (selon|dans|entre)\b",
        r"\bchang[eé] de\b.{1,20}(description|apparence)", r"\bpas le m[eê]me\b",
        r"\bincoh[eé]rence.{1,20}description", r"\bdescription.{1,20}varie",
        r"\bphysique.{1,15}diff[eé]rent", r"\bapparence.{1,15}(change|varie|diff)",
    ],
    "voice": [
        r"\bvoix\b", r"\bvoice\b", r"\bstyle.{1,10}(parole|[eé]criture|expression)",
        r"\bparle\b.{1,20}comment", r"\bempreinte", r"\bfingerprint",
        r"\b[eé]cri(re|s).{1,20}(comme|style|voix)", r"\bton\b.{1,15}personn",
        r"\bcommen[ct].{1,15}parle\b", r"\bmanière.{1,10}(parler|s.exprimer)",
    ],
    "catchup": [
        r"\bcatch.?up\b", r"\brattrap", r"\bmanqu[eé]\b.{1,20}session",
        r"\bpas l[aà]\b.{1,20}sc[eè]ne", r"\bpour comprendre\b",
        r"\bce que.{1,20}sait\b.{1,20}avant", r"\bcontexte\b.{1,20}(avant|pour)",
        r"\bresum[eé]\b.{1,20}(avant|pour|depuis)", r"\bmise [àa] jour\b",
        r"\bprior à\b", r"\bpréquelle\b",
    ],
    "temporal": [
        r"\bmoment\b.{1,20}(habituel|toujours|souvent)",
        r"\bheure\b.{1,20}sc[eè]ne", r"\bnuit\b.{1,15}(scène|perso|character)",
        r"\bmatin\b.{1,15}(scène|perso)", r"\bclustering\b",
        r"\bpattern.{1,15}(temps|horaire|moment)", r"\btemporel",
        r"\bmoment.{1,15}(journée|jour)", r"\bquand.{1,15}apparaît",
    ],
    "tension": [
        r"\btension\b", r"\bcourbe\b", r"\bintensit[eé]\b",
        r"\bpic\b.{1,15}(narratif|drama)", r"\bmonté(e|s)?\b",
        r"\bdramatique\b", r"\bcreux\b.{1,15}(narratif|drama)",
        r"\brythme\b.{1,15}histoir", r"\bclimaxs?\b",
    ],
    "irony": [
        r"\bironie\b", r"\bironi", r"\bne sait pas\b.{1,20}(que|mais)",
        r"\bignore\b.{1,20}(que|tandis)", r"\bcach[eé]\b.{1,20}(que|pour)",
        r"\bsecret\b.{1,20}(que|dont)", r"\bparadoxe\b", r"\btragi",
        r"\bqui sait.{1,20}qui ne sait pas", r"\bcaché à\b",
    ],
    "goals": [
        r"\bobjectif", r"\bbut\b", r"\bmotivation", r"\bveut\b.{1,20}(accomplir|atteindre|obtenir)",
        r"\bcherche [àa]\b", r"\bpourquoi\b.{1,20}(fait|agit|se comporte)",
        r"\bdesire\b", r"\baspire\b", r"\bambition", r"\bplan\b.{1,15}perso",
    ],
    "power": [
        r"\bpouvoir\b", r"\bdomin", r"\bhi[eé]rarchie\b", r"\binfluence\b",
        r"\bcommande\b", r"\bobéit\b", r"\bsoumet\b", r"\bcontrôle\b",
        r"\bautorité\b", r"\brapport.{1,10}force", r"\bqui.{1,15}(dirige|commande|domine)",
    ],
    "missing": [
        r"\bmanquante?\b.{1,15}sc[eè]ne", r"\bsc[eè]ne.{1,15}manquante?",
        r"\bpas montr[eé]\b", r"\bjamais vu\b", r"\bhors.?[eé]cran\b",
        r"\boff.?screen\b", r"\bgap\b", r"\btrou\b.{1,10}(narratif|histoir)",
        r"\bréf[eé]renc[eé]\b.{1,15}(mais|sans|jamais)", r"\bcause.{1,10}absent",
    ],
    "validate": [
        r"\bcoh[eé]renc[eé]\b", r"\bvalid", r"\bv[eé]rifi", r"\bcheck\b",
        r"\bcontredit\b", r"\bimpossible\b.{1,15}(si|vu|car)",
        r"\bcohérent\b", r"\binconsistanc", r"\bcontradiction.{1,10}lore",
        r"\bfait.{1,15}confirm[eé]\b", r"\bmort\b.{1,20}(réapparaît|vivant)",
    ],
    "foreshadow": [
        r"\bpr[eé]sag", r"\bforeshadow", r"\bannonce\b", r"\bpr[eé]monit",
        r"\bgagne.{1,15}importance", r"\bprend.{1,15}ampleur", r"\bmontee\b",
        r"\b[eé]merg", r"\bdevient central", r"\bsignal faible",
    ],
    "roleplay": [
        r"\bque dirait\b", r"\bqu.est.ce que.{1,20}dirait\b",
        r"\br[eé]pond(re|s)?.{1,15}comme\b", r"\bparle.{1,15}comme\b",
        r"\b[àa] la place de\b", r"\bdu point de vue de\b",
        r"\bselon\b.{1,20}(que|quoi|comment)", r"\bdiscours de\b",
        r"\bimite\b.{1,10}(personn|voix)", r"\bincarne\b",
    ],
    "react": [
        r"\bcomment r[eé]agirait\b", r"\br[eé]action de\b.{1,20}[àa]\b",
        r"\bsi\b.{1,30}apprenait\b", r"\bsi\b.{1,30}voyait\b",
        r"\bsi\b.{1,30}[eé]tait face [àa]\b", r"\bface [àa]\b.{1,20}comment\b",
        r"\bque ferait\b", r"\bcomment.{1,20}ferait\b.{1,20}si\b",
        r"\bhypoth[eé]tique\b", r"\bscenario\b.{1,20}(si|avec)\b",
        r"\bimaginons que\b", r"\bet si\b.{1,20}(devait|pouvait|[eé]tait)\b",
    ],
}


def detect_intent(question: str) -> str:
    q = question.lower()
    scores = {intent: 0 for intent in _INTENT_PATTERNS}
    for intent, patterns in _INTENT_PATTERNS.items():
        for pat in patterns:
            if re.search(pat, q):
                scores[intent] += 1
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "entity"


# ── Shared helpers ────────────────────────────────────────────────────────────

def _names_for(entity: dict) -> list[str]:
    names = []
    if entity.get("name"):
        names.append(entity["name"].lower())
    for a in entity.get("appellations") or []:
        names.append(str(a).lower())
    return names


def _entity_matches(entity: dict, terms: list[str]) -> bool:
    entity_names = _names_for(entity)
    for term in terms:
        t = term.lower().strip()
        if not t or len(t) < 3:
            continue
        for name in entity_names:
            if t in name or name in t:
                return True
    return False


def _extract_terms(question: str, lore: dict) -> list[str]:
    q_low = question.lower()
    matched = []
    for cat in ("characters", "places", "concepts"):
        for entity in lore[cat]:
            for name in _names_for(entity):
                if len(name) >= 3 and name in q_low:
                    matched.append(name)
    if matched:
        return list(set(matched))
    # fallback: capitalized words
    return [w.lower() for w in re.findall(r'\b[A-ZÀ-Ÿ][a-zà-ÿ]{2,}\b', question)]


def _fmt_entity(entity: dict, cat: str) -> str:
    lines = [f"[{cat.upper()}] {entity.get('name', '?')}"]
    if entity.get("appellations"):
        lines.append(f"  aliases: {', '.join(str(a) for a in entity['appellations'])}")
    for field in ("description", "description_physical", "description_psychological",
                  "job", "significance", "type", "author"):
        if entity.get(field):
            lines.append(f"  {field}: {entity[field]}")
    for field in ("beliefs", "attributes", "related_characters", "main_locations", "relations"):
        items = entity.get(field)
        if items:
            lines.append(f"  {field}: {', '.join(str(x) for x in items)}")
    if entity.get("appearances"):
        lines.append(f"  scenes ({len(entity['appearances'])}): {', '.join(str(s) for s in entity['appearances'][:5])}")
    return "\n".join(lines)


def _scene_summary(scene_id: str) -> str:
    what = _load_json(ANALYSIS_DIR / scene_id / "what.json")
    how  = _load_json(ANALYSIS_DIR / scene_id / "how.json")
    parts = []
    if what.get("summary"):
        parts.append(f"[{scene_id}] {what['summary']}")
    for ev in (what.get("events") or [])[:6]:
        desc  = ev.get("description", "")
        chars = ", ".join(ev.get("characters") or [])
        if desc:
            parts.append(f"  • {desc}" + (f" ({chars})" if chars else ""))
    if how.get("context_synthesis"):
        parts.append(f"  [context] {how['context_synthesis']}")
    return "\n".join(parts)


_ANSWER_PROMPT = """\
You are an expert analyst of a narrative roleplay universe.
Answer in the same language as the question.
Use ONLY the provided context. Be precise, cite characters and scenes when relevant.
If context is insufficient, say so clearly.

CONTEXT:
{context}

QUESTION: {question}

ANSWER:"""


def _llm_answer(context: str, question: str) -> str:
    if len(context) > MAX_CTX_CHARS:
        context = context[:MAX_CTX_CHARS] + "\n[... context truncated ...]"
    return call_llm(
        _ANSWER_PROMPT.format(context=context, question=question),
        num_predict=-1,
        num_ctx=16384,
    )


# ── Handler: analytical ───────────────────────────────────────────────────────

def query_analytical(question: str, lore: dict) -> str:
    chars = lore["characters"]
    q = question.lower()

    # Contradiction detection
    if re.search(r"contradictions?|incoh[eé]rence", q):
        issues = []
        for c in chars:
            name = c.get("name", "?")
            apps = c.get("appearances") or []
            desc = c.get("description_psychological") or ""
            if len(apps) > 2 and not desc:
                issues.append(f"- {name}: {len(apps)} scenes but no psychological description")
        if issues:
            return "Potential inconsistencies / gaps:\n" + "\n".join(issues)
        return "No obvious inconsistencies detected."

    # Open narrative threads
    if re.search(r"fil(s)? narratif|ouvert|non d[eé]velopp|plotline", q):
        general_how = _load_yaml(LORE_DIR / "general_how.yaml")
        axes = general_how.get("narrative_axes") or []
        weak = [a for a in axes if a.get("strength") == "moderate" or len(a.get("scenes") or []) <= 2]
        if not weak:
            return "No weak/open narrative threads detected yet. Run more scenes first."
        lines = ["Open / underdeveloped narrative threads:"]
        for a in weak:
            scenes_str = ", ".join(a.get("scenes") or [])
            lines.append(f"\n• **{a['name']}** — {a.get('summary', '')}")
            lines.append(f"  elements: {', '.join(a.get('elements') or [])}")
            lines.append(f"  scenes: {scenes_str or 'none recorded'}")
        return "\n".join(lines)

    # Default: underexploited characters
    sorted_chars = sorted(chars, key=lambda c: len(c.get("appearances") or []))
    threshold = 3
    low = [c for c in sorted_chars if len(c.get("appearances") or []) <= threshold]

    if not low:
        return f"All characters appear in more than {threshold} scenes."

    lines = [f"Characters with ≤{threshold} scene appearances:\n"]
    for c in low:
        name  = c.get("name", "?")
        apps  = c.get("appearances") or []
        job   = c.get("job") or "unknown role"
        psych = (c.get("description_psychological") or "")[:80]
        locs  = ", ".join((c.get("main_locations") or [])[:3]) or "—"
        lines.append(f"• **{name}** ({len(apps)} scene(s)) — {job}")
        rels  = ", ".join((c.get("relations") or [])[:3]) or "—"
        if psych:
            lines.append(f"  {psych}")
        lines.append(f"  locations: {locs}")
        lines.append(f"  locations: {rels}")
        lines.append(f"  scenes: {', '.join(str(s) for s in apps) or 'none'}")
    return "\n".join(lines)


# ── Handler: events (full-text over what.json) ────────────────────────────────

def query_events(question: str, lore: dict) -> str:
    if not ANALYSIS_DIR.exists():
        return "No analyzed scenes found. Run the pipeline first."

    # Keywords: remove common stop words
    stop = {"qui", "que", "quoi", "est", "ce", "il", "elle", "les", "des", "une", "un",
            "le", "la", "de", "du", "en", "et", "ou", "si", "a", "au", "aux", "y"}
    keywords = [
        w.lower() for w in re.findall(r'\b\w{3,}\b', question)
        if w.lower() not in stop
    ]

    matches = []
    for scene_dir in sorted(ANALYSIS_DIR.iterdir()):
        if not scene_dir.is_dir():
            continue
        what = _load_json(scene_dir / "what.json")
        scene_id = scene_dir.name
        for ev in (what.get("events") or []):
            desc = (ev.get("description") or "").lower()
            if any(kw in desc for kw in keywords):
                chars = ", ".join(ev.get("characters") or [])
                matches.append({
                    "scene": scene_id,
                    "type":  ev.get("type", ""),
                    "desc":  ev.get("description", ""),
                    "chars": chars,
                })

    if not matches:
        return f"No events found matching keywords: {', '.join(keywords)}"

    context_lines = [f"{len(matches)} matching event(s) found:\n"]
    for m in matches[:20]:
        context_lines.append(
            f"[{m['scene']}] ({m['type']}) {m['desc']}"
            + (f" — {m['chars']}" if m['chars'] else "")
        )

    context = "\n".join(context_lines)
    return _llm_answer(context, question)


# ── Handler: links (how.json — ownership, location, relations) ────────────────

def query_links(question: str, lore: dict) -> str:
    if not ANALYSIS_DIR.exists():
        return "No analyzed scenes found. Run the pipeline first."

    terms = _extract_terms(question, lore)
    q_low = question.lower()

    # Detect target link types from question
    target_types: list[str] = []
    if re.search(r"appartient|poss[eè]de|propri[eé]t", q_low):
        target_types += ["owns", "belongs_to"]
    if re.search(r"vit|habite|demeure|domicile", q_low):
        target_types += ["inhabits", "lives_at"]
    if re.search(r"alli[eé]|ami|soutient", q_low):
        target_types += ["supports", "ally"]
    if re.search(r"ennemi|oppose|rival|combat", q_low):
        target_types += ["opposes", "enemy"]

    link_matches = []
    rel_matches  = []

    for scene_dir in sorted(ANALYSIS_DIR.iterdir()):
        if not scene_dir.is_dir():
            continue
        how = _load_json(scene_dir / "how.json")
        sid = scene_dir.name

        for link in (how.get("links") or []):
            frm  = (link.get("from_element") or "").lower()
            to   = (link.get("to_element") or "").lower()
            ltype = (link.get("link_type") or "").lower()
            desc  = link.get("description", "")

            term_hit  = any(t in frm or t in to for t in terms) if terms else True
            type_hit  = any(tt in ltype for tt in target_types) if target_types else True

            if term_hit or type_hit:
                link_matches.append(f"[{sid}] {frm} —{ltype}→ {to}: {desc[:80]}")

        for rel in (how.get("character_relations") or []):
            frm  = (rel.get("from_char") or "").lower()
            to   = (rel.get("to_char") or "").lower()
            rtype = rel.get("relation_type", "")
            sent  = rel.get("sentiment", "")
            desc  = rel.get("description", "")

            if not terms or any(t in frm or t in to for t in terms):
                rel_matches.append(
                    f"[{sid}] {frm} → {to} ({rtype}, {sent}): {desc[:80]}"
                )

    if not link_matches and not rel_matches:
        return "No relevant links or relations found."

    parts = []
    if link_matches:
        parts.append("**Causal / ownership links:**\n" + "\n".join(link_matches[:15]))
    if rel_matches:
        parts.append("**Character relations:**\n" + "\n".join(rel_matches[:15]))

    context = "\n\n".join(parts)
    return _llm_answer(context, question)


# ── Handler: combo (scenes with multiple characters) ──────────────────────────

def query_combo(question: str, lore: dict) -> str:
    terms = _extract_terms(question, lore)
    char_names = set()
    for c in lore["characters"]:
        for name in _names_for(c):
            if any(t in name or name in t for t in terms):
                char_names.add(c.get("name", "").lower())

    if len(char_names) < 2:
        return (
            "Could not identify two or more characters in the question. "
            f"Terms detected: {terms}"
        )

    matches = []
    for scene_dir in sorted(ANALYSIS_DIR.iterdir()):
        if not scene_dir.is_dir():
            continue
        who = _load_json(scene_dir / "who.json")
        scene_chars = {c.lower() for c in (who.get("characters") or [])}
        if char_names.issubset(scene_chars):
            matches.append(scene_dir.name)

    if not matches:
        char_list = ", ".join(char_names)
        return f"No scenes found where all of these characters appear together: {char_list}"

    context_parts = [f"Scenes with {', '.join(char_names)} together:\n"]
    for sid in matches[:MAX_SCENES]:
        context_parts.append(_scene_summary(sid))

    return _llm_answer("\n\n".join(context_parts), question)


# ── Handler: entity (enriched RAG) ────────────────────────────────────────────

def query_entity(question: str, lore: dict) -> str:
    terms = _extract_terms(question, lore)

    # Semantic retrieval first (fast, scalable)
    sem_hits = _store_search(question, n=10) if _STORE_OK else []
    sem_names = {h["name"].lower() for h in sem_hits}

    matched = {"characters": [], "places": [], "concepts": []}
    for cat in ("characters", "places", "concepts"):
        for entity in lore[cat]:
            name = (entity.get("name") or "").lower()
            if _entity_matches(entity, terms) or name in sem_names:
                matched[cat].append(entity)

    sections = []

    # Entity profiles
    for cat in ("characters", "places", "concepts"):
        for entity in matched[cat]:
            sections.append(_fmt_entity(entity, cat))

    # Character relations from how.json for matched characters
    char_names_matched = {
        c.get("name", "").lower() for c in matched["characters"]
    }
    if char_names_matched:
        rel_lines = []
        for scene_dir in sorted(ANALYSIS_DIR.iterdir()):
            if not scene_dir.is_dir():
                continue
            how = _load_json(scene_dir / "how.json")
            for rel in (how.get("character_relations") or []):
                frm = (rel.get("from_char") or "").lower()
                to  = (rel.get("to_char") or "").lower()
                if frm in char_names_matched or to in char_names_matched:
                    rel_lines.append(
                        f"  {frm} → {to} ({rel.get('relation_type')}, {rel.get('sentiment')}): "
                        f"{rel.get('description', '')[:80]}"
                    )
        if rel_lines:
            sections.append("[CHARACTER RELATIONS]\n" + "\n".join(rel_lines[:15]))

    # Narrative axes from general_how
    general_how = _load_yaml(LORE_DIR / "general_how.yaml")
    axes = general_how.get("narrative_axes") or []
    relevant_axes = [
        a for a in axes
        if any(
            t in " ".join(a.get("elements") or []).lower() or
            t in (a.get("name") or "").lower()
            for t in terms
        )
    ]
    if relevant_axes:
        ax_lines = ["[NARRATIVE AXES]"]
        for a in relevant_axes[:3]:
            ax_lines.append(f"• {a['name']}: {a.get('summary', '')}")
            ax_lines.append(f"  elements: {', '.join(a.get('elements') or [])}")
        sections.append("\n".join(ax_lines))

    # Scene details
    scene_ids = set()
    for cat in ("characters", "places", "concepts"):
        for entity in matched[cat]:
            for sid in (entity.get("appearances") or []):
                scene_ids.add(str(sid))

    scene_parts = []
    for sid in sorted(scene_ids)[:MAX_SCENES]:
        s = _scene_summary(sid)
        if s:
            scene_parts.append(s)
    if scene_parts:
        sections.append("[RELEVANT SCENES]\n" + "\n\n".join(scene_parts))

    # Fallback: recent narrative context
    if not sections:
        how_ctx = lore.get("how_context") or {}
        if how_ctx:
            overview = "\n".join(
                f"- {sid}: {synth}" for sid, synth in list(how_ctx.items())[-8:]
            )
            sections.append(f"[RECENT NARRATIVE CONTEXT]\n{overview}")

    if not sections:
        return "No relevant information found. Run the pipeline first."

    return _llm_answer("\n\n".join(sections), question)


# ── Handler: arc (character evolution) ───────────────────────────────────────

def query_arc(question: str, lore: dict) -> str:
    terms = _extract_terms(question, lore)
    target = next((c for c in lore["characters"] if _entity_matches(c, terms)), None)
    if not target and _STORE_OK:
        hits = search_chars(question, n=1)
        if hits:
            hit_name = hits[0]["name"].lower()
            target = next((c for c in lore["characters"] if c.get("name","").lower() == hit_name), None)
    if not target:
        return f"Character not identified. Terms detected: {terms}"

    char_name = target.get("name", "")
    appearances = sorted(str(s) for s in (target.get("appearances") or []))
    if not appearances:
        return f"{char_name} has no recorded scenes."

    context_parts = [f"Character arc — {char_name} across {len(appearances)} scenes:\n"]
    for sid in appearances:
        who = _load_json(ANALYSIS_DIR / sid / "who.json")
        details = who.get("details") or []
        char_detail = next(
            (d for d in details if char_name.lower() in (d.get("canonical_name") or "").lower()),
            {}
        )
        how = _load_json(ANALYSIS_DIR / sid / "how.json")
        rels = [
            r for r in (how.get("character_relations") or [])
            if char_name.lower() in (r.get("from_char") or "").lower()
            or char_name.lower() in (r.get("to_char") or "").lower()
        ]
        lines = [f"\n[{sid}]"]
        if char_detail.get("description_psychological"):
            lines.append(f"  state: {char_detail['description_psychological'][:100]}")
        if char_detail.get("beliefs"):
            lines.append(f"  beliefs: {', '.join(char_detail['beliefs'][:3])}")
        for r in rels[:3]:
            lines.append(
                f"  → {r.get('to_char') if r.get('from_char','').lower() in char_name.lower() else r.get('from_char')}"
                f" ({r.get('relation_type')}, {r.get('sentiment')})"
            )
        context_parts.append("\n".join(lines))

    return _llm_answer("\n".join(context_parts), question)


# ── Handler: knows / doesn't know ────────────────────────────────────────────

def query_knows(question: str, lore: dict) -> str:
    terms = _extract_terms(question, lore)
    target = next((c for c in lore["characters"] if _entity_matches(c, terms)), None)
    if not target and _STORE_OK:
        hits = search_chars(question, n=1)
        if hits:
            hit_name = hits[0]["name"].lower()
            target = next((c for c in lore["characters"] if c.get("name","").lower() == hit_name), None)
    if not target:
        return f"Character not identified. Terms detected: {terms}"

    char_name   = target.get("name", "")
    char_scenes = set(str(s) for s in (target.get("appearances") or []))
    related     = {c.lower() for c in (target.get("related_characters") or [])}

    known, unknown = [], []
    for scene_dir in sorted(ANALYSIS_DIR.iterdir()):
        if not scene_dir.is_dir():
            continue
        sid  = scene_dir.name
        what = _load_json(scene_dir / "what.json")
        for ev in (what.get("events") or []):
            if ev.get("type") not in ("revelation", "decision"):
                continue
            desc  = ev.get("description", "")
            chars = [c.lower() for c in (ev.get("characters") or [])]
            if sid in char_scenes:
                known.append(f"[{sid}] {desc}")
            else:
                # relevant if involves related characters or entities from the question
                if related & set(chars) or any(t in desc.lower() for t in terms):
                    unknown.append(f"[{sid}] {desc} (chars: {', '.join(chars)})")

    context = (
        f"Character: {char_name}\n\n"
        f"KNOWS — revelations & decisions from {len(char_scenes)} scene(s):\n"
        + ("\n".join(known[:15]) or "none recorded")
        + f"\n\nDOESN'T KNOW — relevant events in scenes where {char_name} was absent:\n"
        + ("\n".join(unknown[:15]) or "none found")
    )
    return _llm_answer(context, question)


# ── Handler: story-so-far ─────────────────────────────────────────────────────

def query_story(question: str, lore: dict) -> str:
    general_how  = _load_yaml(LORE_DIR / "general_how.yaml")
    general_what = _load_yaml(LORE_DIR / "general_what.yaml")

    parts = []

    overall = general_how.get("overall_narrative_summary", "")
    if overall:
        parts.append(f"## Overall narrative\n{overall}")

    axes = [a for a in (general_how.get("narrative_axes") or []) if a.get("strength") == "strong"]
    if axes:
        lines = ["## Active narrative axes"]
        for a in axes[:5]:
            lines.append(f"**{a['name']}**: {a.get('summary','')}")
            lines.append(f"  elements: {', '.join(a.get('elements',[])[:6])}")
        parts.append("\n".join(lines))

    rec = general_what.get("recurrences", {})
    if rec.get("overall_summary"):
        parts.append(f"## Recurring content\n{rec['overall_summary']}")
    if rec.get("themes"):
        parts.append("**Recurring themes**: " + ", ".join(rec["themes"][:8]))

    strongest = (general_how.get("strongest_links") or [])[:6]
    if strongest:
        lines = ["## Strongest causal links"]
        for lnk in strongest:
            lines.append(
                f"- {lnk.get('from_element')} —{lnk.get('link_type')}→ {lnk.get('to_element')}: "
                f"{lnk.get('description','')[:80]}"
            )
        parts.append("\n".join(lines))

    if not parts:
        return "No general synthesis available yet. Run more scenes through the pipeline first."

    # No LLM needed — already synthesized data
    return "\n\n".join(parts)


# ── Handler: importance (scene ranking) ──────────────────────────────────────

_TYPE_SCORE = {"revelation": 3, "decision": 2, "emotional": 2, "action": 1, "conversation": 0.5}


def query_importance(question: str, lore: dict) -> str:
    if not ANALYSIS_DIR.exists():
        return "No analyzed scenes. Run the pipeline first."

    scores = []
    for scene_dir in sorted(ANALYSIS_DIR.iterdir()):
        if not scene_dir.is_dir():
            continue
        what = _load_json(scene_dir / "what.json")
        how  = _load_json(scene_dir / "how.json")
        who  = _load_json(scene_dir / "who.json")

        score = 0.0
        for ev in (what.get("events") or []):
            score += _TYPE_SCORE.get(ev.get("type", ""), 0.5)
            score += len(ev.get("characters") or []) * 0.3

        score += len(how.get("links") or []) * 0.4
        score += len(how.get("character_relations") or []) * 0.5
        score += len(who.get("characters") or []) * 0.3

        scores.append((scene_dir.name, score, what.get("summary", "")))

    scores.sort(key=lambda x: x[1], reverse=True)

    lines = [f"Scene importance ranking ({len(scores)} scenes):\n"]
    for rank, (sid, sc, summary) in enumerate(scores[:15], 1):
        lines.append(f"{rank:2d}. [{sid}] score={sc:.1f} — {summary[:90]}")

    return "\n".join(lines)


# ── Handler: continuity checker ───────────────────────────────────────────────

def query_continuity(question: str, lore: dict) -> str:
    issues = []

    # Characters: track description_physical changes across scenes
    for char in lore["characters"]:
        name = char.get("name", "?")
        descs: list[tuple[str, str]] = []
        for sid in (char.get("appearances") or []):
            who = _load_json(ANALYSIS_DIR / str(sid) / "who.json")
            for d in (who.get("details") or []):
                if name.lower() in (d.get("canonical_name") or "").lower():
                    phys = d.get("description_physical", "").strip()
                    if phys:
                        descs.append((str(sid), phys))
        if len(descs) >= 2:
            # Flag pairs where descriptions differ significantly
            unique = list({d for _, d in descs})
            if len(unique) > 1:
                issues.append(
                    f"**{name}** — physical description varies across scenes:\n"
                    + "\n".join(f"  [{sid}] {d[:80]}" for sid, d in descs[:4])
                )

    # Places: track attribute/description changes
    for place in lore["places"]:
        name = place.get("name", "?")
        descs = []
        for sid in (place.get("appearances") or []):
            where = _load_json(ANALYSIS_DIR / str(sid) / "where.json")
            for loc in (where.get("details") or []):
                if name.lower() in (loc.get("canonical_name") or "").lower():
                    desc = loc.get("description", "").strip()
                    if desc:
                        descs.append((str(sid), desc))
        if len(descs) >= 2:
            unique = list({d for _, d in descs})
            if len(unique) > 1:
                issues.append(
                    f"**{name}** (place) — description varies:\n"
                    + "\n".join(f"  [{sid}] {d[:80]}" for sid, d in descs[:4])
                )

    if not issues:
        return "No continuity issues detected in physical/location descriptions."

    context = f"{len(issues)} potential continuity issue(s) found:\n\n" + "\n\n".join(issues[:10])
    return _llm_answer(context, question)


# ── Handler: foreshadowing detector ──────────────────────────────────────────

def query_foreshadow(question: str, lore: dict) -> str:
    if not ANALYSIS_DIR.exists():
        return "No analyzed scenes. Run the pipeline first."

    concept_timeline: dict[str, list[str]] = {}
    scenes_ordered = sorted(
        (d for d in ANALYSIS_DIR.iterdir() if d.is_dir()),
        key=lambda d: d.name
    )

    for scene_dir in scenes_ordered:
        which = _load_json(scene_dir / "which.json")
        for concept in (which.get("concepts") or []):
            concept_timeline.setdefault(concept, []).append(scene_dir.name)

    if not concept_timeline:
        return "No concepts indexed yet. Run the pipeline first."

    candidates = []
    total = len(scenes_ordered)
    for concept, appearances in concept_timeline.items():
        if len(appearances) < 2:
            continue
        # First half vs second half frequency
        midpoint = scenes_ordered[total // 2].name if total > 1 else ""
        early = [s for s in appearances if s <= midpoint]
        late  = [s for s in appearances if s > midpoint]
        if late and len(late) > len(early) * 1.5:
            candidates.append({
                "concept":    concept,
                "early":      len(early),
                "late":       len(late),
                "first_seen": appearances[0],
                "total":      len(appearances),
            })

    candidates.sort(key=lambda x: x["late"] / max(x["early"], 0.5), reverse=True)

    if not candidates:
        return "No clear foreshadowing pattern detected yet (need more scenes for trend analysis)."

    lines = [f"Concepts gaining narrative weight over time ({len(candidates)} found):\n"]
    for c in candidates[:10]:
        ratio = c["late"] / max(c["early"], 0.5)
        lines.append(
            f"• **{c['concept']}** — first seen: {c['first_seen']}, "
            f"early: {c['early']}x, late: {c['late']}x (×{ratio:.1f})"
        )

    return "\n".join(lines)


# ── Handler: creative ─────────────────────────────────────────────────────────

_CREATIVE_PROMPT = """\
You are a creative narrative consultant for a roleplay campaign.
Based on the existing lore below, suggest 3 to 5 narrative ideas that:
- Are consistent with what already exists
- Develop underexploited characters or threads
- Create interesting interactions between existing elements

Answer in the same language as the request.

EXISTING LORE:
{context}

REQUEST: {question}

NARRATIVE IDEAS:"""


def query_creative(question: str, lore: dict) -> str:
    general_how  = _load_yaml(LORE_DIR / "general_how.yaml")
    general_what = _load_yaml(LORE_DIR / "general_what.yaml")

    parts = []

    # Underexploited characters
    chars = sorted(lore["characters"], key=lambda c: len(c.get("appearances") or []))
    low_chars = chars[:5]
    if low_chars:
        lines = ["Underexploited characters:"]
        for c in low_chars:
            lines.append(
                f"- {c.get('name')} ({len(c.get('appearances') or [])} scenes): "
                f"{c.get('description_psychological') or c.get('job') or '?'}"
            )
        parts.append("\n".join(lines))

    # Narrative axes
    axes = general_how.get("narrative_axes") or []
    if axes:
        lines = ["Narrative axes:"]
        for a in axes[:4]:
            lines.append(f"- {a['name']}: {a.get('summary', '')} (elements: {', '.join(a.get('elements') or [])})")
        parts.append("\n".join(lines))

    # Recurring themes
    recurrences = general_what.get("recurrences") or {}
    themes = recurrences.get("themes") or []
    if themes:
        parts.append("Recurring themes: " + ", ".join(themes[:8]))

    # Open threads
    weak_axes = [a for a in axes if a.get("strength") == "moderate" or len(a.get("scenes") or []) <= 2]
    if weak_axes:
        lines = ["Open / underdeveloped threads:"]
        for a in weak_axes[:3]:
            lines.append(f"- {a['name']}: {a.get('summary', '')}")
        parts.append("\n".join(lines))

    context = "\n\n".join(parts) or "No lore yet — run the pipeline first."

    if len(context) > MAX_CTX_CHARS:
        context = context[:MAX_CTX_CHARS]

    return call_llm(
        _CREATIVE_PROMPT.format(context=context, question=question),
        num_predict=-1,
        num_ctx=16384,
    )


# ── Handler: voice fingerprint ───────────────────────────────────────────────

def query_voice(question: str, lore: dict) -> str:
    terms   = _extract_terms(question, lore)
    target  = next((c for c in lore["characters"] if _entity_matches(c, terms)), None)
    voices_dir = LORE_DIR / "voices"

    if not target:
        # List all available voice profiles
        if not voices_dir.exists():
            return "No voice profiles yet. Run the pipeline first."
        profiles = sorted(voices_dir.glob("*.yaml"))
        if not profiles:
            return "No voice profiles yet. Run the pipeline first."
        lines = [f"Available voice profiles ({len(profiles)}):\n"]
        for p in profiles:
            data = _load_yaml(p)
            lines.append(f"- **{data.get('name', p.stem)}** — {data.get('formality_level','?')}, "
                         f"{data.get('vocabulary_register','?')}, {len(data.get('scenes_analyzed',[]))} scenes")
        return "\n".join(lines)

    import re as _re
    slug = _re.sub(r'\s+', '_', target.get("name", "").lower().strip())
    voice = _load_yaml(voices_dir / f"{slug}.yaml")
    if not voice:
        return f"No voice profile for {target.get('name')} yet. Run the pipeline on their scenes first."

    lines = [f"## Voice profile — {voice.get('name', '?').title()}\n"]
    lines.append(f"**Formality:** {voice.get('formality_level','?')} | "
                 f"**Register:** {voice.get('vocabulary_register','?')} | "
                 f"**Sentence length:** {voice.get('avg_sentence_length','?')}")
    ratio = voice.get("dialogue_vs_action_ratio")
    if ratio is not None:
        lines.append(f"**Dialogue/action ratio:** {ratio:.0%} dialogue")
    if voice.get("communication_style"):
        lines.append(f"\n**Communication style:** {voice['communication_style']}")
    if voice.get("speech_quirks"):
        lines.append(f"\n**Speech quirks:** {', '.join(voice['speech_quirks'])}")
    if voice.get("recurring_themes_in_speech"):
        lines.append(f"**Recurring themes:** {', '.join(voice['recurring_themes_in_speech'])}")
    if voice.get("roleplay_prompt"):
        lines.append(f"\n### Write in their voice:\n_{voice['roleplay_prompt']}_")
    lines.append(f"\n_Based on {len(voice.get('scenes_analyzed',[]))} scene(s)_")
    return "\n".join(lines)


# ── Handler: catch-up ─────────────────────────────────────────────────────────

_CATCHUP_PROMPT = """\
You are a storyteller briefing a player who missed some sessions.
Answer in the same language as the question.
Given the context below, write a concise catch-up from {char_name}'s point of view:
what they lived through, what they know, what they might be feeling now.
Keep it narrative and immersive, not a dry list.

WHAT {char_name_upper} EXPERIENCED (scenes they were in):
{known_events}

WHAT HAPPENED WHILE THEY WERE ABSENT (that may still be relevant):
{world_events}

REQUEST: {question}

CATCH-UP:"""


def query_catchup(question: str, lore: dict) -> str:
    terms  = _extract_terms(question, lore)
    target = next((c for c in lore["characters"] if _entity_matches(c, terms)), None)
    if not target and _STORE_OK:
        hits = search_chars(question, n=1)
        if hits:
            hit_name = hits[0]["name"].lower()
            target = next((c for c in lore["characters"] if c.get("name","").lower() == hit_name), None)
    if not target:
        return f"Character not identified. Terms: {terms}"

    char_name   = target.get("name", "")
    char_scenes = set(str(s) for s in (target.get("appearances") or []))

    known_lines, world_lines = [], []
    for scene_dir in sorted(ANALYSIS_DIR.iterdir()):
        if not scene_dir.is_dir():
            continue
        sid  = scene_dir.name
        what = _load_json(scene_dir / "what.json")
        summary = what.get("summary", "")
        events  = [e.get("description","") for e in (what.get("events") or [])
                   if e.get("type") in ("revelation","decision","emotional")]

        if sid in char_scenes:
            if summary:
                known_lines.append(f"[{sid}] {summary}")
            known_lines.extend(f"  • {e}" for e in events[:3])
        else:
            if events:
                world_lines.extend(f"[{sid}] {e}" for e in events[:2])

    context_known = "\n".join(known_lines[:20]) or "No scenes recorded."
    context_world = "\n".join(world_lines[:15]) or "Nothing significant."

    return call_llm(
        _CATCHUP_PROMPT.format(
            char_name=char_name, char_name_upper=char_name.upper(),
            known_events=context_known, world_events=context_world,
            question=question,
        ),
        num_predict=-1, num_ctx=12000,
    )


# ── Handler: temporal clustering ─────────────────────────────────────────────

def query_temporal(question: str, lore: dict) -> str:
    if not ANALYSIS_DIR.exists():
        return "No analyzed scenes. Run the pipeline first."

    # Build: character → {time_of_day: count}
    char_time: dict[str, dict[str, int]] = {}
    # Build: time_of_day → [scene_ids]
    time_scenes: dict[str, list] = {}

    for scene_dir in sorted(ANALYSIS_DIR.iterdir()):
        if not scene_dir.is_dir():
            continue
        when = _load_json(scene_dir / "when.json")
        who  = _load_json(scene_dir / "who.json")
        tod  = when.get("time_of_day", "unknown")
        sid  = scene_dir.name

        time_scenes.setdefault(tod, []).append(sid)
        for char in (who.get("characters") or []):
            char_time.setdefault(char, {})
            char_time[char][tod] = char_time[char].get(tod, 0) + 1

    lines = ["## Temporal clustering\n"]

    # Time distribution
    lines.append("### Scenes by time of day")
    for tod, scenes in sorted(time_scenes.items(), key=lambda x: -len(x[1])):
        lines.append(f"- **{tod}**: {len(scenes)} scene(s)")

    # Character temporal patterns
    lines.append("\n### Character temporal patterns")
    for char, times in sorted(char_time.items()):
        total = sum(times.values())
        if total < 2:
            continue
        dominant = max(times, key=times.get)
        pct = times[dominant] / total
        if pct >= 0.6:  # clear pattern
            lines.append(f"- **{char}**: mostly {dominant} ({pct:.0%} of {total} scenes)")

    # Locations by time
    loc_time: dict[str, dict[str, int]] = {}
    for scene_dir in sorted(ANALYSIS_DIR.iterdir()):
        if not scene_dir.is_dir():
            continue
        when  = _load_json(scene_dir / "when.json")
        where = _load_json(scene_dir / "where.json")
        tod   = when.get("time_of_day", "unknown")
        for loc in (where.get("locations") or []):
            loc_time.setdefault(loc, {})
            loc_time[loc][tod] = loc_time[loc].get(tod, 0) + 1

    strong_loc = [(loc, times) for loc, times in loc_time.items()
                  if sum(times.values()) >= 2 and max(times.values()) / sum(times.values()) >= 0.6]
    if strong_loc:
        lines.append("\n### Locations with temporal patterns")
        for loc, times in strong_loc:
            dominant = max(times, key=times.get)
            lines.append(f"- **{loc}**: mostly {dominant}")

    return "\n".join(lines)


# ── Handler: objectives / motivations ────────────────────────────────────────

_GOALS_PROMPT = """\
Based on the character data below, infer this character's goals and motivations.
Answer in the same language as the question.

CHARACTER: {name}
Psychology: {psych}
Beliefs: {beliefs}
Likes: {likes}
Dislikes: {dislikes}

Key decisions and revelations they were part of:
{decisions}

Relations with other characters:
{relations}

Produce:
- short_term_goal: what they are trying to achieve right now
- long_term_goal: their deeper ambition or life objective
- hidden_motivation: what might truly drive them beneath the surface
- obstacles: main things standing in their way
- narrative_role: their role in the story (protagonist/antagonist/catalyst/wildcard/etc.)

Answer in structured prose, one paragraph per point.

QUESTION: {question}"""


def query_goals(question: str, lore: dict) -> str:
    terms  = _extract_terms(question, lore)
    target = next((c for c in lore["characters"] if _entity_matches(c, terms)), None)
    if not target and _STORE_OK:
        hits = search_chars(question, n=1)
        if hits:
            hit_name = hits[0]["name"].lower()
            target = next((c for c in lore["characters"] if c.get("name","").lower() == hit_name), None)
    if not target:
        return f"Character not identified. Terms: {terms}"

    char_name = target.get("name", "")
    char_scenes = set(str(s) for s in (target.get("appearances") or []))

    decisions, relations = [], []
    for sid in char_scenes:
        what = _load_json(ANALYSIS_DIR / sid / "what.json")
        how  = _load_json(ANALYSIS_DIR / sid / "how.json")
        for ev in (what.get("events") or []):
            if ev.get("type") in ("decision", "revelation"):
                decisions.append(f"[{sid}] {ev.get('description','')[:100]}")
        for rel in (how.get("character_relations") or []):
            if char_name.lower() in (rel.get("from_char","") + rel.get("to_char","")).lower():
                other = rel.get("to_char") if char_name.lower() in rel.get("from_char","").lower() else rel.get("from_char")
                relations.append(f"{other}: {rel.get('relation_type')} ({rel.get('sentiment')})")

    return call_llm(
        _GOALS_PROMPT.format(
            name=char_name,
            psych=target.get("description_psychological","?")[:200],
            beliefs=", ".join(target.get("beliefs",[])[:5]) or "none",
            likes=", ".join(target.get("likes",[])[:5]) or "none",
            dislikes=", ".join(target.get("dislikes",[])[:5]) or "none",
            decisions="\n".join(decisions[:12]) or "none recorded",
            relations="\n".join(set(relations[:10])) or "none recorded",
            question=question,
        ),
        num_predict=-1, num_ctx=8192,
    )


# ── Handler: dramatic irony ───────────────────────────────────────────────────

def query_irony(question: str, lore: dict) -> str:
    # Build: revelation → {scene, description, chars_present}
    revelations: list[dict] = []
    for scene_dir in sorted(ANALYSIS_DIR.iterdir()):
        if not scene_dir.is_dir():
            continue
        what = _load_json(scene_dir / "what.json")
        who  = _load_json(scene_dir / "who.json")
        chars_present = set(c.lower() for c in (who.get("characters") or []))
        for ev in (what.get("events") or []):
            if ev.get("type") in ("revelation", "decision"):
                revelations.append({
                    "scene":   scene_dir.name,
                    "desc":    ev.get("description",""),
                    "present": chars_present,
                    "chars":   set(c.lower() for c in (ev.get("characters") or [])),
                })

    # Build: char → all scenes they appeared in
    char_scenes: dict[str, set] = {}
    for c in lore["characters"]:
        name = c.get("name","").lower()
        char_scenes[name] = set(str(s) for s in (c.get("appearances") or []))

    irony_cases = []
    for rev in revelations:
        # Characters who know this revelation (were present)
        knowers = rev["present"]
        # Characters who don't know: were NEVER in this scene but are connected
        for char_name, scenes in char_scenes.items():
            if char_name in knowers:
                continue
            # Check if this char interacts with knowers in other scenes
            if not (knowers & set(
                c.lower() for c in
                (lore["characters"][0].get("related_characters", []) if lore["characters"] else [])
            )):
                # simpler: char appears in other scenes, is a known character
                if scenes and char_name not in knowers:
                    irony_cases.append({
                        "knows": sorted(knowers),
                        "doesnt_know": char_name,
                        "revelation": rev["desc"],
                        "scene": rev["scene"],
                    })

    # Deduplicate and limit
    seen, unique = set(), []
    for case in irony_cases:
        key = f"{case['doesnt_know']}|{case['revelation'][:40]}"
        if key not in seen:
            seen.add(key)
            unique.append(case)

    if not unique:
        return "No clear dramatic irony detected yet (need more scenes with revelations and multiple characters)."

    lines = [f"## Dramatic irony — {len(unique[:12])} case(s)\n"]
    for c in unique[:12]:
        knows_str = ", ".join(c["knows"][:3])
        lines.append(
            f"- **{c['doesnt_know']}** doesn't know: _{c['revelation'][:100]}_\n"
            f"  (known to: {knows_str} — scene `{c['scene']}`)"
        )
    return "\n".join(lines)


# ── Handler: missing scenes ───────────────────────────────────────────────────

_MISSING_PROMPT = """\
Analyze these causal links and narrative references from a roleplay story.
Identify events that are MENTIONED as having happened but have no corresponding scene in the corpus.
These are "off-screen" events — referenced as past facts but never directly shown.

Causal links and context:
{links_text}

For each missing scene you identify:
- what happened (inferred from references)
- which characters were likely involved
- why it matters narratively (what it caused or explains)

Answer in the same language as the question. Be concise.

QUESTION: {question}"""


def query_missing(question: str, lore: dict) -> str:
    past_refs = []
    for scene_dir in sorted(ANALYSIS_DIR.iterdir()):
        if not scene_dir.is_dir():
            continue
        how = _load_json(scene_dir / "how.json")
        for link in (how.get("links") or []):
            desc = link.get("description","")
            # Past tense references suggest off-screen events
            if re.search(r'\b(avait|avaient|était|s.était|avait été|had|was|had been|before|prior|previously|autrefois|jadis|dans le passé)\b', desc, re.I):
                past_refs.append(
                    f"[{scene_dir.name}] {link.get('from_element','')} → {link.get('to_element','')}: {desc[:120]}"
                )
        # Also check what.json for past references
        what = _load_json(scene_dir / "what.json")
        for ev in (what.get("events") or []):
            if ev.get("type") == "revelation":
                desc = ev.get("description","")
                if re.search(r'\b(révélé|découvert|appris|revealed|learned|discovered)\b', desc, re.I):
                    past_refs.append(f"[{scene_dir.name}] REVELATION: {desc[:120]}")

    if not past_refs:
        return "No clear off-screen event references detected yet."

    links_text = "\n".join(past_refs[:25])
    return call_llm(
        _MISSING_PROMPT.format(links_text=links_text, question=question),
        num_predict=-1, num_ctx=8192,
    )


# ── Handler: lore continuity validation ──────────────────────────────────────

_VALIDATE_PROMPT = """\
You are a narrative continuity checker for a roleplay story.
Compare the CONFIRMED FACTS against the NEW SCENE content.
Identify any contradiction, impossibility, or inconsistency.
Be specific: quote both the confirmed fact and the conflicting element.
If everything is consistent, say so clearly.
Answer in the same language as the question.

CONFIRMED FACTS (from validated scenes):
{confirmed_facts}

NEW SCENE CONTENT:
{new_content}

QUESTION: {question}

CONTINUITY ANALYSIS:"""


def query_validate(question: str, lore: dict) -> str:
    terms = _extract_terms(question, lore)

    # Load confirmed facts
    confirmed_facts = []
    if CONFIRMED_DIR.exists():
        for yf in sorted(CONFIRMED_DIR.glob("*.yaml")):
            try:
                data = yaml.safe_load(yf.read_text(encoding="utf-8")) or {}
            except Exception:
                continue
            if data.get("status") != "confirmed":
                continue
            sid = data.get("scene_id", yf.stem)
            what = data.get("what") or {}
            who  = data.get("who") or {}
            if what.get("summary"):
                confirmed_facts.append(f"[{sid}] {what['summary']}")
            for char in (who.get("details") or []):
                name = char.get("canonical_name","")
                job  = char.get("job","")
                if name and job:
                    confirmed_facts.append(f"  FACT: {name} is {job}")

    if not confirmed_facts:
        return "No confirmed scenes yet. Annotate scenes first in the Annotate tab."

    # Find the most recent unconfirmed scene to validate
    confirmed_ids = {f.stem for f in CONFIRMED_DIR.glob("*.yaml")} if CONFIRMED_DIR.exists() else set()
    new_scene = None
    for scene_dir in sorted(ANALYSIS_DIR.iterdir(), reverse=True):
        if scene_dir.is_dir() and scene_dir.name not in confirmed_ids:
            new_scene = scene_dir
            break

    if not new_scene:
        return "All analyzed scenes have been confirmed. Nothing to validate."

    what = _load_json(new_scene / "what.json")
    who  = _load_json(new_scene / "who.json")
    new_content = f"Scene: {new_scene.name}\n"
    new_content += f"Summary: {what.get('summary','')}\n"
    new_content += f"Characters: {', '.join(who.get('characters',[]))}\n"
    for ev in (what.get("events") or [])[:8]:
        new_content += f"• {ev.get('description','')}\n"

    return call_llm(
        _VALIDATE_PROMPT.format(
            confirmed_facts="\n".join(confirmed_facts[:30]),
            new_content=new_content,
            question=question,
        ),
        num_predict=-1, num_ctx=10000,
    )


# ── Shared persona builder ────────────────────────────────────────────────────

def _build_persona(char: dict) -> tuple[str, str]:
    """Return (persona_block, voice_block) for a character."""
    import re as _re
    char_name = char.get("name", "?")

    # Character block
    lines = [f"Name: {char_name}"]
    if char.get("job"):
        lines.append(f"Role: {char['job']}")
    if char.get("description_psychological"):
        lines.append(f"Psychology: {char['description_psychological']}")
    if char.get("beliefs"):
        lines.append(f"Beliefs: {', '.join(char['beliefs'][:6])}")
    if char.get("likes"):
        lines.append(f"Likes: {', '.join(char['likes'][:5])}")
    if char.get("dislikes"):
        lines.append(f"Dislikes: {', '.join(char['dislikes'][:5])}")
    ep = char.get("emotional_polarity") or {}
    if ep.get("dominant_emotions"):
        lines.append(f"Dominant emotions: {', '.join(ep['dominant_emotions'][:3])}")
    if ep.get("emotional_triggers"):
        lines.append(f"Emotional triggers: {', '.join(ep['emotional_triggers'][:4])}")
    # Personality axes: only non-zero
    axes = char.get("personality_axes") or {}
    notable = {k: v for k, v in axes.items() if v and abs(int(v)) >= 1}
    if notable:
        axis_strs = []
        for k, v in list(notable.items())[:6]:
            label = k.replace("_", " ")
            side = label.split(" vs ")[0] if v < 0 else (label.split(" vs ")[1] if " vs " in label else label)
            strength = "strongly" if abs(v) == 2 else "somewhat"
            axis_strs.append(f"{strength} {side}")
        lines.append(f"Personality: {', '.join(axis_strs)}")

    persona_block = "\n".join(lines)

    # Voice block
    slug = _re.sub(r'\s+', '_', char_name.lower().strip())
    voice = _load_yaml(LORE_DIR / "voices" / f"{slug}.yaml")
    voice_lines = []
    if voice.get("roleplay_prompt"):
        voice_lines.append(voice["roleplay_prompt"])
    else:
        if voice.get("formality_level"):
            voice_lines.append(f"Formality: {voice['formality_level']}")
        if voice.get("vocabulary_register"):
            voice_lines.append(f"Register: {voice['vocabulary_register']}")
        if voice.get("avg_sentence_length"):
            voice_lines.append(f"Sentence length: {voice['avg_sentence_length']}")
        if voice.get("speech_quirks"):
            voice_lines.append(f"Speech quirks: {', '.join(voice['speech_quirks'][:4])}")
        if voice.get("communication_style"):
            voice_lines.append(f"Style: {voice['communication_style']}")

    voice_block = "\n".join(voice_lines) if voice_lines else "No voice profile yet — improvise from personality."
    return persona_block, voice_block


def _recent_scene_context(char: dict, n: int = 4) -> str:
    appearances = sorted(str(s) for s in (char.get("appearances") or []))[-n:]
    parts = []
    for sid in appearances:
        what = _load_json(ANALYSIS_DIR / sid / "what.json")
        summary = what.get("summary", "")
        if summary:
            parts.append(f"[{sid}] {summary}")
    return "\n".join(parts) or "No recent scenes recorded."


# ── Handler: roleplay (in-character response) ─────────────────────────────────

_ROLEPLAY_PROMPT = """\
You are {name}, a character from a collaborative roleplay story.
Stay strictly in character. Respond ONLY as {name} would — in their voice, with their personality, beliefs, and emotional state.
Answer in the same language as the question.

── CHARACTER ────────────────────────────────────────────────────────────────────
{persona}

── VOICE & STYLE ────────────────────────────────────────────────────────────────
{voice}

── RECENT CONTEXT (your last scenes) ───────────────────────────────────────────
{context}

── QUESTION ─────────────────────────────────────────────────────────────────────
{question}

{name_upper}:"""


def query_roleplay(question: str, lore: dict) -> str:
    terms  = _extract_terms(question, lore)
    target = next((c for c in lore["characters"] if _entity_matches(c, terms)), None)
    # Semantic fallback: if no exact name match, find closest character by embedding
    if not target and _STORE_OK:
        hits = search_chars(question, n=1)
        if hits:
            hit_name = hits[0]["name"].lower()
            target = next((c for c in lore["characters"] if (c.get("name","").lower() == hit_name)), None)
    if not target:
        return (
            f"Could not identify a character in the question. Terms detected: {terms}\n"
            "Try: 'que dirait Rhys à propos de...'"
        )

    char_name = target.get("name", "?")
    persona, voice = _build_persona(target)
    context = _recent_scene_context(target)

    return call_llm(
        _ROLEPLAY_PROMPT.format(
            name=char_name,
            name_upper=char_name.upper(),
            persona=persona,
            voice=voice,
            context=context,
            question=question,
        ),
        num_predict=-1,
        num_ctx=8192,
    )


# ── Handler: react (hypothetical situational reaction) ───────────────────────

_REACT_PROMPT = """\
Analyze how {name} would react to the hypothetical situation described below.
Ground your answer in their character — use their personality, beliefs, emotional triggers, and past behavior as evidence.
Answer in the same language as the question.

── CHARACTER ────────────────────────────────────────────────────────────────────
{persona}

── PAST BEHAVIOR (decisions & actions in their scenes) ─────────────────────────
{decisions}

── HYPOTHETICAL SITUATION ───────────────────────────────────────────────────────
{question}

Describe in order:
1. Immediate emotional reaction (gut feeling, physical response)
2. What they would actually do (concrete action)
3. What they would say — write it in their voice
4. Why — grounded in who they are

REACTION:"""


def query_react(question: str, lore: dict) -> str:
    terms  = _extract_terms(question, lore)
    target = next((c for c in lore["characters"] if _entity_matches(c, terms)), None)
    if not target and _STORE_OK:
        hits = search_chars(question, n=1)
        if hits:
            hit_name = hits[0]["name"].lower()
            target = next((c for c in lore["characters"] if (c.get("name","").lower() == hit_name)), None)
    if not target:
        return (
            f"Could not identify a character in the question. Terms detected: {terms}\n"
            "Try: 'comment réagirait Rhys si...'"
        )

    char_name = target.get("name", "?")
    persona, voice = _build_persona(target)

    # Collect past decisions/actions
    char_scenes = set(str(s) for s in (target.get("appearances") or []))
    decisions = []
    for sid in sorted(char_scenes):
        what = _load_json(ANALYSIS_DIR / sid / "what.json")
        for ev in (what.get("events") or []):
            if ev.get("type") in ("decision", "action", "emotional"):
                decisions.append(f"[{sid}] {ev.get('description','')[:100]}")
    decisions_text = "\n".join(decisions[:15]) or "No behavior data yet."

    # Also inject voice style into persona for in-character quote
    persona_with_voice = persona + "\n\nVoice & style:\n" + voice

    return call_llm(
        _REACT_PROMPT.format(
            name=char_name,
            persona=persona_with_voice,
            decisions=decisions_text,
            question=question,
        ),
        num_predict=-1,
        num_ctx=10000,
    )


# ── Public API ─────────────────────────────────────────────────────────────────

def answer(question: str, verbose: bool = False) -> str:
    lore   = load_all_lore()
    intent = detect_intent(question)

    if verbose:
        print(f"  intent: {intent}")

    handlers = {
        "analytical":  query_analytical,
        "events":      query_events,
        "links":       query_links,
        "combo":       query_combo,
        "arc":         query_arc,
        "knows":       query_knows,
        "story":       query_story,
        "importance":  query_importance,
        "continuity":  query_continuity,
        "foreshadow":  query_foreshadow,
        "creative":    query_creative,
        "voice":       query_voice,
        "catchup":     query_catchup,
        "temporal":    query_temporal,
        "tension":     lambda q, l: "Use the Graph tab → Tension curve.",
        "irony":       query_irony,
        "goals":       query_goals,
        "power":       lambda q, l: "Use the Graph tab → Power map.",
        "missing":     query_missing,
        "validate":    query_validate,
        "roleplay":    query_roleplay,
        "react":       query_react,
    }
    return handlers.get(intent, query_entity)(question, lore)


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if len(sys.argv) > 1:
        q = " ".join(sys.argv[1:])
        print(answer(q, verbose=True))
    else:
        print("RP_IA Query — type your question (empty line to quit)\n")
        while True:
            try:
                q = input("? ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not q:
                break
            print()
            print(answer(q, verbose=False))
            print()
