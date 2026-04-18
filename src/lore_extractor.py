"""
LLM-based lore extraction from RP scene text.
Sends scenes to Mistral and merges structured knowledge into lore.yaml.
Extracted entries are marked confidence: "llm" — promote to "verified" after review.
"""

import json
import sys
from pathlib import Path

import requests
import yaml


OLLAMA_URL = "http://localhost:11434/api/generate"
LLM_MODEL = "mistral"
LORE_PATH = Path("config/lore.yaml")

EXTRACTION_SYSTEM = (
    "Tu es un extracteur de lore pour jeux de rôle narratif écrit. "
    "Tu analyses des scènes et retournes UNIQUEMENT du JSON valide, sans texte supplémentaire. "
    "Tu n'inventes rien : tu extrais seulement ce qui est explicite ou fortement implicite."
)

EXTRACTION_PROMPT = """\
Analyse cette scène de roleplay. Extrais toutes les informations d'univers identifiables.

─── ÉTAT TEMPOREL ───
Chaque entité et chaque relation a un état :
  "est"  = existe actuellement dans la scène
  "fut"  = existait dans le passé, n'existe plus (personnage mort, lieu détruit, relation rompue…)
  "sera" = projeté, planifié, anticipé (intention, promesse, menace future…)
Par défaut "est". Utilise "fut" si le texte emploie le passé pour décrire quelque chose de révolu.

─── ÉPISTÉMOLOGIE DES PERSONNAGES ───
Pour chaque personnage, extrais ce qu'il sait, ignore ou croit :
  sait       = fait établi dont le personnage a connaissance
  ne_sait_pas = fait narratif que le personnage ignore (tension dramatique)
  croit      = croyance possiblement fausse ; précise source_culture ou source_rule si applicable

─── FORMAT JSON ATTENDU ───
Retourne un objet JSON avec cette structure (laisse {{}} ou [] si rien à signaler) :

{{
  "characters": {{
    "NomExact": {{"aliases": [], "description": "", "state": "est"}}
  }},
  "places": {{
    "NomLieu": {{"description": "", "state": "est"}}
  }},
  "events": {{
    "id_court": {{"label": "", "description": "", "timestamp": null, "state": "fut"}}
  }},
  "objects": {{
    "NomObjet": {{"description": "", "state": "est"}}
  }},
  "cultures": {{
    "NomCulture": {{"description": "", "state": "est"}}
  }},
  "intentions": {{
    "id_court": {{
      "character": "NomPersonnage",
      "description": "",
      "status": "en_cours",
      "state": "est"
    }}
  }},
  "narrative_axes": {{
    "id_court": {{
      "label": "",
      "description": "thème fédérateur déduit, pas une déclaration explicite",
      "characters": [], "places": [], "events": [], "objects": [],
      "cultures": [], "intentions": [], "themes": [],
      "state": "est"
    }}
  }},
  "character_knowledge": {{
    "NomPersonnage": {{
      "sait": [
        {{"description": "", "entity": null, "established": "source_scene"}}
      ],
      "ne_sait_pas": [
        {{"description": "", "entity": null}}
      ],
      "croit": [
        {{
          "description": "",
          "entity": null,
          "source_culture": null,
          "source_rule": null,
          "verified": false
        }}
      ]
    }}
  }},
  "universe_rules": [
    "fait d'univers notable : langue, magie, loi sociale, coutume, proverbe"
  ],
  "relations": [
    {{
      "from": "source",
      "from_type": "character|place|event|object|culture|intention|narrative_axis",
      "rel": "type_de_relation",
      "to": "cible",
      "to_type": "character|place|event|object|culture|intention|narrative_axis",
      "state": "est",
      "confidence": "high|low"
    }}
  ]
}}

Types de relations :
  character–character  : frère_de sœur_de parent_de enfant_de ami_de ennemi_de rival_de
                         allié_de amant_de maître_de élève_de membre_de
  character–place      : présent_à habite vient_de fuit_de se_dirige_vers
  character–event      : participe_à cause subit témoin_de
  character–object     : possède utilise cherche a_créé a_perdu
  character–culture    : appartient_à pratique rejette
  character–intention  : a_intention
  place–place          : contient adjacent_à fait_partie_de
  place–event          : lieu_de
  place–culture        : territoire_de influencé_par
  event–event          : précède cause suit
  event–object         : implique
  event–culture        : reflète transgresse
  object–place         : situé_à vient_de
  object–culture       : artefact_de
  intention–event      : vise a_déclenché
  culture–rule         : dicte
  narrative_axis–*     : inclut
  narrative_axis–axis  : croise

Scène :
---
{text}
---"""


# ── Ollama call ───────────────────────────────────────────────────────────────

def _call_mistral(text: str) -> dict:
    prompt = EXTRACTION_SYSTEM + "\n\n" + EXTRACTION_PROMPT.format(text=text)
    try:
        resp = requests.post(
            OLLAMA_URL,
            json={"model": LLM_MODEL, "prompt": prompt, "format": "json", "stream": False},
            timeout=120,
        )
        resp.raise_for_status()
        raw = resp.json().get("response", "{}")
        return json.loads(raw)
    except requests.RequestException as e:
        print(f"  [lore_extractor] Ollama error: {e}", file=sys.stderr)
        return {}
    except json.JSONDecodeError as e:
        print(f"  [lore_extractor] JSON parse error: {e}", file=sys.stderr)
        return {}


# ── Merge logic ───────────────────────────────────────────────────────────────

def _load_lore() -> dict:
    if not LORE_PATH.exists():
        return {
            "characters": {}, "places": {}, "events": {}, "objects": {},
            "cultures": {}, "intentions": {}, "narrative_axes": {},
            "character_knowledge": {}, "universe_rules": [], "relations": [],
        }
    with open(LORE_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _save_lore(data: dict):
    LORE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(LORE_PATH, "w", encoding="utf-8") as f:
        yaml.dump(data, f, allow_unicode=True, sort_keys=False)


def _merge_axis(existing: dict, new: dict) -> dict:
    merged = dict(existing)
    for field in ("characters", "places", "events", "objects", "cultures", "intentions", "themes"):
        existing_list = merged.get(field) or []
        new_items = [x for x in (new.get(field) or []) if x not in existing_list]
        merged[field] = existing_list + new_items
    # Temporal state: allow est→fut transition (degradation only)
    state_order = {"sera": 0, "est": 1, "fut": 2}
    existing_state = merged.get("state", "est")
    new_state = new.get("state", "est")
    if state_order.get(new_state, 1) > state_order.get(existing_state, 1):
        merged["state"] = new_state
    return merged


def _merge_knowledge(existing_profile: dict, new_profile: dict, source: str) -> dict:
    merged = dict(existing_profile)
    for bucket in ("sait", "ne_sait_pas", "croit"):
        existing_items = merged.get(bucket) or []
        existing_descs = {item.get("description") for item in existing_items}
        for item in new_profile.get(bucket) or []:
            desc = item.get("description")
            if desc and desc not in existing_descs:
                existing_items.append({**item, "source": source, "confidence": "llm"})
                existing_descs.add(desc)
        merged[bucket] = existing_items
    return merged


def merge_into_lore(extracted: dict, source: str) -> int:
    """
    Merges extracted dict into lore.yaml.
    - New nodes get confidence='llm' and state defaulting to 'est'.
    - Existing nodes: temporal state is allowed to degrade (est→fut) but not reverse.
    - Relations dedup by (from, rel, to, state) — history across states is preserved.
    - Character knowledge is union-merged per bucket (sait / ne_sait_pas / croit).
    Returns count of new items added.
    """
    if not extracted:
        return 0

    lore = _load_lore()
    added = 0
    state_order = {"sera": 0, "est": 1, "fut": 2}

    # Simple node sections
    for section in ("characters", "places", "events", "objects", "cultures", "intentions"):
        if lore.get(section) is None:
            lore[section] = {}
        for name, data in (extracted.get(section) or {}).items():
            if not name:
                continue
            data = data or {}
            if name not in lore[section]:
                lore[section][name] = {**data, "state": data.get("state", "est"),
                                       "source": source, "confidence": "llm"}
                added += 1
            else:
                # Allow temporal degradation only
                existing_state = lore[section][name].get("state", "est")
                new_state = data.get("state", "est")
                if state_order.get(new_state, 1) > state_order.get(existing_state, 1):
                    lore[section][name]["state"] = new_state

    # Narrative axes — union-merge entity lists, allow state degradation
    if lore.get("narrative_axes") is None:
        lore["narrative_axes"] = {}
    for axis_id, axis_data in (extracted.get("narrative_axes") or {}).items():
        if not axis_id:
            continue
        axis_data = axis_data or {}
        if axis_id not in lore["narrative_axes"]:
            lore["narrative_axes"][axis_id] = {**axis_data,
                                                "state": axis_data.get("state", "est"),
                                                "source": source, "confidence": "llm"}
            added += 1
        else:
            lore["narrative_axes"][axis_id] = _merge_axis(lore["narrative_axes"][axis_id], axis_data)

    # Character knowledge
    if lore.get("character_knowledge") is None:
        lore["character_knowledge"] = {}
    for char_name, profile in (extracted.get("character_knowledge") or {}).items():
        if not char_name or not profile:
            continue
        if char_name not in lore["character_knowledge"]:
            lore["character_knowledge"][char_name] = {"sait": [], "ne_sait_pas": [], "croit": []}
        before = sum(
            len(lore["character_knowledge"][char_name].get(b, []))
            for b in ("sait", "ne_sait_pas", "croit")
        )
        lore["character_knowledge"][char_name] = _merge_knowledge(
            lore["character_knowledge"][char_name], profile, source
        )
        after = sum(
            len(lore["character_knowledge"][char_name].get(b, []))
            for b in ("sait", "ne_sait_pas", "croit")
        )
        added += after - before

    # Universe rules
    existing_rules: list = lore.get("universe_rules") or []
    existing_set = set(existing_rules)
    for rule in extracted.get("universe_rules") or []:
        if rule and rule not in existing_set:
            existing_rules.append(rule)
            existing_set.add(rule)
            added += 1
    lore["universe_rules"] = existing_rules

    # Relations — dedup by (from, rel, to, state) to preserve temporal history
    existing_rels: list = lore.get("relations") or []
    existing_keys = {
        (r.get("from"), r.get("rel"), r.get("to"), r.get("state", "est"))
        for r in existing_rels
    }
    for rel in extracted.get("relations") or []:
        state = rel.get("state", "est")
        key = (rel.get("from"), rel.get("rel"), rel.get("to"), state)
        if None in key[:3] or key in existing_keys:
            continue
        existing_rels.append({**rel, "state": state, "source": source})
        existing_keys.add(key)
        added += 1
    lore["relations"] = existing_rels

    _save_lore(lore)
    return added


# ── Public API ────────────────────────────────────────────────────────────────

def extract_and_merge(scene_text: str, source: str, verbose: bool = False) -> int:
    if verbose:
        print(f"  [lore_extractor] extracting {source} …", end=" ", flush=True)
    extracted = _call_mistral(scene_text)
    added = merge_into_lore(extracted, source)
    if verbose:
        print(f"{added} new items")
    return added


def extract_from_file(filepath: Path, scene_texts: list[tuple[str, str]], verbose: bool = True) -> int:
    total = 0
    for text, source in scene_texts:
        if len(text.strip()) < 50:
            continue
        total += extract_and_merge(text, source, verbose=verbose)
    if verbose:
        print(f"  → {total} lore items from {filepath.name}")
    return total


# ── Standalone CLI ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent))

    from preprocessing import load_config, build_alias_map, process_export, group_into_scenes, scene_to_text
    from lore import load_lore

    exports_dir = sys.argv[1] if len(sys.argv) > 1 else "data/exports"
    exports_path = Path(exports_dir)
    json_files = list(exports_path.glob("**/*.json"))

    if not json_files:
        print(f"No JSON files found in {exports_dir}")
        sys.exit(0)

    config = load_config()
    lore = load_lore()
    alias_map = build_alias_map(config)
    alias_map.update(lore.character_aliases())

    grand_total = 0
    for filepath in json_files:
        print(f"Processing: {filepath.name}")
        messages = process_export(filepath, config, lore=lore)
        scenes = group_into_scenes(messages)
        pairs = [
            (text, f"{filepath.stem}:{meta.get('start', '')}")
            for scene in scenes
            for text, meta in [scene_to_text(scene, alias_map)]
        ]
        grand_total += extract_from_file(filepath, pairs, verbose=True)

    print(f"\nTotal added to {LORE_PATH}: {grand_total}")
