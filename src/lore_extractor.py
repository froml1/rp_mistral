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

from analyzer import compress_scene_text


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
            json={"model": LLM_MODEL, "prompt": prompt, "format": "json", "stream": False,
                  "keep_alive": -1,
                  "options": {"temperature": 0, "top_k": 1, "num_predict": -1, "num_ctx": 8192}},
            timeout=300,
        )
        resp.raise_for_status()
        raw = resp.json().get("response", "{}")
        print(f"  [lore_extractor] réponse {len(raw)} chars")
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


_CONFIDENCE_RANK = {"verified": 3, "high": 2, "low": 1, "llm": 0}


def _conf_gt(a: str, b: str) -> bool:
    return _CONFIDENCE_RANK.get(a, 0) > _CONFIDENCE_RANK.get(b, 0)


def _merge_knowledge(existing_profile: dict, new_profile: dict, source: str) -> dict:
    """
    Merge per bucket. Contradiction = même entity dans deux buckets différents.
    Si new_confidence > existing_confidence, déplacer vers le nouveau bucket.
    """
    merged = {b: list(existing_profile.get(b) or []) for b in ("sait", "ne_sait_pas", "croit")}

    for new_bucket in ("sait", "ne_sait_pas", "croit"):
        for item in new_profile.get(new_bucket) or []:
            entity = item.get("entity")
            desc   = item.get("description")
            if not desc:
                continue
            new_conf = item.get("confidence", "llm")

            # Cherche une contradiction : même entity (ou même desc) dans un autre bucket
            contradiction_found = False
            if entity:
                for other_bucket in ("sait", "ne_sait_pas", "croit"):
                    if other_bucket == new_bucket:
                        continue
                    for idx, existing in enumerate(merged[other_bucket]):
                        if existing.get("entity") == entity:
                            existing_conf = existing.get("confidence", "llm")
                            if _conf_gt(new_conf, existing_conf):
                                merged[other_bucket].pop(idx)
                                merged[new_bucket].append({**item, "source": source, "confidence": new_conf})
                            contradiction_found = True
                            break
                    if contradiction_found:
                        break

            if contradiction_found:
                continue

            # Pas de contradiction — ajout simple si description absente
            existing_descs = {i.get("description") for i in merged[new_bucket]}
            if desc not in existing_descs:
                merged[new_bucket].append({**item, "source": source, "confidence": new_conf})

    return merged


def merge_into_lore(extracted: dict, source: str) -> int:
    """
    Merges extracted dict into lore.yaml.
    - New nodes get confidence='llm' and state defaulting to 'est'.
    - Existing nodes: temporal state is allowed to degrade (est→fut) but not reverse.
    - Relations dedup by (from, rel, to) — contradiction (state différent) remplace si new_conf > existing_conf.
    - Character knowledge : contradiction inter-bucket (même entity) remplace si new_conf > existing_conf.
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
        raw = extracted.get(section) or {}
        if not isinstance(raw, dict):
            raw = {}
        for name, data in raw.items():
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
    raw_axes = extracted.get("narrative_axes") or {}
    if not isinstance(raw_axes, dict):
        raw_axes = {}
    for axis_id, axis_data in raw_axes.items():
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
    raw_ck = extracted.get("character_knowledge") or {}
    if not isinstance(raw_ck, dict):
        raw_ck = {}
    for char_name, profile in raw_ck.items():
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

    # Relations — dedup by (from, rel, to)
    # Contradiction = même triplet, state différent → remplace si new_confidence > existing
    # State identique → doublon, ignoré
    # State différent + confidence égale/inférieure → garde les deux (progression temporelle possible)
    def _rel_triplet(r: dict) -> tuple:
        return (str(r.get("from") or ""), str(r.get("rel") or ""), str(r.get("to") or ""))

    def _rel_key(r: dict) -> tuple:
        return (*_rel_triplet(r), str(r.get("state") or "est"))

    existing_rels: list = [r for r in (lore.get("relations") or []) if isinstance(r, dict)]
    existing_by_triplet: dict[tuple, int] = {
        _rel_triplet(r): i for i, r in enumerate(existing_rels)
    }
    existing_keys = {_rel_key(r) for r in existing_rels}

    for rel in extracted.get("relations") or []:
        if not isinstance(rel, dict):
            continue
        triplet = _rel_triplet(rel)
        if "" in triplet:
            continue
        key = _rel_key(rel)
        new_conf = rel.get("confidence", "llm")

        if key in existing_keys:
            continue  # doublon exact

        if triplet in existing_by_triplet:
            # Contradiction : même triplet, state différent
            idx = existing_by_triplet[triplet]
            existing_conf = existing_rels[idx].get("confidence", "llm")
            if _conf_gt(new_conf, existing_conf):
                existing_keys.discard(_rel_key(existing_rels[idx]))
                existing_rels[idx] = {**rel, "source": source}
                existing_keys.add(key)
                existing_by_triplet[triplet] = idx
                added += 1
            # sinon : contradiction mais confidence inférieure → garder les deux
            else:
                existing_rels.append({**rel, "source": source})
                existing_keys.add(key)
                added += 1
        else:
            existing_rels.append({**rel, "source": source})
            existing_keys.add(key)
            existing_by_triplet[triplet] = len(existing_rels) - 1
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