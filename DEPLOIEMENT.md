# Déploiement — RP_IA

## Prérequis machine

| Composant | Minimum | Recommandé |
|---|---|---|
| RAM | 8 Go | 16 Go |
| Stockage | 10 Go libres | 20 Go |
| OS | Linux / macOS / Windows | Linux |
| Python | 3.11+ | 3.12 |
| GPU | Non requis | Non requis (CPU suffit) |

---

## Étape 1 — Installer Ollama

```bash
# Linux / macOS
curl -fsSL https://ollama.com/install.sh | sh

# Télécharger le modèle (~4 Go, une seule fois)
ollama pull mistral
```

Windows : télécharger l'installeur depuis https://ollama.com/download

> Si `address already in use` : Ollama tourne déjà, passez à la suite.

---

## Étape 2 — Préparer l'environnement Python

```bash
cd /chemin/vers/RP_IA
python3 -m venv .venv
source .venv/bin/activate        # Windows : .venv\Scripts\activate
pip install -r requirements.txt
```

---

## Étape 3 — Placer les exports Discord

Exporter vos salons avec [DiscordChatExporter](https://github.com/Tyrrrz/DiscordChatExporter) au format **JSON**.

Déposer les fichiers dans :
```
data/exports/
  arc-1-prologue.json
  arc-2-traversee.json
  ...
```

C'est le **seul dossier à remplir manuellement**.

---

## Étape 4 — Lancer la pipeline

```bash
# Pipeline complète (étapes 1 → 4)
.venv/bin/python src/pipeline.py

# Reprendre à partir d'une étape
.venv/bin/python src/pipeline.py --from-step 3

# Une seule étape
.venv/bin/python src/pipeline.py --only-step 4

# Retraiter une scène spécifique (étape 4)
.venv/bin/python src/pipeline.py --only-step 4 --scene nom_scene_000
```

### Ce que fait chaque étape

| Étape | Nom | Entrée | Sortie |
|---|---|---|---|
| 1 | Purge | `data/exports/` | `data/purged/` |
| 2 | Translate | `data/purged/` | `data/translated/` |
| 3 | Subdivide | `data/translated/` | `data/scenes/` |
| 4 | Analyze | `data/scenes/` | `data/analysis/` + `data/lore/` |

La pipeline est **idempotente** : elle saute les fichiers déjà traités. Ajouter de nouveaux exports et relancer ne retraite que le nouveau contenu.

---

## Étape 5 — Lancer l'interface

```bash
.venv/bin/python src/interface.py
```

Ouvrir : **http://localhost:7860**

### Onglets disponibles

| Onglet | Usage |
|---|---|
| **Query** | Poser des questions sur l'univers RP |
| **Pipeline** | Démarrer / surveiller la pipeline sans terminal |
| **Lore** | Parcourir les fiches personnages, lieux, concepts |

---

## Structure des données produites

```
data/
├── exports/              ← VOS FICHIERS JSON (à déposer ici)
├── purged/               ← messages RP filtrés
├── translated/           ← messages traduits en anglais
├── scenes/               ← scènes découpées
├── analysis/
│   └── {scene_id}/
│       ├── when.json     ← contexte temporel
│       ├── where.json    ← lieux
│       ├── who.json      ← personnages
│       ├── which.json    ← concepts/factions/objets
│       ├── what.json     ← événements
│       └── how.json      ← liens causaux
└── lore/
    ├── characters/       ← une fiche YAML par personnage
    ├── places/           ← une fiche YAML par lieu
    ├── concepts/         ← une fiche YAML par concept
    └── how_context.yaml  ← synthèse narrative cumulée
```

---

## Structure du projet

```
RP_IA/
├── data/
│   └── exports/          ← exports Discord à déposer ici
├── src/
│   ├── pipeline.py       ← point d'entrée principal
│   ├── purger.py         ← filtrage RP/HRP
│   ├── llm.py            ← wrapper Ollama/Mistral
│   ├── query.py          ← moteur de requête
│   ├── interface.py      ← interface Gradio
│   └── steps/
│       ├── purge.py
│       ├── translate.py
│       ├── subdivide.py
│       ├── analyze_when.py
│       ├── analyze_where.py
│       ├── analyze_who.py
│       ├── analyze_which.py
│       ├── analyze_what.py
│       └── analyze_how.py
├── requirements.txt
└── DEPLOIEMENT.md
```

---

## Workflow pour de nouveaux exports

```bash
# 1. Déposer le(s) nouveau(x) fichier(s) dans data/exports/
# 2. Relancer la pipeline (saute ce qui est déjà traité)
.venv/bin/python src/pipeline.py
```

---

## Dépannage

**Ollama ne répond pas**
```bash
ollama list    # vérifie que le service tourne
ollama serve   # le lancer si besoin
```

**Modèle manquant**
```bash
ollama pull mistral
```

**Pipeline lente**
Normal sur CPU. Mistral 7B traite ~2-5 scènes/minute selon leur longueur.
L'étape 4 (analyze) est la plus longue — elle fait 5 appels LLM par scène.

**Reprendre après interruption**
La pipeline est idempotente. Relancer la même commande, elle reprend là où elle s'est arrêtée.

**Retraiter une scène**
```bash
# Supprimer le dossier d'analyse puis relancer
rm -rf data/analysis/nom_scene_000
.venv/bin/python src/pipeline.py --only-step 4 --scene nom_scene_000
```
