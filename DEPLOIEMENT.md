# Notes de déploiement — IA Roleplay Discord

## Prérequis machine

| Composant | Minimum | Recommandé |
|---|---|---|
| RAM | 8 Go | 16 Go |
| Stockage | 10 Go libres | 20 Go |
| OS | Linux / macOS / Windows | Linux ou macOS |
| Python | 3.10+ | 3.11+ |
| GPU | Non requis | Non requis (CPU suffit) |

---

## Étape 1 — Installer Ollama

Ollama fait tourner Mistral en local.

```bash
# Linux / macOS
curl -fsSL https://ollama.com/install.sh | sh

# Vérifier que ça tourne
ollama --version
```

Puis télécharger les deux modèles nécessaires :

```bash
# Le LLM principal (génération de réponses) — ~4 Go
ollama pull mistral

# Le modèle d'embedding (transformation en vecteurs) — ~300 Mo
ollama pull nomic-embed-text
```

> Ces téléchargements se font une seule fois. Ensuite tout tourne hors-ligne.

---

## Étape 2 — Installer les dépendances Python

```bash
# Depuis le dossier racine du projet
cd /chemin/vers/RP_IA

# Créer un environnement virtuel (recommandé)
python3 -m venv .venv
source .venv/bin/activate        # Linux/macOS
# ou .venv\Scripts\activate      # Windows

# Installer les librairies
pip install -r requirements.txt
```

---

## Étape 3 — Préparer les exports Discord

Utiliser [DiscordChatExporter](https://github.com/Tyrrrz/DiscordChatExporter) pour exporter vos salons.

- Format d'export : **JSON**
- Placer les fichiers `.json` dans : `data/exports/`

Structure attendue :
```
data/
  exports/
    arc-1-prologue.json
    arc-2-la-traversee.json
    arc-3-la-forteresse.json
```

---

## Étape 4 — Phase 0 : bootstrap des personnages

```bash
python src/phase0_bootstrap.py data/exports/
```

Cela génère `config/personnages_draft.yaml`.

Ouvrir ce fichier, vérifier les détections, corriger si besoin :
- Fusionner les alias d'un même personnage
- Supprimer les faux positifs (noms détectés par erreur)
- Compléter les PNJ et les noms d'arcs

Puis renommer/copier :
```bash
cp config/personnages_draft.yaml config/personnages.yaml
# Editer config/personnages.yaml avec votre éditeur
```

---

## Étape 5 — Indexation

```bash
python src/indexer.py data/exports/
```

Cette étape peut prendre quelques minutes selon le volume de messages.
Elle stocke les vecteurs dans `data/index/` (persistant sur disque).

> À relancer à chaque ajout de nouveaux exports. Les nouveaux documents
> s'ajoutent à la base existante sans écraser les anciens.

---

## Étape 6 — Lancer l'interface

```bash
# S'assurer qu'Ollama tourne en arrière-plan
ollama serve &

# Lancer l'interface web
python src/interface.py
```

Ouvrir dans le navigateur : **http://localhost:7860**

Cliquer sur **"Initialiser le moteur"** puis poser vos questions.

---

## Workflow d'enrichissement (nouveaux exports)

```
1. Exporter le nouveau salon Discord → data/exports/nouveau-canal.json
2. python src/phase0_bootstrap.py data/exports/   # re-détecter si nouveaux persos
3. Corriger config/personnages.yaml si besoin
4. python src/indexer.py data/exports/            # ré-indexer (ajout non destructif)
5. Relancer l'interface
```

---

## Structure du projet

```
RP_IA/
├── config/
│   ├── personnages.yaml          # Config joueurs/personnages (à éditer)
│   └── personnages_draft.yaml    # Brouillon généré par phase0
├── data/
│   ├── exports/                  # Fichiers JSON Discord (à remplir)
│   └── index/                    # Base vectorielle ChromaDB (auto-généré)
├── src/
│   ├── phase0_bootstrap.py       # Extraction automatique des personnages
│   ├── preprocessing.py          # Nettoyage et structuration des messages
│   ├── indexer.py                # Chunking + embedding + stockage
│   ├── rag_pipeline.py           # Pipeline RAG (recherche + Mistral)
│   └── interface.py              # Interface Gradio
├── requirements.txt
├── DEPLOIEMENT.md                # Ce fichier
└── projet_IA_RP_documentation.md # Documentation pédagogique
```

---

## Dépannage courant

**Ollama ne répond pas**
```bash
# Vérifier que le service tourne
ollama list
# Si non, le lancer
ollama serve
```

**Erreur "model not found"**
```bash
ollama pull mistral
ollama pull nomic-embed-text
```

**L'indexation est lente**
Normal sur CPU. Un corpus de 5000 messages prend ~5-10 minutes.
Ne pas interrompre, laisser tourner.

**Réponses imprécises**
- Vérifier que `config/personnages.yaml` est bien rempli
- Augmenter `top_k` dans `rag_pipeline.py` (ligne `build_query_engine`) pour consulter plus de scènes
- Reformuler la question avec le nom exact du personnage ou de l'arc

---

## Évolutions possibles (plus tard)

| Évolution | Complexité | Bénéfice |
|---|---|---|
| Passer à LLaMA 3.1 8B | Faible (`ollama pull llama3.1`) | Meilleure narration |
| Filtres par arc/personnage | Moyenne | Questions plus précises |
| Résumés automatiques par session | Moyenne | Vue d'ensemble rapide |
| Fine-tuning sur le style RP | Élevée | Réponses très spécialisées |
