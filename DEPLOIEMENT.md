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

**Linux / macOS**
```bash
curl -fsSL https://ollama.com/install.sh | sh
```

**Windows**
Télécharger et exécuter l'installeur depuis : https://ollama.com/download

**Vérifier que ça tourne**
```bash
ollama --version
```

> Si vous obtenez `address already in use` au démarrage, Ollama tourne déjà en arrière-plan — c'est normal, passez à la suite.

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

Prérequis : Python 3.11+ installé.
- Linux/macOS : généralement déjà présent (`python3 --version`)
- Windows : télécharger depuis python.org — cocher **"Add to PATH"** à l'installation

**Linux / macOS**
```bash
cd /chemin/vers/RP_IA
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**Windows** (PowerShell)
```powershell
cd C:\chemin\vers\RP_IA
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

> **Windows — erreur de politique d'exécution** sur l'activation du venv :
> ```powershell
> Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
> ```
> Puis relancer `.venv\Scripts\activate`.

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

## Étape 3b — Purge RP/HRP (recommandé)

Les exports Discord contiennent souvent des conversations informelles mélangées aux scènes RP.
Cette étape filtre le contenu non-RP avant l'indexation.

**Signaux détectés automatiquement :**
- Séparateurs de scène : `---` / `___` / `***` → délimitent les blocs RP
- HRP heuristique : emojis excessifs, liens, langage informel
- HRP dans la scène : `((...))` déjà géré plus tard par le pipeline
- Long délai temporel entre messages → possible fin de scène

**Linux / macOS**
```bash
# Purge heuristique seule (rapide)
python3 src/purger.py data/exports/

# Avec Mistral pour les cas ambigus (plus précis, plus lent)
python3 src/purger.py data/exports/ --with-llm
```

**Windows**
```powershell
python src\purger.py data\exports\
python src\purger.py data\exports\ --with-llm
```

Les exports filtrés sont écrits dans `data/exports_filtered/`.
**Les étapes suivantes utilisent ce dossier filtré à la place de `data/exports/`.**

---

## Étape 4 — Phase 0 : bootstrap des personnages

**Linux / macOS**
```bash
python3 src/bootstrap.py data/exports/
```

**Windows**
```powershell
python src\bootstrap.py data\exports\
```

Cela génère `config/personnages_draft.yaml` et `config/lore_draft.yaml`.

Ouvrir ces fichiers, vérifier les détections, corriger si besoin :
- Fusionner les alias d'un même personnage
- Supprimer les faux positifs (noms détectés par erreur)
- Compléter les PNJ et les noms d'arcs

Puis copier vers les fichiers de config actifs :

**Linux / macOS**
```bash
cp config/personnages_draft.yaml config/personnages.yaml
```

**Windows**
```powershell
copy config\personnages_draft.yaml config\personnages.yaml
```

---

## Étape 5 — Indexation

**Linux / macOS**
```bash
# Indexation simple
python3 src/indexer.py data/exports/

# Avec extraction de lore et tags thématiques (plus lent, nécessite Mistral)
python3 src/indexer.py data/exports/ --with-lore --with-tags
```

**Windows**
```powershell
python src\indexer.py data\exports\
# ou
python src\indexer.py data\exports\ --with-lore --with-tags
```

Cette étape peut prendre quelques minutes selon le volume de messages.
Elle stocke les vecteurs dans `data/index/` (persistant sur disque).

> À relancer à chaque ajout de nouveaux exports. Les nouveaux documents
> s'ajoutent à la base existante sans écraser les anciens.
>
> En cas d'erreur à la première indexation, vider l'index et recommencer :
> - Linux/macOS : `rm -rf data/index/*`
> - Windows : `Remove-Item data\index\* -Recurse`

---

## Étape 6 — Lancer l'interface

**Linux / macOS**
```bash
python3 src/interface.py
```

**Windows**
```powershell
python src\interface.py
```

Ouvrir dans le navigateur : **http://localhost:7860**

Cliquer sur **"Initialiser le moteur"** puis poser vos questions.

---

## Workflow d'enrichissement (nouveaux exports)

**Linux / macOS**
```bash
# 1. Déposer le nouveau fichier JSON dans data/exports/
# 2. Re-bootstrap si nouveaux personnages
python3 src/bootstrap.py data/exports/
# 3. Corriger config/personnages.yaml si besoin
# 4. Ré-indexer (ajout non destructif)
python3 src/indexer.py data/exports/
# 5. Relancer l'interface
```

**Windows**
```powershell
python src\bootstrap.py data\exports\
python src\indexer.py data\exports\
```

---

## Différences Linux/macOS → Windows

| Linux/macOS | Windows (PowerShell) |
|---|---|
| `python3` | `python` |
| `source .venv/bin/activate` | `.venv\Scripts\activate` |
| `/` dans les chemins | `\` dans les chemins |
| `cp fichier dest` | `copy fichier dest` |
| `rm -rf dossier/*` | `Remove-Item dossier\* -Recurse` |

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
