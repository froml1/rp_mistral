# Projet IA Conversationnelle — Analyse de Roleplay Discord

> Documentation pédagogique issue de nos échanges techniques.
> Rédigée pour quelqu'un qui découvre l'IA et souhaite comprendre les choix faits.

---

## Sommaire

1. [C'est quoi l'objectif du projet ?](#1-cest-quoi-lobjectif-du-projet)
2. [Les grandes briques technologiques](#2-les-grandes-briques-technologiques)
3. [Comprendre le RAG — le cœur du projet](#3-comprendre-le-rag--le-cœur-du-projet)
4. [Les modèles de langage (LLM)](#4-les-modèles-de-langage-llm)
5. [L'architecture complète expliquée](#5-larchitecture-complète-expliquée)
6. [Pourquoi ces choix technologiques ?](#6-pourquoi-ces-choix-technologiques-)
7. [Ce que le projet ne fera pas (volontairement)](#7-ce-que-le-projet-ne-fera-pas-volontairement)
8. [Glossaire des termes techniques](#8-glossaire-des-termes-techniques)

---

## 1. C'est quoi l'objectif du projet ?

On veut construire une **IA conversationnelle** (un assistant à qui on peut poser des questions) capable de :

- **Lire des exports Discord** : Discord permet d'exporter l'historique d'un salon sous forme de fichier texte structuré (JSON ou CSV). Ces fichiers contiennent les messages, les auteurs, les dates, etc.
- **Comprendre le roleplay narratif écrit** : le RP Discord a ses propres codes — des balises comme `*action*`, `**emphase**`, des annotations hors-personnage comme `[OOC : ...]`, des noms de personnages distincts des pseudos joueurs, etc.
- **Répondre à des questions précises** sur ce contenu : "Qui a introduit le personnage X ?", "Résume l'arc de la session 3", "Quel est le ton dominant de ce chapitre ?"
- **S'enrichir au fil du temps** : quand de nouvelles sessions sont exportées, l'IA peut les intégrer sans qu'on doive tout recommencer.

L'IA ne génère **pas** de RP elle-même — elle **analyse** et **répond** à propos du RP existant.

---

## 2. Les grandes briques technologiques

Le projet est composé de plusieurs outils distincts qui travaillent ensemble. Voici chacun, expliqué simplement.

### DiscordChatExporter

- **C'est quoi ?** Un outil (programme en ligne de commande) qui se connecte à Discord et télécharge l'historique d'un salon sous forme de fichier lisible par un ordinateur.
- **Pourquoi ?** Discord ne fournit pas directement ses données dans un format pratique pour l'IA. Cet outil fait le pont.
- **Ce qu'il produit :** un fichier JSON (une liste structurée de messages avec auteur, date, contenu, réactions...).

### Python (script de nettoyage)

- **C'est quoi ?** Un langage de programmation très utilisé en science des données et en IA.
- **Pourquoi ici ?** Le texte brut d'un export Discord est "sale" pour une IA : il contient des emojis, des mentions `@pseudo`, des balises de mise en forme Markdown, des fautes d'orthographe intentionnelles ou non, des annotations RP. Un script Python va **normaliser** tout ça — le rendre plus compréhensible pour l'IA.
- **Exemple de nettoyage :** transformer `**@Joueur1** : *Il s'avance lentement...*` en quelque chose de structuré avec les métadonnées séparées du texte narratif.

### Ollama

- **C'est quoi ?** Un outil qui permet de faire tourner des modèles de langage (les IA) **en local sur ton ordinateur**, sans avoir besoin d'internet ni de compte chez OpenAI.
- **Pourquoi ?** Confidentialité des données RP, gratuité, et contrôle total.
- **Comment ça marche ?** Tu installes Ollama, tu lui dis quel modèle télécharger (ex : `ollama pull mistral`), et tu peux ensuite l'interroger.

### ChromaDB ou Qdrant (base vectorielle)

- **C'est quoi ?** Une base de données spéciale. Pas comme une base de données classique qui stocke du texte brut — celle-ci stocke des **vecteurs** (expliqué dans le glossaire).
- **Pourquoi ?** Pour que l'IA puisse retrouver rapidement les passages de texte les plus pertinents par rapport à une question posée, même dans des milliers de messages.
- **Analogie :** c'est comme un index de bibliothèque ultra-intelligent, capable de trouver "les passages qui parlent de trahison" même si tu n'as pas utilisé ce mot exact.

### LlamaIndex

- **C'est quoi ?** Un framework Python (une boîte à outils) qui orchestre tout : il prend ta question, va chercher les bons passages dans la base vectorielle, les envoie au modèle de langage, et te retourne une réponse.
- **Pourquoi pas LangChain ?** Les deux font des choses similaires. LlamaIndex est plus orienté "recherche dans des documents", ce qui correspond mieux à notre cas.

### Open WebUI ou Gradio (interface)

- **C'est quoi ?** L'interface graphique — la fenêtre dans laquelle tu vas taper tes questions et lire les réponses. Comme une interface de chat.
- **Open WebUI** : ressemble à ChatGPT, s'intègre directement avec Ollama. Clé en main.
- **Gradio** : plus simple, souvent utilisé pour des prototypes rapides. Idéal pour tester.

---

## 3. Comprendre le RAG — le cœur du projet

**RAG = Retrieval-Augmented Generation** (Génération Augmentée par Récupération)

C'est **la technique la plus importante** du projet. Elle permet à l'IA de répondre à des questions sur **tes** documents, même si elle ne les a jamais "appris" pendant son entraînement.

### Sans RAG

L'IA ne connaît que ce qu'elle a appris pendant son entraînement (des millions de textes publics). Elle ne sait rien de tes sessions de RP. Si tu lui poses une question dessus, elle invente ou dit qu'elle ne sait pas.

### Avec RAG

```
Tu poses une question
        ↓
L'IA cherche dans ta base de documents
(les exports Discord déjà indexés)
        ↓
Elle récupère les 3-5 passages les plus pertinents
        ↓
Elle combine ta question + ces passages
pour formuler une réponse précise
```

### Pourquoi c'est puissant pour ce projet

- Quand tu ajoutes un nouvel export Discord → il est découpé, indexé, ajouté à la base
- Immédiatement interrogeable, **sans réentraîner l'IA**
- L'IA peut citer ses sources (quel message, quel auteur, quelle date)

### Le chunking (découpage)

Avant d'indexer un document, on le découpe en **chunks** (morceaux). C'est important car l'IA ne peut pas lire 50 000 messages d'un coup — elle travaille sur des extraits.

**Bonne pratique pour le RP :** découper par scène ou par session, pas au hasard tous les 500 mots. Sinon on coupe un arc narratif en deux et la réponse perd son sens.

---

## 4. Les modèles de langage (LLM)

**LLM = Large Language Model** (Grand Modèle de Langage)

C'est l'"intelligence" de l'IA — le composant qui lit du texte et génère des réponses cohérentes.

### Comment ils fonctionnent (très simplement)

Ces modèles ont été entraînés sur des quantités massives de textes (Wikipedia, livres, forums...). Ils ont appris à **prédire le mot suivant** de manière si sophistiquée qu'ils peuvent raisonner, résumer, expliquer, traduire.

### Ceux qu'on a choisis

| Modèle | Taille | Pour quel usage |
|---|---|---|
| **Mistral 7B** | Léger (~4 Go) | Démarrer, tester, machine modeste |
| **Phi-3 Mini** | Très léger (~2 Go) | Ordinateur peu puissant |
| **LLaMA 3.1 8B** | Léger (~5 Go) | Meilleure compréhension narrative |
| **Mistral Nemo 12B** | Moyen (~7 Go) | Meilleure qualité, machine correcte |

### Le "quantisé" (Q4, Q5...)

Les modèles originaux sont énormes. La **quantisation** est une technique de compression qui réduit leur taille avec peu de perte de qualité. Un modèle "Q4" utilise 4 bits par paramètre au lieu de 16 — il prend 4x moins de place et tourne sans GPU dédié.

### Pas besoin de GPU puissant

Avec un modèle 7B quantisé, une machine avec **16 Go de RAM** et un bon processeur suffit. L'IA tourne en local, sans carte graphique dédiée nécessaire.

---

## 5. L'architecture complète expliquée

Voici ce qui se passe de bout en bout, dans l'ordre :

```
[1] Export Discord (fichier JSON)
         ↓
[2] Script Python de nettoyage
    → Supprime les balises inutiles
    → Structure les métadonnées (auteur, date, canal, personnage)
    → Identifie les balises RP (OOC, actions, dialogues)
         ↓
[3] Chunking (découpage en scènes/sessions)
         ↓
[4] Embedding (transformation en vecteurs)
    → Chaque chunk devient un "vecteur" numérique
    → Modèle utilisé : nomic-embed-text (open-source, local)
         ↓
[5] Stockage dans ChromaDB / Qdrant
    → La base vectorielle garde les vecteurs + le texte original
         ↓
[6] L'utilisateur pose une question via l'interface
         ↓
[7] LlamaIndex orchestre :
    → Transforme la question en vecteur
    → Cherche les chunks les plus proches dans la base
    → Envoie question + chunks au LLM
         ↓
[8] Ollama + Mistral/LLaMA génère la réponse
         ↓
[9] Réponse affichée dans Open WebUI / Gradio
```

### Schéma simplifié

```
Exports Discord → Nettoyage → Base vectorielle
                                     ↕
              Question utilisateur → LLM → Réponse
```

---

## 6. Pourquoi ces choix technologiques ?

### Tout open-source

- **Gratuit** : pas d'abonnement, pas de coût par requête
- **Privé** : tes données RP ne quittent jamais ta machine
- **Modifiable** : tu peux adapter chaque brique à tes besoins

### RAG plutôt que fine-tuning

Le **fine-tuning** consiste à réentraîner un modèle sur tes données. C'est coûteux (GPU, temps, expertise), et chaque nouvel export nécessiterait un nouveau cycle d'entraînement.

Le **RAG** est plus souple : tu ajoutes des documents à la base vectorielle, l'IA les exploite immédiatement. Parfait pour un corpus vivant comme du RP actif.

### Local plutôt que cloud

- Pas de dépendance à OpenAI, Anthropic, Google
- Pas de frais variables selon l'usage
- Confidentialité totale du contenu RP (qui peut être sensible ou simplement privé)

---

## 7. Ce que le projet ne fera pas (volontairement)

| Ce qu'on ne fait pas | Pourquoi c'est un bon choix |
|---|---|
| Fine-tuning du modèle | Trop coûteux, le RAG suffit |
| Hébergement cloud | Inutile, et moins privé |
| Génération de RP | Hors scope — on analyse, on ne crée pas |
| Correction automatique des fautes | On les tolère, elles font partie du style |

---

## 8. Glossaire des termes techniques

**API** : interface permettant à deux programmes de communiquer entre eux.

**Chunk / Chunking** : morceau de texte découpé depuis un document plus grand. Le chunking est l'opération de découpage.

**ChromaDB / Qdrant** : bases de données vectorielles open-source. Stockent des vecteurs pour permettre une recherche par similarité sémantique.

**Embedding** : transformation d'un texte en un vecteur numérique. Des textes au sens proche produisent des vecteurs proches.

**FastAPI** : framework Python pour créer des APIs web rapidement.

**Fine-tuning** : réentraînement partiel d'un modèle IA sur un jeu de données spécifique. Coûteux mais produit un modèle très spécialisé.

**Gradio** : outil Python pour créer des interfaces web simples autour de modèles IA.

**JSON** : format de fichier structuré lisible par les machines (et les humains). Ressemble à `{"clé": "valeur"}`. Utilisé par les exports Discord.

**LangChain** : framework Python pour construire des applications autour de LLMs. Concurrent de LlamaIndex.

**LlamaIndex** : framework Python spécialisé dans la construction de pipelines RAG sur des documents.

**LLM (Large Language Model)** : modèle de langage entraîné sur de grandes quantités de texte. Exemples : Mistral, LLaMA, GPT.

**Mistral / LLaMA / Phi** : noms de familles de modèles LLM open-source. Mistral vient d'une startup française, LLaMA de Meta, Phi de Microsoft.

**nomic-embed-text** : modèle open-source de création d'embeddings (vecteurs). Léger et performant, tourne en local.

**OOC (Out Of Character)** : terme RP désignant un message hors-personnage, méta-communication entre joueurs.

**Ollama** : outil permettant de télécharger et faire tourner des LLMs en local, simplement.

**Open WebUI** : interface graphique web pour Ollama. Ressemble à ChatGPT, hébergée localement.

**Pipeline** : chaîne d'opérations automatisées, où la sortie d'une étape est l'entrée de la suivante.

**Prompt / Prompt système** : texte d'instruction donné à l'IA pour cadrer son comportement. Le prompt système définit son rôle ("Tu es un assistant spécialisé dans l'analyse de RP...").

**Quantisation** : technique de compression d'un modèle IA. Réduit l'usage mémoire avec peu de perte de qualité. Les formats Q4, Q5 indiquent le niveau de compression.

**RAG (Retrieval-Augmented Generation)** : technique combinant recherche dans des documents et génération de texte. Permet à un LLM de répondre sur des documents qu'il n'a pas appris.

**RAM** : mémoire vive de l'ordinateur. Les LLMs locaux y résident entièrement pendant leur exécution.

**Vecteur** : liste de nombres représentant mathématiquement le "sens" d'un texte. Deux textes proches sémantiquement ont des vecteurs proches dans l'espace mathématique.

---

*Document évolutif — mis à jour à chaque échange technique.*
*Dernière mise à jour : avril 2026*
