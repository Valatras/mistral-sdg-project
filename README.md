# Bellai

Bellai explore un corpus local de droit du travail belge. Ses réponses sont informatives et ne constituent pas des avis juridiques.

## Installation

Prérequis : Python 3.14 ou supérieur (géré par uv), [uv](https://docs.astral.sh/uv/) et Poppler pour les PDF. Sur Debian ou Ubuntu, installer Poppler avec `sudo apt install poppler-utils`. Sur macOS, utiliser `brew install poppler`. Vérifier que `pdftotext` est disponible dans le PATH.

Depuis la racine du dépôt :

```sh
uv sync --locked
cp -n .env.example .env
```

Renseigner `MISTRAL_API_KEY` dans `.env` ou dans l’environnement. Les modèles sont définis dans `src/bellai/config.py` par rôle. `EMBEDDING_MODEL` sert à l’indexation et à la recherche, `RAG_ANSWER_MODEL` à la commande `ask`, `INTAKE_MODEL` à l’extraction du dossier et `LEGAL_RESEARCH_MODEL` à la synthèse des résultats de recherche. Après un changement de `EMBEDDING_MODEL`, relancer `bellai ingest` pour réindexer le corpus. À l’indexation, Mistral reçoit le texte des passages. À chaque question, le service reçoit la question et les passages sélectionnés. Utiliser un corpus et des questions dont la transmission est autorisée.

## Utilisation

```sh
uv run --locked bellai ingest
uv run --locked bellai ask "Que disent les documents sur la durée du préavis en Belgique ?"
uv run --locked bellai ask "Que disent les documents sur la réforme du chômage ?" --top-k 3 --debug
uv run --locked bellai ingest --rebuild
uv run --locked python -m unittest discover -s tests -v
```

`ingest` parcourt récursivement `fichier_rag/` et accepte les PDF, les fichiers `.txt` et les fichiers `.md` en UTF-8. `--documents` permet de choisir un autre dossier. `--chunk-size` et `--chunk-overlap` règlent les longueurs en caractères (1200 et 150 par défaut). La commande signale les fichiers non pris en charge, les échecs d’extraction et les pages vides. Elle renvoie le code 1 si elle signale des problèmes, même si elle a pu indexer les passages utilisables. En cas d’échec des embeddings, l’index précédent reste disponible.

Relancer `ingest` après avoir changé le corpus ou les paramètres de découpage. Chroma conserve l’index local dans `.bellai/chroma/`, ignoré par Git. Les commandes acceptent `--index CHEMIN` pour choisir un autre **dossier**. Si les passages et le modèle d’embedding sont identiques, la collection active est réutilisée sans appel API. `--rebuild` force le recalcul. Chaque reconstruction publie une nouvelle collection après son remplissage complet et conserve les collections précédentes sur disque.

Lors du premier `bellai ingest` dans le dossier par défaut, les embeddings compatibles de l’ancien fichier `.bellai/index.json` sont importés automatiquement. Seuls les passages dont l’identifiant ou le modèle a changé nécessitent un appel d’embedding. Le fichier JSON d’origine reste disponible et les recherches utilisent ensuite Chroma. Le CLI est aussi disponible via `uv run --locked python -m bellai`.

Le mode `--debug` affiche les passages récupérés, le score cosinus et l’identifiant de passage sur stderr. La réponse affiche les références effectivement citées, par exemple `[S1]`, avec le chemin du fichier et le numéro de page PDF (à partir de 1).

## Pipeline et limites

`src/bellai/RAG/documents.py` extrait le texte page par page et découpe les longs passages aux limites de paragraphes, de phrases, de lignes ou de mots lorsque possible. Il tient compte des titres d’articles lorsqu’ils sont reconnaissables. `src/bellai/RAG/index.py` stocke le texte, les vecteurs et la provenance dans Chroma et effectue une recherche approximative HNSW par distance cosinus. `src/bellai/RAG/answer.py` fournit les passages au modèle, demande une réponse sourcée ou un constat d’insuffisance et vérifie que les identifiants cités figurent dans les résultats de recherche.

Le notebook `Notebooks/RAGs/RAGs.ipynb` fournit les principes de provenance par page, de découpage avec recouvrement, de réutilisation des embeddings, de recherche dense et de réponse structurée avec abstention. Le SDK Mistral déjà utilisé dans le notebook `Notebooks/LLM_Fundamentals/jup.ipynb` évite une dépendance supplémentaire à LangChain. La recherche hybride, le suivi des coûts, l’évaluation hit@k/MRR restent hors du périmètre actuel.

L’extraction ne fait pas d’OCR et peut mal restituer les tableaux ou les mises en page complexes. Le corpus local peut être incomplet ou obsolète. La présence d’une citation permet de remonter au passage, mais la vérification automatique des identifiants ne démontre ni que le passage soutient l’affirmation ni que la règle est juridiquement applicable.

## Intake agent

L’intake agent est le point d’entrée du traitement d’un dossier. Il lit le récit de l’utilisateur, souvent émotionnel ou désordonné, et en extrait un dossier structuré (`CaseFile`) : thème, statut, catégorie ouvrier ou employé, type de contrat, commission paritaire, régions, dates, salaire, âge, protections particulières, questions juridiques reformulées. Il pose des questions de clarification quand des informations indispensables manquent, puis prépare les requêtes, la langue et les indications de thème et de région pour la recherche.

L’intake utilise LangGraph et prend en charge deux fournisseurs pour l’extraction structurée.

| Fournisseur | Modèle configuré dans `src/bellai/config.py` | Clé API |
|---|---|---|
| `mistral` (par défaut) | `INTAKE_MODEL` | `MISTRAL_API_KEY` |
| `claude` | `CLAUDE_INTAKE_MODEL` | `ANTHROPIC_API_KEY` |

Choisir le fournisseur avec `--provider` ou `INTAKE_PROVIDER=claude`. En Python, passer `extracteur_pour("claude")` (depuis `bellai.agents.intake_agent.graph`) à `build_graph(extracteur=...)`.

### Essayer l’agent

```sh
uv run --locked python -m bellai.agents.intake_agent
uv run --locked python -m bellai.agents.intake_agent --provider claude
```

Saisir le récit, puis valider une ligne vide. Répondre aux éventuelles questions de clarification en une ligne. Le terminal affiche ensuite le `CaseFile`, l’ancienneté calculée et le contexte RAG. Les clés peuvent être définies dans `.env`. Le récit et les réponses aux clarifications sont envoyés au fournisseur choisi.

Depuis du code :

```python
from langgraph.types import Command
from bellai.agents.intake_agent import build_graph

app = build_graph()
config = {"configurable": {"thread_id": "session-1"}}
out = app.invoke({"messages": [("user", "On m'a licencié hier après 6 ans...")]}, config)

if "__interrupt__" in out:                 # informations manquantes
    print(out["__interrupt__"][0].value)   # questions à poser
    out = app.invoke(Command(resume="Je gagne 3000 € brut"), config)

out["case_file"]     # CaseFile (Pydantic)
out["contexte_rag"]  # requêtes, langue et indications de thème et de région
```

`build_graph(extracteur=...)` accepte n’importe quelle fonction `messages -> CaseFile`. On peut ainsi tester le graphe sans appel API.

### Interface web

```sh
uv run --locked python -m bellai.web          # ouvre http://127.0.0.1:8000
uv run --locked python -m bellai.web --port 8080 --no-browser
```

La page permet de choisir le fournisseur, de charger un récit de `evals/`, de répondre aux clarifications, puis d’afficher le dossier, l’ancienneté et le contexte RAG et de lancer la recherche juridique avec ses sources. Le serveur n’écoute que sur la machine locale. Les sessions sont perdues à son arrêt.

### Fonctionnement

```
START → intake → validation ─┬─ champs manquants ─→ clarification → intake
                             └─ dossier complet ──→ preparation_rag → END
```

| Nœud | Rôle | LLM |
|---|---|---|
| `intake` | Extrait le `CaseFile` du récit et des réponses de clarification. | oui |
| `validation` | Normalise, calcule l’ancienneté, liste les champs manquants et rédige les questions. | non |
| `clarification` | Met le graphe en pause avec `interrupt()` et attend la réponse de l’utilisateur. Au maximum 2 tours (`MAX_TOURS_CLARIFICATION`). | non |
| `preparation_rag` | Traduit le dossier en requêtes et en indications de thème et de région. | non |

Le LLM ne fait que l’extraction des faits. Les calculs et les vérifications sont faits en Python :

- la région est déduite du code postal (communes germanophones 4700–4799 comprises) et prime sur celle proposée par le LLM ;
- le numéro de commission paritaire est nettoyé (`"CP 200"` devient `"200"`) ;
- une date de début de contrat dans le futur et un âge improbable sont rejetés ;
- l’ancienneté est calculée en mois entre la date de début et la notification du licenciement.

Champs indispensables par thème (`REQUIS` dans `normalisation.py`) :

| Thème | Champs requis |
|---|---|
| `licenciement` | date de début, date de notification, catégorie, salaire brut mensuel |
| `chomage` | région du domicile, âge, date de fin de contrat |
| `embauche`, `discrimination` | région du lieu de travail |

### Contexte RAG

`preparation_rag` produit par exemple :

```python
{"requetes": ["Le licenciement est-il valable sans explication ?"],
 "themes": ["licenciement"],
 "niveaux": ["ue", "federal", "wallonie"],
 "langue": "fr"}
```

Les niveaux `ue` et `federal` sont toujours inclus. La région ajoutée dépend du thème : la région du domicile pour le chômage, puisque le service compétent (Forem, Actiris, VDAB ou ADG) en dépend, et la région du lieu de travail pour l’embauche, la discrimination et le licenciement. Si la région est inconnue, toutes les régions sont conservées pour ne pas exclure le bon texte.

Le corpus contient le chemin source et, pour les PDF, le numéro de page. Les indications `themes` et `niveaux` sont disponibles dans le contexte et la recherche porte sur l’ensemble du corpus.

### Fichiers

| Fichier | Contenu |
|---|---|
| `schema.py` | `CaseFile`, le contrat partagé avec les agents suivants |
| `prompts.py` | Prompt d’extraction, avec la date du jour injectée |
| `normalisation.py` | Normalisation, champs requis, questions de clarification (FR et NL), ancienneté |
| `rag_context.py` | Construction du contexte RAG |
| `graph.py` | État, nœuds, extracteurs Mistral et Claude, `extracteur_pour()` et `build_graph()` |
| `__main__.py` | Essai interactif dans le terminal |

### Évaluation

`evals/intake_user_stories.json` contient 17 récits de test : licenciement, chômage, embauche et cas limites (plusieurs thèmes, infos contradictoires, hors sujet, tentative d’injection). Ils sont rédigés en français, néerlandais, anglais et allemand. Pour chaque récit, le fichier indique les valeurs attendues (`attendu`), les champs qui doivent rester vides (`doit_rester_null`, pour détecter les valeurs inventées) et les pièges visés.

### Limites connues

- Le LLM peut encore déduire une valeur non mentionnée, par exemple un `CDI` supposé.
- Les questions de clarification n’existent qu’en français et en néerlandais. Pour l’anglais et l’allemand, elles s’affichent en français.
- La région n’est déduite que d’un code postal. Une ville citée sans code postal repose sur l’interprétation du LLM.
- Le thème `autre` ne déclenche aucune clarification.
- Les sessions sont gardées en mémoire (`MemorySaver`) et perdues à l’arrêt du processus.


## Legal research agent

Le code RAG se trouve dans `src/bellai/agents/legal_research_agent/`. Le module `documents.py` extrait et découpe les documents, `index.py` gère l’index Chroma et la recherche de passages, et `answer.py` génère les réponses et leurs citations. Ces modules sont utilisés par les commandes `ingest` et `ask` ainsi que par le parcours de recherche après l’intake.

Après l’ingestion, lancer le parcours interactif complet :

```sh
uv run --locked python -m bellai.agents.intake_agent --research
uv run --locked python -m bellai.agents.intake_agent --provider claude --research
```

La recherche utilise Mistral pour les embeddings et la synthèse, y compris avec Claude pour l’intake. Le parcours Claude avec `--research` requiert donc les deux clés.

Après les clarifications, l’agent récupère dans Chroma jusqu’à cinq passages par requête distincte préparée par l’intake, puis génère un constat dans la langue du dossier. Chaque constat cite les extraits utilisés avec leur chemin et leur page, ou indique que les sources sont insuffisantes. Les identifiants `[S1]`, `[S2]`, etc. sont propres à chaque réponse. Les éléments manquants du dossier restent visibles. `--index` sélectionne un dossier Chroma et `--top-k` règle le nombre de passages par requête.

Depuis du code, `bellai.agents.legal_research_agent.research(out["contexte_rag"], ...)` accepte les fonctions d’embedding et de génération en paramètres. Il renvoie des résultats structurés avec la question, le constat et la provenance de chaque citation.

## Traçage (LangSmith)

Le traçage est désactivé par défaut. Pour l’activer, créer une clé API sur [LangSmith](https://smith.langchain.com) (ou [la région EU](https://eu.smith.langchain.com)), puis renseigner dans `.env` :

```sh
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=lsv2_...
LANGSMITH_PROJECT=bellai
LANGSMITH_ENDPOINT=https://eu.api.smith.langchain.com   # seulement pour un compte EU
```

Les commandes habituelles envoient ensuite leurs traces dans le projet choisi. Sont tracés : les nœuds du graphe LangGraph et l’état entre les nœuds, l’extraction Mistral (`ChatMistralAI`), l’extraction Claude (`claude_intake`) ainsi que les embeddings (`mistral_embed`) et les générations (`mistral_chat`) du RAG et de la recherche juridique.

Les récits, les réponses de clarification et les passages du corpus sont transmis à LangSmith. Activer le traçage de préférence avec les récits de `evals/`. Avec des données réelles, ajouter `LANGSMITH_HIDE_INPUTS=true` et `LANGSMITH_HIDE_OUTPUTS=true` pour ne conserver que la structure et les durées.
