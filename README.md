# Bellai

Bellai explore un corpus local de droit du travail belge. Ses réponses sont informatives et ne constituent pas des avis juridiques.

## Installation

Prérequis : Python 3.14 ou supérieur (géré par uv), [uv](https://docs.astral.sh/uv/) et Poppler pour les PDF. Sur Debian ou Ubuntu, installer Poppler avec `sudo apt install poppler-utils`. Sur macOS, utiliser `brew install poppler`. Vérifier que `pdftotext` est disponible dans le PATH.

Depuis la racine du dépôt :

```sh
uv sync --locked
cp -n .env.example .env
```

Renseigner `MISTRAL_API_KEY` dans `.env` ou dans l’environnement. Les modèles par défaut sont `mistral-embed` et `mistral-small-latest`. Les variables `MISTRAL_EMBED_MODEL` et `MISTRAL_CHAT_MODEL` permettent de les configurer. À l’indexation, Mistral reçoit le texte des passages. À chaque question, le service reçoit la question et les passages sélectionnés. Utiliser un corpus et des questions dont la transmission est autorisée.

## Utilisation

```sh
uv run --locked bellai ingest
uv run --locked bellai ask "Que disent les documents sur la durée du préavis en Belgique ?"
uv run --locked bellai ask "Que disent les documents sur la réforme du chômage ?" --top-k 3 --debug
uv run --locked bellai ingest --rebuild
uv run --locked python -m unittest discover -s tests -v
```

`ingest` parcourt récursivement `fichier_rag/` et accepte les PDF, les fichiers `.txt` et les fichiers `.md` en UTF-8. `--documents` permet de choisir un autre dossier. `--chunk-size` et `--chunk-overlap` règlent les longueurs en caractères (1200 et 150 par défaut). La commande signale les fichiers non pris en charge, les échecs d’extraction et les pages vides. Elle renvoie le code 1 si elle signale des problèmes, même si elle a pu indexer les passages utilisables. En cas d’échec des embeddings, l’index précédent reste disponible.

Relancer `ingest` après avoir changé le corpus ou les paramètres de découpage. Si les passages et le modèle d’embedding sont identiques, l’index est réutilisé sans appel API. `--rebuild` force le recalcul. L’index contient les passages et leurs vecteurs dans `.bellai/index.json`, ignoré par Git. Les deux commandes acceptent `--index CHEMIN` pour utiliser un autre fichier. Le CLI est aussi disponible via `uv run --locked python -m bellai`.

Le mode `--debug` affiche les passages récupérés, le score cosinus et l’identifiant de passage sur stderr. La réponse affiche les références effectivement citées, par exemple `[S1]`, avec le chemin du fichier et le numéro de page PDF (à partir de 1).

## Pipeline et limites

`src/bellai/RAG/documents.py` extrait le texte page par page et découpe les longs passages aux limites de paragraphes, de phrases, de lignes ou de mots lorsque possible. Il tient compte des titres d’articles lorsqu’ils sont reconnaissables. `src/bellai/RAG/index.py` stocke les vecteurs normalisés et effectue une recherche exhaustive par similarité cosinus. `src/bellai/RAG/answer.py` fournit les passages au modèle, demande une réponse sourcée ou un constat d’insuffisance et vérifie que les identifiants cités figurent dans les résultats de recherche.

Le notebook `Notebooks/RAGs/RAGs.ipynb` fournit les principes de provenance par page, de découpage avec recouvrement, de mise en cache JSON, de recherche dense et de réponse structurée avec abstention. Le SDK Mistral déjà utilisé dans le notebook `Notebooks/LLM_Fundamentals/jup.ipynb` évite une dépendance supplémentaire à LangChain. La recherche hybride, le suivi des coûts, l’évaluation hit@k/MRR et la recherche agentique restent hors de ce MVP.

L’extraction ne fait pas d’OCR et peut mal restituer les tableaux ou les mises en page complexes. Le corpus local peut être incomplet ou obsolète. La présence d’une citation permet de remonter au passage, mais la vérification automatique des identifiants ne démontre ni que le passage soutient l’affirmation ni que la règle est juridiquement applicable.

## Intake agent

L’intake agent est le premier agent de la boucle multi-agents. Il lit le récit de l’utilisateur, souvent émotionnel ou désordonné, et en extrait un dossier structuré (`CaseFile`) : thème, statut, catégorie ouvrier ou employé, type de contrat, commission paritaire, régions, dates, salaire, âge, protections particulières, questions juridiques reformulées. Il pose des questions de clarification quand des informations indispensables manquent, puis prépare les requêtes et les filtres de métadonnées pour la recherche dans le RAG. L’agent généraliste qui suit reçoit ce dossier.

Il est orchestré avec LangGraph (`langgraph`). Deux fournisseurs de LLM sont disponibles pour l’extraction, tous deux en sortie structurée validée par le schéma Pydantic :

| Fournisseur | Modèle | Bibliothèque | Clé dans `.env` |
|---|---|---|---|
| `mistral` (défaut) | `mistral-small-latest` | `langchain-mistralai` | `MISTRAL_API_KEY` |
| `claude` | `claude-haiku-5-5` | `anthropic` (`messages.parse`) | `ANTHROPIC_API_KEY` |

Le fournisseur se choisit avec l’option `--provider`, la variable `INTAKE_PROVIDER=claude` ou, en code, `extracteur_pour("claude")`.

### Essayer l’agent

```sh
uv run python -m bellai.intake_agent                    # Mistral
uv run python -m bellai.intake_agent --provider claude  # Claude
```

Saisir le récit, puis valider une ligne vide. Répondre aux éventuelles questions de clarification en une ligne. Le terminal affiche ensuite le `CaseFile`, l’ancienneté calculée et le contexte RAG. Les clés sont lues depuis `.env`. Le récit et les réponses sont envoyés au fournisseur choisi (Mistral ou Anthropic).

Depuis du code :

```python
from langgraph.types import Command
from bellai.intake_agent import build_graph
from bellai.intake_agent.graph import extracteur_pour

app = build_graph(extracteur=extracteur_pour("claude"))  # ou build_graph() : INTAKE_PROVIDER, sinon Mistral
config = {"configurable": {"thread_id": "session-1"}}
out = app.invoke({"messages": [("user", "On m'a licencié hier après 6 ans...")]}, config)

if "__interrupt__" in out:                 # informations manquantes
    print(out["__interrupt__"][0].value)   # questions à poser
    out = app.invoke(Command(resume="Je gagne 3000 € brut"), config)

out["case_file"]     # CaseFile (Pydantic)
out["contexte_rag"]  # requêtes et filtres pour la recherche
```

`build_graph(extracteur=...)` accepte n’importe quelle fonction `messages -> CaseFile`. On peut ainsi tester le graphe sans appel API.

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
| `preparation_rag` | Traduit le dossier en requêtes et en filtres de métadonnées. | non |

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

Les valeurs de `themes` et `niveaux` doivent correspondre aux métadonnées posées à l’ingestion du corpus :

- niveaux : `ue`, `federal`, `wallonie`, `bruxelles`, `flandre`, `communaute_germanophone` ;
- thèmes : `embauche`, `licenciement`, `chomage`, `discrimination`.

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
- Les modèles sont fixés dans `graph.py` (`default_extracteur`, `claude_extracteur`) et ne lisent pas encore `MISTRAL_CHAT_MODEL`.
- L’extracteur Claude n’a été vérifié que contre une API simulée (requête, schéma, lecture de la réponse), pas encore sur de vrais récits.
