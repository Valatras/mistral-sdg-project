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
