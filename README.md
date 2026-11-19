## Quickstart depuis github

### 1. Installer uv :
macOS and Linux:
```curl -LsSf https://astral.sh/uv/install.sh | sh```

Windows (PowerShell):
```powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"```

### 2. Initialiser le projet

```
git clone <Prendre l'url de mon projet>
cd mistral-sdg-project
uv sync
```

`uv sync` va :
- Lire le fichier pyproject.toml.
- Lire uv.lock pour connaître les versions exactes des dépendances.
- Trouver ou installer une version de Python compatible, si nécessaire et si les téléchargements Python sont autorisés.
- Créer automatiquement le dossier .venv/.
- Installer toutes les dépendances nécessaires.

### Les points d'accès sont visible sur le fichier pyproject.toml, section [scripts]
Si par exemple tu veux faire exécuter le fichier **main.py** du dossier src/bellai/ :
```uv run bellai-main```

ou simplement pour le fichier **__init__.py** :
```uv run bellai```
