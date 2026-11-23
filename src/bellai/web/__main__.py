"""Interface web locale pour essayer l'intake agent et la recherche juridique.

    uv run python -m bellai.web            # puis ouvrir http://127.0.0.1:8000
    uv run python -m bellai.web --port 8080

Serveur de la bibliothèque standard, accessible seulement depuis cette machine.
Les récits sont envoyés au fournisseur choisi (Mistral ou Claude), comme avec le terminal.
"""

import argparse
import json
import threading
import uuid
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from dotenv import load_dotenv
from langgraph.types import Command

from ..agents.legal_research_agent.answer import format_source
from ..agents.legal_research_agent.index import DEFAULT_INDEX
from ..agents.intake_agent import build_graph
from ..agents.intake_agent.graph import FOURNISSEURS, extracteur_pour
from ..agents.legal_research_agent import research
from ..config import LEGAL_RESEARCH_MODEL, Models

ICI = Path(__file__).parent
STORIES = Path(__file__).resolve().parents[3] / "evals" / "intake_user_stories.json"

_graphes: dict[str, object] = {}       # un graphe compilé par fournisseur
_sessions: dict[str, str] = {}         # thread_id -> fournisseur
_verrou = threading.Lock()


def _graphe(fournisseur: str):
    with _verrou:
        if fournisseur not in _graphes:
            _graphes[fournisseur] = build_graph(extracteur=extracteur_pour(fournisseur))
        return _graphes[fournisseur]


def _etat(out: dict) -> dict:
    """Réponse JSON : questions de clarification en attente, ou dossier final."""
    if "__interrupt__" in out:
        return {"status": "questions", "questions": out["__interrupt__"][0].value}
    return {
        "status": "done",
        "case_file": out["case_file"].model_dump(mode="json", exclude_defaults=True),
        "anciennete_mois": out.get("anciennete_mois"),
        "manquants": out.get("manquants", []),
        "contexte_rag": out["contexte_rag"],
    }


def demarrer(body: dict) -> dict:
    recit = (body.get("recit") or "").strip()
    if not recit:
        raise ValueError("Le récit est vide.")
    fournisseur = body.get("provider") or "mistral"
    if fournisseur not in FOURNISSEURS:
        raise ValueError(f"Fournisseur inconnu : {fournisseur}")
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}, "tags": ["web", fournisseur]}
    out = _graphe(fournisseur).invoke({"messages": [("user", recit)]}, config)
    _sessions[thread_id] = fournisseur
    return {"thread_id": thread_id, "provider": fournisseur, **_etat(out)}


def _session(body: dict) -> tuple[object, dict]:
    thread_id = body.get("thread_id")
    if thread_id not in _sessions:
        raise ValueError("Session inconnue : recommencez avec un nouveau récit.")
    fournisseur = _sessions[thread_id]
    return _graphe(fournisseur), {"configurable": {"thread_id": thread_id}, "tags": ["web", fournisseur]}


def reprendre(body: dict) -> dict:
    app, config = _session(body)
    out = app.invoke(Command(resume=(body.get("reponse") or "").strip()), config)
    return {"thread_id": body["thread_id"], **_etat(out)}


def rechercher(body: dict) -> dict:
    app, config = _session(body)
    valeurs = app.get_state(config).values
    if "contexte_rag" not in valeurs:
        raise ValueError("Le dossier n'est pas encore complet.")
    models = Models(chat_model=LEGAL_RESEARCH_MODEL)
    findings = research(
        valeurs["contexte_rag"], embedding_model=models.embedding_model,
        embed=models.embed, chat=models.chat, index=DEFAULT_INDEX,
        top_k=max(1, int(body.get("top_k") or 5)),
    )
    return {"findings": [
        {
            "question": f.question,
            "answer": f.answer,
            "sources": [
                {"label": label, "ref": format_source(chunk), "text": chunk.text}
                for label, chunk in f.sources.items()
            ],
        }
        for f in findings
    ]}


def recits() -> dict:
    if not STORIES.exists():
        return {"stories": []}
    data = json.loads(STORIES.read_text(encoding="utf-8"))
    return {"stories": [
        {"id": s["id"], "objectif": s.get("objectif", ""), "recit": s["recit"],
         "reponses": s.get("reponses_clarification", [])}
        for s in data.get("stories", [])
    ]}


ROUTES_POST = {"/api/start": demarrer, "/api/resume": reprendre, "/api/research": rechercher}


class Handler(BaseHTTPRequestHandler):
    def _respond(self, status: int, content_type: str, data: bytes) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _json(self, status: int, payload: dict) -> None:
        self._respond(status, "application/json; charset=utf-8", json.dumps(payload, ensure_ascii=False).encode())

    def do_GET(self) -> None:
        if self.path in ("/", "/index.html"):
            data = (ICI / "index.html").read_bytes()
            self._respond(HTTPStatus.OK, "text/html; charset=utf-8", data)
        elif self.path == "/api/stories":
            self._json(HTTPStatus.OK, recits())
        elif self.path == "/api/providers":
            self._json(HTTPStatus.OK, {"providers": list(FOURNISSEURS)})
        else:
            self._json(HTTPStatus.NOT_FOUND, {"error": "Introuvable"})

    def do_POST(self) -> None:
        route = ROUTES_POST.get(self.path)
        if route is None:
            return self._json(HTTPStatus.NOT_FOUND, {"error": "Introuvable"})
        try:
            longueur = int(self.headers.get("Content-Length") or 0)
            body = json.loads(self.rfile.read(longueur) or b"{}")
            self._json(HTTPStatus.OK, route(body))
        except ValueError as exc:
            self._json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
        except Exception as exc:  # erreurs API (clé, quota, réseau…) : affichées dans la page
            self._json(HTTPStatus.BAD_GATEWAY, {"error": f"{type(exc).__name__} : {exc}"})

    def log_message(self, fmt: str, *args) -> None:
        print(f"[web] {self.command} {self.path} → {args[1] if len(args) > 1 else ''}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Interface web locale de BellAI.")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--no-browser", action="store_true", help="Ne pas ouvrir le navigateur")
    args = parser.parse_args()
    load_dotenv()
    url = f"http://127.0.0.1:{args.port}"
    serveur = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"BellAI → {url}  (Ctrl+C pour arrêter)")
    if not args.no_browser:
        threading.Timer(0.5, webbrowser.open, [url]).start()
    try:
        serveur.serve_forever()
    except KeyboardInterrupt:
        print("\nArrêt.")


if __name__ == "__main__":
    main()
