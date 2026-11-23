"""Essai interactif de l'intake agent dans le terminal.

    uv run python -m bellai.agents.intake_agent
"""

import argparse
import json
import sys
import uuid
from pathlib import Path

from chromadb.errors import ChromaError
from dotenv import load_dotenv
from langgraph.types import Command

from ..legal_research_agent.answer import format_source
from ..legal_research_agent.index import DEFAULT_INDEX
from ...config import LEGAL_RESEARCH_MODEL, Models
from ..legal_research_agent import research
from . import build_graph
from .graph import FOURNISSEURS, extracteur_pour


def main() -> int:
    parser = argparse.ArgumentParser(description="Recueillir un dossier et rechercher des sources juridiques.")
    parser.add_argument(
        "--provider", choices=FOURNISSEURS,
        help="Fournisseur de l’intake (INTAKE_PROVIDER ou Mistral par défaut)",
    )
    parser.add_argument("--research", action="store_true", help="Rechercher après les clarifications")
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()
    if args.top_k < 1:
        parser.error("--top-k doit être positif.")

    load_dotenv()
    try:
        app = build_graph(extracteur=extracteur_pour(args.provider))
    except ValueError as exc:
        print(f"Erreur de configuration : {exc}", file=sys.stderr)
        return 1
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}

    print("Racontez votre situation (ligne vide pour terminer) :")
    lignes = []
    while (ligne := input("> ")) != "":
        lignes.append(ligne)
    out = app.invoke({"messages": [("user", "\n".join(lignes))]}, config)

    # Boucle de clarification : le graphe se met en pause tant qu'il manque des infos
    while "__interrupt__" in out:
        print("\nL'agent a besoin de précisions :")
        for q in out["__interrupt__"][0].value:
            print(f"  - {q}")
        reponse = input("> ")
        out = app.invoke(Command(resume=reponse), config)

    print("\n=== CaseFile ===")
    print(out["case_file"].model_dump_json(indent=2, exclude_defaults=True))
    print(f"\nAncienneté (mois) : {out.get('anciennete_mois')}")
    print("\n=== Contexte RAG ===")
    print(json.dumps(out["contexte_rag"], indent=2, ensure_ascii=False))

    if args.research:
        models = Models(chat_model=LEGAL_RESEARCH_MODEL)
        try:
            findings = research(
                out["contexte_rag"], embedding_model=models.embedding_model,
                embed=models.embed, chat=models.chat, index=args.index, top_k=args.top_k,
            )
        except (ChromaError, OSError, ValueError) as exc:
            print(f"Erreur de recherche : {exc}", file=sys.stderr)
            return 1
        print("\n=== Recherche juridique ===")
        if out["manquants"]:
            print("Éléments du dossier manquants :", ", ".join(out["manquants"]))
        if not findings:
            print("Le dossier ne contient aucune question ni aucun fait exploitable pour la recherche.")
        for finding in findings:
            print(f"\n{finding.question}\n{finding.answer}")
            for label, chunk in finding.sources.items():
                print(f"[{label}] {format_source(chunk)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
