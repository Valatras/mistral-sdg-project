"""Essai interactif de l'intake agent dans le terminal.

    uv run python -m bellai.intake_agent
    uv run python -m bellai.intake_agent --provider claude
"""

import argparse
import json
import uuid

from dotenv import load_dotenv
from langgraph.types import Command

from . import build_graph
from .graph import FOURNISSEURS, extracteur_pour


def main() -> None:
    parser = argparse.ArgumentParser(description="Essai interactif de l'intake agent")
    parser.add_argument("--provider", choices=list(FOURNISSEURS),
                        help="Fournisseur du LLM (défaut : INTAKE_PROVIDER, sinon mistral)")
    args = parser.parse_args()

    load_dotenv()
    app = build_graph(extracteur=extracteur_pour(args.provider))
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


if __name__ == "__main__":
    main()
