"""Graphe LangGraph de l'intake agent : extraction → validation → (clarification) → router."""

import os
from typing import Annotated, Any, Callable, TypedDict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.types import interrupt
from langsmith import traceable
from pydantic import BaseModel, Field, create_model

from ...config import CLAUDE_INTAKE_MODEL, INTAKE_MODEL
from .normalisation import anciennete_mois, champs_requis_manquants, normaliser, rediger_questions
from .prompts import prompt_intake
from .rag_context import ContexteRAG, construire_contexte_rag
from .schema import CaseFile

MAX_TOURS_CLARIFICATION = 2


class State(TypedDict, total=False):
    messages: Annotated[list, add_messages]
    case_file: CaseFile | None
    anciennete_mois: int | None
    manquants: list[str]
    questions: list[str]
    tours_clarification: int
    contexte_rag: ContexteRAG


Extracteur = Callable[[list], CaseFile]


def default_extracteur(model: str | None = None) -> Extracteur:
    """Extracteur Mistral (lit MISTRAL_API_KEY dans l'environnement)."""
    from langchain_mistralai import ChatMistralAI

    llm = ChatMistralAI(model=model or INTAKE_MODEL, temperature=0)
    structured = llm.with_structured_output(CaseFile)
    return structured.invoke


def _schema_tout_requis(modele: type[BaseModel]) -> type[BaseModel]:
    """Copie du modèle où chaque champ est obligatoire (même type, défaut indiqué en description).

    Les sorties structurées de Claude compilent le schéma en grammaire : chaque champ optionnel
    l'agrandit, et les 23 champs optionnels de CaseFile dépassent la limite ("Schema is too complex").
    Claude remplit donc explicitement la valeur par défaut quand l'info n'est pas donnée.
    """
    champs = {}
    for nom, info in modele.model_fields.items():
        defaut = info.get_default(call_default_factory=True)
        description = f"{info.description or ''} (si non mentionné : {defaut!r})".strip()
        champs[nom] = (info.annotation, Field(..., description=description))
    return create_model(f"{modele.__name__}Requis", __doc__=modele.__doc__, **champs)


def claude_extracteur(model: str = CLAUDE_INTAKE_MODEL) -> Extracteur:
    """Extracteur Claude (lit ANTHROPIC_API_KEY dans l'environnement)."""
    import anthropic

    client = anthropic.Anthropic()
    schema_claude = _schema_tout_requis(CaseFile)
    ROLES = {"human": "user", "user": "user", "ai": "assistant", "assistant": "assistant"}

    @traceable(run_type="llm", name="claude_intake", metadata={"ls_provider": "anthropic", "ls_model_name": model})
    def extraire(messages: list) -> CaseFile:
        # Le graphe envoie [("system", prompt), *messages LangChain] : on convertit au format Claude
        system, history = "", []
        for m in messages:
            role, content = m if isinstance(m, tuple) else (m.type, m.content)
            if role == "system":
                system = content
            else:
                history.append({"role": ROLES[role], "content": content})

        response = client.messages.parse(
            model=model,
            max_tokens=16000,
            system=system,
            messages=history,
            output_format=schema_claude,         # tous champs requis (voir _schema_tout_requis)
            output_config={"effort": "low"},     # extraction simple : peu de réflexion nécessaire
        )
        if response.stop_reason == "refusal":
            raise RuntimeError("Claude a refusé la requête")
        return CaseFile.model_validate(response.parsed_output.model_dump())

    return extraire


FOURNISSEURS = {"mistral": default_extracteur, "claude": claude_extracteur}


def extracteur_pour(fournisseur: str | None = None) -> Extracteur:
    """Extracteur du fournisseur demandé, sinon INTAKE_PROVIDER, sinon Mistral."""
    nom = (fournisseur or os.getenv("INTAKE_PROVIDER") or "mistral").lower()
    if nom not in FOURNISSEURS:
        raise ValueError(f"Fournisseur inconnu : {nom!r} (choix : {', '.join(FOURNISSEURS)})")
    return FOURNISSEURS[nom]()


def build_graph(extracteur: Extracteur | None = None, checkpointer: Any | None = None):
    """Construit et compile le graphe. `extracteur` est injectable (tests sans appel API)."""
    extraire = extracteur or extracteur_pour()

    def intake(state: State) -> dict:
        case = extraire([("system", prompt_intake()), *state["messages"]])
        return {"case_file": case}

    def validation(state: State) -> dict:
        case = normaliser(state["case_file"])
        manquants = champs_requis_manquants(case)
        return {
            "case_file": case,
            "anciennete_mois": anciennete_mois(case),
            "manquants": manquants,
            # Les questions sont rédigées ici : à la reprise après interrupt(),
            # le nœud clarification est réexécuté, autant qu'il ne fasse rien de coûteux.
            "questions": rediger_questions(manquants, case.langue),
        }

    def clarification(state: State) -> dict:
        reponse = interrupt(state["questions"])
        return {
            "messages": [("user", reponse)],
            "tours_clarification": state.get("tours_clarification", 0) + 1,
        }

    def apres_validation(state: State) -> str:
        if state["questions"] and state.get("tours_clarification", 0) < MAX_TOURS_CLARIFICATION:
            return "clarification"
        return "preparation_rag"

    def preparation_rag(state: State) -> dict:
        return {"contexte_rag": construire_contexte_rag(state["case_file"])}

    g = StateGraph(State)
    g.add_node("intake", intake)
    g.add_node("validation", validation)
    g.add_node("clarification", clarification)
    g.add_node("preparation_rag", preparation_rag)
    g.add_edge(START, "intake")
    g.add_edge("intake", "validation")
    g.add_conditional_edges("validation", apres_validation, ["clarification", "preparation_rag"])
    g.add_edge("clarification", "intake")
    g.add_edge("preparation_rag", END)

    if checkpointer is None:
        # Autorise explicitement CaseFile dans les checkpoints (sinon avertissement, bloqué à terme)
        serde = JsonPlusSerializer(allowed_msgpack_modules=[(CaseFile.__module__, CaseFile.__name__)])
        checkpointer = MemorySaver(serde=serde)
    return g.compile(checkpointer=checkpointer)
