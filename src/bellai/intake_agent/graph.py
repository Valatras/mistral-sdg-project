"""Graphe LangGraph de l'intake agent : extraction → validation → (clarification) → router."""

from typing import Annotated, Any, Callable, TypedDict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.types import interrupt

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


def default_extracteur(model: str = "mistral-small-latest") -> Extracteur:
    """Extracteur Mistral (lit MISTRAL_API_KEY dans l'environnement)."""
    from langchain_mistralai import ChatMistralAI

    llm = ChatMistralAI(model=model, temperature=0)
    structured = llm.with_structured_output(CaseFile)
    return structured.invoke


def build_graph(extracteur: Extracteur | None = None, checkpointer: Any | None = None):
    """Construit et compile le graphe. `extracteur` est injectable (tests sans appel API)."""
    extraire = extracteur or default_extracteur()

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
    g.add_edge("preparation_rag", END)  # à remplacer par l'agent généraliste

    if checkpointer is None:
        # Autorise explicitement CaseFile dans les checkpoints (sinon avertissement, bloqué à terme)
        serde = JsonPlusSerializer(allowed_msgpack_modules=[("bellai.intake_agent.schema", "CaseFile")])
        checkpointer = MemorySaver(serde=serde)
    return g.compile(checkpointer=checkpointer)
