"""Research the legal questions prepared by the intake agent."""

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from ...RAG.answer import generate_answer
from ...RAG.documents import Chunk
from ...RAG.index import DEFAULT_INDEX, Embed, load_index, retrieve
from ..intake_agent.rag_context import ContexteRAG


@dataclass(frozen=True)
class ResearchFinding:
    question: str
    answer: str
    sources: dict[str, Chunk]


def research(
    context: ContexteRAG,
    *,
    embedding_model: str,
    embed: Embed,
    chat: Callable[[str, str], str],
    index: Path = DEFAULT_INDEX,
    top_k: int = 5,
) -> list[ResearchFinding]:
    """Return a grounded finding per distinct, nonempty query."""
    if top_k < 1:
        raise ValueError("Provide a positive top-k.")
    queries = dict.fromkeys(query.strip() for query in context["requetes"] if query.strip())
    if not queries:
        return []

    collection = load_index(index, embedding_model)
    findings = []
    # The corpus has source and page metadata, but no verified theme or jurisdiction labels.
    for question in queries:
        matches = retrieve(collection, question, top_k, embed)
        answer, sources = generate_answer(question, matches, chat, language=context["langue"])
        findings.append(ResearchFinding(question, answer, sources))
    return findings
