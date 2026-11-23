"""Persist passage embeddings and search them with local Chroma."""

import hashlib
import json
import math
import os
from collections.abc import Callable
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import NamedTuple
from uuid import uuid4

import chromadb
from chromadb.api import ClientAPI
from chromadb.api.models.Collection import Collection
from chromadb.errors import ChromaError, NotFoundError

from .documents import Chunk

Embed = Callable[[list[str]], list[list[float]]]
DEFAULT_INDEX = Path(".bellai/chroma")


class Match(NamedTuple):
    chunk: Chunk
    score: float


def _validate_vectors(vectors: list[list[float]], count: int) -> None:
    if len(vectors) != count or any(
        not vector or not (0 < math.hypot(*vector) < math.inf) for vector in vectors
    ):
        raise ValueError("Invalid embedding response.")


def _active_collection(path: Path, client: ClientAPI) -> Collection | None:
    try:
        name = (path / "active").read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        return None
    try:
        return client.get_collection(name, embedding_function=None)
    except NotFoundError:
        raise ValueError("The active Chroma collection is missing. Run bellai ingest.") from None


def load_index(path: Path, model: str) -> Collection:
    if not (path / "active").is_file():
        raise ValueError(f"No Chroma index at {path}. Run bellai ingest first.")
    collection = _active_collection(path, chromadb.PersistentClient(path=str(path.resolve())))
    if collection is None:
        raise ValueError(f"No Chroma index at {path}. Run bellai ingest first.")
    if (collection.metadata or {}).get("model") != model:
        raise ValueError("The embedding model changed. Run bellai ingest.")
    if not collection.count():
        raise ValueError("The index is empty. Run bellai ingest.")
    return collection


def _legacy_vectors(
    path: Path, model: str, chunks: list[Chunk]
) -> dict[str, list[float]]:
    """Reuse compatible vectors from the original default JSON index on first ingest."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if data["version"] != 1 or data["model"] != model:
            return {}
        ids = {chunk.id for chunk in chunks}
        return {
            row["chunk"]["id"]: row["vector"]
            for row in data["entries"]
            if row["chunk"]["id"] in ids
        }
    except (OSError, ValueError, KeyError, TypeError):
        return {}


def build_index(
    path: Path,
    model: str,
    chunks: list[Chunk],
    embed: Embed,
    *,
    rebuild: bool = False,
    legacy_path: Path | None = None,
) -> bool:
    if not chunks:
        raise ValueError("No usable text was extracted. The index was not updated.")
    if path.is_file():
        raise ValueError("Pass a directory to --index for Chroma.")

    fingerprint = hashlib.sha256("\n".join(chunk.id for chunk in chunks).encode()).hexdigest()
    client = chromadb.PersistentClient(path=str(path.resolve()))
    active = _active_collection(path, client)
    metadata = {"model": model, "fingerprint": fingerprint}
    if (
        not rebuild and active is not None and active.metadata == metadata
        and active.count() == len(chunks)
    ):
        return False

    cached = {}
    if legacy_path is not None and not rebuild and active is None:
        cached = _legacy_vectors(legacy_path, model, chunks)
    name = f"passages-{uuid4().hex}"
    collection = client.create_collection(
        name,
        metadata=metadata,
        configuration={"hnsw": {"space": "cosine"}},
        embedding_function=None,
    )
    temporary = None
    try:
        for start in range(0, len(chunks), 32):
            batch = chunks[start:start + 32]
            missing = [chunk for chunk in batch if chunk.id not in cached]
            if missing:
                generated = embed([chunk.text for chunk in missing])
                _validate_vectors(generated, len(missing))
                cached.update((chunk.id, vector) for chunk, vector in zip(missing, generated, strict=True))
            vectors = [cached.pop(chunk.id) for chunk in batch]
            _validate_vectors(vectors, len(batch))
            collection.add(
                ids=[chunk.id for chunk in batch],
                embeddings=vectors,
                documents=[chunk.text for chunk in batch],
                metadatas=[
                    {"source": chunk.source, **({"page": chunk.page} if chunk.page is not None else {})}
                    for chunk in batch
                ],
            )
        # Publish only complete collections. Previous readers can finish on their own collection.
        with NamedTemporaryFile(dir=path, mode="w", encoding="utf-8", delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(name)
        os.replace(temporary, path / "active")
    except Exception:
        try:
            client.delete_collection(name)
        except ChromaError:
            pass
        raise
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return True


def retrieve(collection: Collection, question: str, k: int, embed: Embed) -> list[Match]:
    if k < 1 or not question.strip():
        raise ValueError("Provide a nonempty question and a positive top-k.")
    vectors = embed([question])
    _validate_vectors(vectors, 1)
    result = collection.query(
        query_embeddings=vectors, n_results=k, include=["documents", "metadatas", "distances"]
    )
    return [
        Match(Chunk(id_, metadata["source"], metadata.get("page"), document), 1.0 - distance)
        for id_, metadata, document, distance in zip(
            result["ids"][0], result["metadatas"][0], result["documents"][0], result["distances"][0],
            strict=True,
        )
    ]
