"""Build, persist and search a small local vector index."""

import hashlib
import json
import math
import os
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import NamedTuple

from .documents import Chunk

Embed = Callable[[list[str]], list[list[float]]]


class Entry(NamedTuple):
    chunk: Chunk
    vector: list[float]


class Match(NamedTuple):
    chunk: Chunk
    score: float


def normalize(vector: list[float]) -> list[float]:
    norm = math.hypot(*vector)
    if not norm or not math.isfinite(norm):
        raise ValueError("Invalid embedding vector.")
    return [value / norm for value in vector]


def load_index(path: Path, model: str) -> tuple[str, list[Entry]]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise ValueError(f"No index at {path}. Run bellai ingest first.") from None
    try:
        if data["version"] != 1 or data["model"] != model:
            raise ValueError("Index format or embedding model changed. Run bellai ingest.")
        entries = [Entry(Chunk(**row["chunk"]), row["vector"]) for row in data["entries"]]
        if not entries or not entries[0].vector:
            raise ValueError("The index is empty. Run bellai ingest.")
        return data["fingerprint"], entries
    except (KeyError, TypeError) as exc:
        raise ValueError("Invalid index. Run bellai ingest.") from exc


def build_index(
    path: Path, model: str, chunks: list[Chunk], embed: Embed, *, rebuild: bool = False
) -> bool:
    if not chunks:
        raise ValueError("No usable text was extracted. The index was not updated.")
    fingerprint = hashlib.sha256("\n".join(chunk.id for chunk in chunks).encode()).hexdigest()
    if path.is_file() and not rebuild:
        try:
            existing, _ = load_index(path, model)
        except ValueError:
            pass
        else:
            if existing == fingerprint:
                return False

    entries = []
    dimension = None
    for start in range(0, len(chunks), 32):
        batch = chunks[start:start + 32]
        vectors = embed([chunk.text for chunk in batch])
        for chunk, vector in zip(batch, vectors, strict=True):
            if dimension is None:
                dimension = len(vector)
            if len(vector) != dimension:
                raise ValueError("Inconsistent embedding dimensions.")
            entries.append({"chunk": asdict(chunk), "vector": normalize(vector)})

    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=path.parent, encoding="utf-8", mode="w", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump({"version": 1, "model": model, "fingerprint": fingerprint, "entries": entries},
                      handle, ensure_ascii=False, allow_nan=False)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return True


def retrieve(entries: list[Entry], question: str, k: int, embed: Embed) -> list[Match]:
    if k < 1 or not question.strip():
        raise ValueError("Provide a nonempty question and a positive top-k.")
    vector, = embed([question])
    query = normalize(vector)
    matches = [
        Match(chunk, sum(a * b for a, b in zip(query, vector, strict=True)))
        for chunk, vector in entries
    ]
    return sorted(matches, key=lambda match: match.score, reverse=True)[:k]
