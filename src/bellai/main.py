"""CLI for local ingestion and grounded questions."""

import argparse
import os
import sys
from functools import cached_property
from pathlib import Path

from dotenv import load_dotenv
from mistralai.client import Mistral

from .RAG.answer import format_source, generate_answer
from .RAG.documents import load_chunks
from .RAG.index import build_index, load_index, retrieve


class Models:
    def __init__(self, embedding_model: str, chat_model: str):
        self.embedding_model = embedding_model
        self.chat_model = chat_model

    @cached_property
    def client(self) -> Mistral:
        key = os.getenv("MISTRAL_API_KEY", "").strip()
        if not key:
            raise ValueError("Set MISTRAL_API_KEY in .env or your environment.")
        return Mistral(api_key=key)

    def embed(self, texts: list[str]) -> list[list[float]]:
        client = self.client
        try:
            response = client.embeddings.create(model=self.embedding_model, inputs=texts)
            rows = sorted(response.data, key=lambda row: row.index)
            if [row.index for row in rows] != list(range(len(texts))):
                raise ValueError("Unexpected embedding response indices.")
            return [list(row.embedding) for row in rows]
        except Exception as exc:
            raise ValueError(f"Mistral embedding request failed ({type(exc).__name__}).") from None

    def chat(self, system: str, payload: str) -> str:
        client = self.client
        try:
            response = client.chat.complete(
                model=self.chat_model,
                messages=[{"role": "system", "content": system}, {"role": "user", "content": payload}],
                temperature=0,
                response_format={"type": "json_object"},
            )
            content = response.choices[0].message.content
            if not isinstance(content, str):
                raise ValueError("Unexpected response content.")
            return content
        except Exception as exc:
            raise ValueError(f"Mistral chat request failed ({type(exc).__name__}).") from None


def main(argv: list[str] | None = None) -> int:
    load_dotenv(Path.cwd() / ".env")
    parser = argparse.ArgumentParser(prog="bellai", description="Explore local Belgian employment-law documents.")
    commands = parser.add_subparsers(dest="command", required=True)

    ingest = commands.add_parser("ingest", help="Build or reuse the local index")
    ingest.add_argument("--documents", type=Path, default=Path("fichier_rag"))
    ingest.add_argument("--chunk-size", type=int, default=1200)
    ingest.add_argument("--chunk-overlap", type=int, default=150)
    ingest.add_argument("--rebuild", action="store_true", help="Regenerate all embeddings")

    ask = commands.add_parser("ask", help="Retrieve excerpts and generate a cited answer")
    ask.add_argument("question")
    ask.add_argument("--top-k", type=int, default=5)
    ask.add_argument("--debug", action="store_true")

    for command in (ingest, ask):
        command.add_argument("--index", type=Path, default=Path(".bellai/index.json"))
    args = parser.parse_args(argv)
    models = Models(
        os.getenv("MISTRAL_EMBED_MODEL", "mistral-embed"),
        os.getenv("MISTRAL_CHAT_MODEL", "mistral-small-latest"),
    )
    try:
        if args.command == "ingest":
            chunks, issues = load_chunks(args.documents, args.chunk_size, args.chunk_overlap)
            for issue in issues:
                print(issue, file=sys.stderr)
            count = len({chunk.source for chunk in chunks})
            print(f"Extracted {len(chunks)} chunks from {count} files. {len(issues)} issues.", flush=True)
            updated = build_index(args.index, models.embedding_model, chunks, models.embed, rebuild=args.rebuild)
            print(f"{'Built' if updated else 'Reused'} index at {args.index}")
            return 1 if issues else 0

        _, entries = load_index(args.index, models.embedding_model)
        matches = retrieve(entries, args.question, args.top_k, models.embed)
        if args.debug:
            for i, (chunk, score) in enumerate(matches, 1):
                print(f"[S{i}] {format_source(chunk)} | score={score:.4f} | chunk={chunk.id}\n{chunk.text}\n",
                      file=sys.stderr)
        answer, sources = generate_answer(args.question, matches, models.chat)
        print(answer)
        if sources:
            print("\nSources")
            for label, chunk in sources.items():
                print(f"[{label}] {format_source(chunk)}")
        return 0
    except (OSError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
