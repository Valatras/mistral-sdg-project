"""CLI for local ingestion and grounded questions."""

import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv
from chromadb.errors import ChromaError

from .RAG.answer import format_source, generate_answer
from .RAG.documents import load_chunks
from .RAG.index import DEFAULT_INDEX, build_index, load_index, retrieve
from .config import configured_models


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
        command.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    args = parser.parse_args(argv)
    models = configured_models()
    try:
        if args.command == "ingest":
            chunks, issues = load_chunks(args.documents, args.chunk_size, args.chunk_overlap)
            for issue in issues:
                print(issue, file=sys.stderr)
            count = len({chunk.source for chunk in chunks})
            print(f"Extracted {len(chunks)} chunks from {count} files. {len(issues)} issues.", flush=True)
            legacy_path = DEFAULT_INDEX.with_name("index.json") if args.index == DEFAULT_INDEX else None
            updated = build_index(
                args.index, models.embedding_model, chunks, models.embed,
                rebuild=args.rebuild, legacy_path=legacy_path,
            )
            print(f"{'Built' if updated else 'Reused'} index at {args.index}")
            return 1 if issues else 0

        collection = load_index(args.index, models.embedding_model)
        matches = retrieve(collection, args.question, args.top_k, models.embed)
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
    except (ChromaError, OSError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
