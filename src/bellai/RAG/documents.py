"""Extract local documents and preserve page provenance through chunking."""

import hashlib
import re
import subprocess
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

SUPPORTED = {".pdf", ".txt", ".md"}
ARTICLE = re.compile(
    r"(?im)^[ \t]*(?:art(?:icle)?\.?[ \t]+(?:\d|premier\b|unique\b))"
)


@dataclass(frozen=True)
class Chunk:
    id: str
    source: str
    page: int | None
    text: str


def read_pages(path: Path) -> list[tuple[int | None, str]]:
    if path.suffix.lower() != ".pdf":
        return [(None, path.read_text(encoding="utf-8-sig"))]

    try:
        result = subprocess.run(
            ["pdftotext", "-enc", "UTF-8", "-layout", str(path.resolve()), "-"],
            capture_output=True, check=True, timeout=120,
        )
    except FileNotFoundError:
        raise RuntimeError("PDF extraction requires pdftotext (Poppler).") from None
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"pdftotext exited with status {exc.returncode}") from None
    pages = result.stdout.decode("utf-8").removesuffix("\f").split("\f")
    return list(enumerate(pages, 1))


def split_text(text: str, size: int, overlap: int) -> Iterator[str]:
    """Keep short article sections intact and prefer natural boundaries in long ones."""
    if not 0 <= overlap < size:
        raise ValueError("Chunk settings must satisfy 0 <= overlap < size.")
    boundaries = [0, *(m.start() for m in ARTICLE.finditer(text) if m.start()), len(text)]
    for left, right in zip(boundaries, boundaries[1:]):
        section = text[left:right].strip()
        start = 0
        while start < len(section):
            end = min(start + size, len(section))
            if end < len(section):
                # Search the latter half of the window and always advance past overlap.
                minimum = start + max(size // 2, overlap + 1)
                for pattern in (r"\n\s*\n", r"[.!?]\s+", r"\n", r"\s+"):
                    cuts = list(re.finditer(pattern, section[minimum:end]))
                    if cuts:
                        end = minimum + cuts[-1].end()
                        break
            chunk = section[start:end].strip()
            if chunk:
                yield chunk
            if end == len(section):
                break
            start = end - overlap


def load_chunks(root: Path, size: int, overlap: int) -> tuple[list[Chunk], list[str]]:
    if not root.is_dir():
        raise ValueError(f"Document directory does not exist: {root}")
    if not 0 <= overlap < size:
        raise ValueError("Chunk settings must satisfy 0 <= overlap < size.")
    chunks: list[Chunk] = []
    issues: list[str] = []
    for path in sorted(root.rglob("*")):
        source = path.relative_to(root).as_posix()
        if path.is_symlink():
            issues.append(f"{source}: skipped symbolic link")
            continue
        if not path.is_file():
            continue
        if path.suffix.lower() not in SUPPORTED:
            issues.append(f"{source}: unsupported format")
            continue
        try:
            pages = read_pages(path)
        except (OSError, UnicodeError, RuntimeError, subprocess.TimeoutExpired) as exc:
            issues.append(f"{source}: extraction failed ({exc})")
            continue
        for page, text in pages:
            if not text.strip():
                location = f"{source}, p. {page}" if page is not None else source
                issues.append(f"{location}: no extractable text")
            for number, part in enumerate(split_text(text, size, overlap), 1):
                identity = f"{source}\0{page}\0{number}\0{part}"
                chunk_id = hashlib.sha256(identity.encode()).hexdigest()[:24]
                chunks.append(Chunk(chunk_id, source, page, part))
    return chunks, issues
