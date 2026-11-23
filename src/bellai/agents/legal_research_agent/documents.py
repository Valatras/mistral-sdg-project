"""Extract local documents and preserve page provenance through chunking."""

import hashlib
import re
import subprocess
from collections.abc import Iterator
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path

# Extensions reconnues à l’ingestion, indépendamment de la casse.
SUPPORTED = {".pdf", ".txt", ".md"}
# Repère les débuts d’articles tels que « Art. 3 », « Article premier » ou « Article unique ».
ARTICLE = re.compile(
    r"(?im)^[ \t]*(?:art(?:icle)?\.?[ \t]+(?:\d|premier\b|unique\b))"
)
CUT_PATTERNS = tuple(re.compile(pattern) for pattern in (r"\n\s*\n", r"[.!?]\s+", r"\n", r"\s+"))


@dataclass(frozen=True)
# Passage de texte indexable et informations permettant de retrouver sa source.
class Chunk:
    # Identifiant stable dérivé de l’emplacement et du texte du passage.
    id: str
    # Chemin du fichier relatif à la racine du corpus.
    source: str
    # Numéro de page PDF à partir de 1, ou None pour un fichier texte.
    page: int | None
    # Contenu utilisé pour l’embedding et transmis au modèle si le passage est retenu.
    text: str


# Lit le document et renvoie une liste de couples (numéro de page, texte).
def read_pages(path: Path) -> list[tuple[int | None, str]]:
    if path.suffix.lower() != ".pdf":
        # Lit le fichier texte d’un seul bloc. utf-8-sig accepte un éventuel BOM.
        return [(None, path.read_text(encoding="utf-8-sig"))]

    try:
        # Poppler extrait le texte UTF-8 sur stdout en conservant la disposition des lignes.
        result = subprocess.run(
            ["pdftotext", "-enc", "UTF-8", "-layout", str(path.resolve()), "-"],
            capture_output=True, check=True, timeout=120,
        )
    except FileNotFoundError:
        raise RuntimeError("PDF extraction requires pdftotext (Poppler).") from None
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"pdftotext exited with status {exc.returncode}") from None
    # Sépare les pages sur les sauts de page et retire le séparateur final.
    pages = result.stdout.decode("utf-8").removesuffix("\f").split("\f")
    # Associe les textes à leurs pages PDF en conservant la position des pages vides.
    return list(enumerate(pages, 1))


# Produit des passages d’au plus size caractères.
# overlap fixe le recouvrement des fenêtres au sein d’une même section.
def split_text(text: str, size: int, overlap: int) -> Iterator[str]:
    """Préserve les sections courtes et privilégie les coupures naturelles."""
    if not 0 <= overlap < size:
        raise ValueError("Chunk settings must satisfy 0 <= overlap < size.")
    # Délimite les sections avec les débuts d’articles et les deux bords du texte.
    boundaries = [0, *(m.start() for m in ARTICLE.finditer(text) if m.start()), len(text)]
    for left, right in pairwise(boundaries):
        section = text[left:right].strip()
        start = 0
        while start < len(section):
            end = min(start + size, len(section))
            if end < len(section):
                # Cherche une coupure dans la seconde moitié et garantit que la fenêtre avance malgré le recouvrement.
                minimum = start + max(size // 2, overlap + 1)
                # Préfère un paragraphe, puis une phrase, une ligne ou un espace.
                for pattern in CUT_PATTERNS:
                    cut_end = max((cut.end() for cut in pattern.finditer(section, minimum, end)), default=0)
                    if cut_end:
                        end = cut_end
                        break
            chunk = section[start:end].strip()
            if chunk:
                yield chunk
            if end == len(section):
                break
            # Reprend les derniers caractères de la fenêtre précédente.
            start = end - overlap


# Parcourt root récursivement et applique size et overlap à chaque page ou fichier texte.
def load_chunks(root: Path, size: int, overlap: int) -> tuple[list[Chunk], list[str]]:
    if not root.is_dir():
        raise ValueError(f"Document directory does not exist: {root}")
    if not 0 <= overlap < size:
        raise ValueError("Chunk settings must satisfy 0 <= overlap < size.")
    # Accumule les passages exploitables et les problèmes destinés au rapport d’ingestion.
    chunks: list[Chunk] = []
    issues: list[str] = []
    # Trie les chemins pour garantir un ordre de traitement stable.
    for path in sorted(root.rglob("*")):
        # Construit le chemin affiché dans les citations et les rapports.
        source = path.relative_to(root).as_posix()
        # Écarte les liens symboliques, qui pourraient pointer hors du corpus.
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
                # Précise la page dans le signalement lorsqu’un numéro est disponible.
                location = f"{source}, p. {page}" if page is not None else source
                issues.append(f"{location}: no extractable text")
                continue
            for number, part in enumerate(split_text(text, size, overlap), 1):
                # Combine chemin, page, position et texte pour distinguer les passages.
                # Le préfixe de l’empreinte SHA-256 forme l’identifiant stocké avec le passage.
                identity = f"{source}\0{page}\0{number}\0{part}"
                chunk_id = hashlib.sha256(identity.encode()).hexdigest()[:24]
                chunks.append(Chunk(chunk_id, source, page, part))
    # Renvoie les passages extraits et la liste des problèmes rencontrés.
    return chunks, issues
