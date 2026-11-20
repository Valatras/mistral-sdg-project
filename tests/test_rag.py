import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import Mock, patch

from bellai.RAG.answer import format_source, generate_answer
from bellai.RAG.documents import Chunk, load_chunks, split_text
from bellai.RAG.index import Match, build_index, load_index, retrieve
from bellai.main import main


class RagTests(unittest.TestCase):
    def test_loading_preserves_relative_path_pdf_pages_and_stable_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "nested").mkdir()
            (root / "nested" / "text.txt").write_text("Préavis", encoding="utf-8")
            (root / "sample.PDF").touch()
            (root / "broken.pdf").touch()
            (root / "ignored.csv").touch()
            extracted = Mock(stdout="Page une\f\fPage trois\f".encode())
            with patch("bellai.RAG.documents.subprocess.run", return_value=extracted):
                chunks, issues = load_chunks(root, 100, 15)
                self.assertEqual(chunks, load_chunks(root, 100, 15)[0])
            self.assertIn(("nested/text.txt", None), [(c.source, c.page) for c in chunks])
            self.assertEqual([c.page for c in chunks if c.source == "sample.PDF"], [1, 3])
            self.assertIn("sample.PDF, p. 2: no extractable text", issues)
            self.assertIn("ignored.csv: unsupported format", issues)
            self.assertEqual(len({c.id for c in chunks}), len(chunks))
            with patch("bellai.RAG.documents.subprocess.run", side_effect=FileNotFoundError):
                _, failures = load_chunks(root, 100, 15)
            self.assertIn("broken.pdf: extraction failed", "\n".join(failures))

    def test_chunking_keeps_article_sections_and_overlaps_long_text(self):
        self.assertEqual(
            list(split_text("Art. 1. Premier texte.\n\nArticle 2. Second texte.", 100, 15)),
            ["Art. 1. Premier texte.", "Article 2. Second texte."],
        )
        self.assertEqual(list(split_text("abcdefghijklmnop", 8, 2)),
                         ["abcdefgh", "ghijklmn", "mnop"])
        with self.assertRaises(ValueError):
            list(split_text("text", 10, 10))

    def test_persistence_reuse_rebuild_and_retrieval(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "index.json"
            a = Chunk("1", "notice.pdf", 2, "Notice period")
            b = Chunk("2", "other.txt", None, "Something else")
            embed = Mock(return_value=[[2, 0], [0, 4]])
            self.assertTrue(build_index(path, "model", [a, b], embed))
            self.assertFalse(build_index(path, "model", [a, b], embed))
            embed.assert_called_once()
            _, entries = load_index(path, "model")
            self.assertEqual(retrieve(entries, "notice", 1, Mock(return_value=[[3, 0]])),
                             [Match(a, 1.0)])
            with self.assertRaisesRegex(ValueError, "embedding model changed"):
                load_index(path, "other-model")
            original = path.read_bytes()
            with self.assertRaises(ValueError):
                build_index(path, "model", [a, b], Mock(return_value=[[0, 0]]), rebuild=True)
            self.assertEqual(path.read_bytes(), original)
            self.assertTrue(build_index(path, "model", [a, b], embed, rebuild=True))
            self.assertTrue(build_index(path, "model", [b], Mock(return_value=[[0, 4]])))

    def test_citations_and_abstention(self):
        a = Chunk("1", "nested/notice.pdf", 2, "Notice period")
        b = Chunk("2", "other.txt", None, "Other topic")
        matches = [Match(a, 0.9), Match(b, 0.1)]
        chat = Mock(return_value=json.dumps({"found": True, "answer": "See [S1]."}))
        self.assertEqual(generate_answer("Question?", matches, chat), ("See [S1].", {"S1": a}))
        self.assertEqual(format_source(a), "nested/notice.pdf, p. 2")
        self.assertEqual(format_source(b), "other.txt")
        self.assertEqual(json.loads(chat.call_args.args[1])["excerpts"][0]["page"], 2)
        chat.return_value = '{"found": true, "answer": "See [S1] and [S3]."}'
        with self.assertRaisesRegex(ValueError, "unknown citation"):
            generate_answer("Question?", matches, chat)
        chat.return_value = '{"found": false, "answer": "Sources insuffisantes."}'
        self.assertEqual(generate_answer("Question?", matches, chat), ("Sources insuffisantes.", {}))

    def test_cli_ingest_and_ask_with_debug(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            docs = root / "docs"
            docs.mkdir()
            (docs / "notice.txt").write_text("Exemple de texte.", encoding="utf-8")
            index_args = ["--index", str(root / "index.json")]
            stdout, stderr = io.StringIO(), io.StringIO()
            with (
                patch("bellai.main.Models.embed", return_value=[[1, 0]]),
                patch("bellai.main.Models.chat", return_value='{"found": true, "answer": "Exemple [S1]."}'),
                redirect_stdout(stdout),
                redirect_stderr(stderr),
            ):
                self.assertEqual(main(["ingest", "--documents", str(docs), *index_args]), 0)
                self.assertEqual(main(["ask", "Quel exemple ?", "--debug", *index_args]), 0)
            self.assertIn("[S1] notice.txt", stdout.getvalue())
            self.assertIn("score=1.0000", stderr.getvalue())
