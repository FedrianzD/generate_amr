"""Offline data-preservation, resume, and rollback checks for the converter."""

import importlib.util
from contextlib import closing
import sqlite3
import tempfile
import zipfile
from pathlib import Path

import penman


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "documents_to_sqlite", ROOT / "scripts/liputan6/documents_to_sqlite.py"
)
converter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(converter)


def graph(doc, index, concept="say-01"):
    return (f'# ::id {doc}_{index}\n# ::snt Teks Indonesia é.\n'
            '# ::input_truncated true\n# ::custom preserved value\n'
            f'(s / {concept} :ARG0 (p / person) :ARG1 p '
            ':value "quoted (text)" :polarity -)\n')


def expect_failure(folder, database, message, commit_every=100):
    try:
        converter.convert(folder, database, commit_every=commit_every)
    except ValueError as exc:
        assert message in str(exc), str(exc)
    else:
        raise AssertionError(f"Expected failure: {message}")


def main():
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        source = root / "train"
        source.mkdir()
        database = root / "train.sqlite"
        doc = source / "100.txt"
        original = graph("100", 10) + "\n" + graph("100", 2)
        doc.write_text(original, encoding="utf-8-sig")
        report = converter.convert(source, database, commit_every=1)
        assert report["stored_documents"] == 1
        assert report["added_graphs"] == 2
        with closing(sqlite3.connect(database)) as connection:
            rows = connection.execute(
                "SELECT sent_idx, amr_penman FROM graphs ORDER BY sent_idx"
            ).fetchall()
        assert [row[0] for row in rows] == [2, 10]
        for index, text in rows:
            assert converter.same_graph(
                penman.decode(graph("100", index), model=converter.MODEL),
                penman.decode(text, model=converter.MODEL),
            )
        assert doc.read_text(encoding="utf-8-sig") == original
        assert converter.convert(source, database)["added_graphs"] == 0
        doc.write_text(original + "\n" + graph("100", 3), encoding="utf-8")
        assert converter.convert(source, database)["added_graphs"] == 1
        # New sentence 1 precedes conflicting sentence 2: the insert must roll back.
        doc.write_text(graph("100", 1) + "\n" + graph("100", 2, "change-01"),
                       encoding="utf-8")
        expect_failure(source, database, "Conflicting stored graph")
        with closing(sqlite3.connect(database)) as connection:
            assert connection.execute("SELECT COUNT(*) FROM graphs").fetchone() == (3,)
        # Missing source rows remain in storage; partial documents are valid.
        doc.write_text(graph("100", 2), encoding="utf-8")
        assert converter.convert(source, database)["stored_graphs"] == 3
        for text, error in [("", "Empty document"),
                            (graph("wrong", 0), "Missing/invalid sentence ID"),
                            (graph("100", 0) + "\n" + graph("100", 0),
                             "Duplicate sentence index"),
                            ("# ::id 100_0\n(b / broken", "Failed importing")]:
            doc.write_text(text, encoding="utf-8")
            expect_failure(source, database, error)
        # Earlier batches persist, but the failed batch is rolled back as a unit.
        batch = root / "batch"
        batch.mkdir()
        (batch / "1.txt").write_text(graph("1", 0), encoding="utf-8")
        (batch / "2.txt").write_text(graph("2", 0), encoding="utf-8")
        (batch / "3.txt").write_text("", encoding="utf-8")
        batch_db = root / "batch.sqlite"
        expect_failure(batch, batch_db, "Empty document", commit_every=2)
        with closing(sqlite3.connect(batch_db)) as connection:
            assert connection.execute("SELECT COUNT(*) FROM graphs").fetchone() == (2,)
        rollback_db = root / "rollback.sqlite"
        expect_failure(batch, rollback_db, "Empty document", commit_every=3)
        with closing(sqlite3.connect(rollback_db)) as connection:
            assert connection.execute("SELECT COUNT(*) FROM graphs").fetchone() == (0,)
        legacy = root / "dev"
        legacy.mkdir()
        (legacy / "200_0.txt").write_text(graph("200", 0), encoding="utf-8")
        (legacy / "200_2.txt").write_text(graph("200", 2), encoding="utf-8")
        legacy_db = root / "dev.sqlite"
        legacy_report = converter.convert(legacy, legacy_db, split="dev")
        assert legacy_report["stored_documents"] == 1
        assert legacy_report["stored_graphs"] == 2
        with closing(sqlite3.connect(legacy_db)) as connection:
            assert connection.execute(
                "SELECT split, doc_id, sent_idx FROM graphs ORDER BY sent_idx"
            ).fetchall() == [("dev", "200", 0), ("dev", "200", 2)]
        assert converter.convert(legacy, legacy_db, split="dev")["added_graphs"] == 0
        zipped = root / "dev.zip"
        with zipfile.ZipFile(zipped, "w") as archive:
            for path in legacy.glob("*.txt"):
                archive.write(path, arcname=f"amr_graphs/{path.name}")
        assert converter.convert(zipped, legacy_db, split="dev")["unchanged_graphs"] == 2
        zip_report = converter.convert(zipped, root / "zip.sqlite", split="dev")
        assert zip_report["stored_documents"] == 1 and zip_report["stored_graphs"] == 2
        unrelated = root / "unrelated.sqlite"
        with closing(sqlite3.connect(unrelated)) as connection:
            connection.execute("CREATE TABLE other (value TEXT)")
        expect_failure(batch, unrelated, "unrelated database")
    print("PASS: metadata/triples, numeric order, source preservation, partial documents, "
          "idempotence, incremental resume, malformed inputs, conflict and batch rollback")


if __name__ == "__main__":
    main()
