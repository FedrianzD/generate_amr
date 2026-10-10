"""Verify targeted SQLite SMATCH generation and lossless atomic NPZ merge."""
import json
import sqlite3
import sys
import tempfile
import zipfile
from contextlib import closing
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts/liputan6"))
import build_smatch_adjacency as builder
from npy_to_npz import merge_npz


def main():
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        source = root / "source/train"
        source.mkdir(parents=True)
        (source / "100.json").write_text(json.dumps({"clean_article": [["a"], ["b"]]}))
        database = root / "train.sqlite"
        with closing(sqlite3.connect(database)) as db, db:
            db.executescript("""
                CREATE TABLE storage_metadata (key TEXT PRIMARY KEY, value TEXT);
                INSERT INTO storage_metadata VALUES ('schema_version','amr-documents-v1');
                CREATE TABLE graphs (split TEXT, doc_id TEXT, sent_idx INTEGER,
                    sentence_id TEXT, amr_penman TEXT, PRIMARY KEY(split,doc_id,sent_idx));
            """)
            for index in [1, 0]:
                db.execute("INSERT INTO graphs VALUES (?,?,?,?,?)",
                           ("train", "100", index, f"100_{index}",
                            f"# ::id 100_{index}\n(a / alpha)"))
        ids = root / "ids.csv"
        ids.write_text("id\n100\n")
        output = root / "adjacency"
        output.mkdir()
        archive = output / "train.npz"
        np.savez_compressed(archive, **{"200": np.zeros((1, 1), dtype=np.float32)})
        old_bytes = archive.read_bytes()
        with zipfile.ZipFile(archive) as handle:
            old_member = handle.read("200.npy")
        report = builder.process_sqlite(database, "train", ids, root / "source", output, archive)
        assert report["merge"]["added"] == 1
        assert Path(report["merge"]["backup"]).read_bytes() == old_bytes
        with np.load(archive) as handle:
            assert set(handle.files) == {"100", "200"}
            np.testing.assert_array_equal(handle["100"], np.array([[0, 1], [1, 0]], np.float32))
        with zipfile.ZipFile(archive) as handle:
            assert handle.read("200.npy") == old_member
        published = archive.read_bytes()
        assert merge_npz(archive, {"100": output / "train/100.npy"})["added"] == 0
        assert archive.read_bytes() == published
        bad = root / "conflict.npy"
        np.save(bad, np.zeros((2, 2), dtype=np.float32))
        try:
            merge_npz(archive, {"100": bad})
        except ValueError as exc:
            assert "Conflicting" in str(exc)
        else:
            raise AssertionError("Expected conflict rejection")
        assert archive.read_bytes() == published
        with closing(sqlite3.connect(database)) as db, db:
            db.execute("DELETE FROM graphs WHERE sent_idx=1")
        try:
            builder.process_sqlite(database, "train", ids, root / "source", output, archive)
        except ValueError as exc:
            assert "Incomplete" in str(exc)
        else:
            raise AssertionError("Incomplete document accepted")
        assert archive.read_bytes() == published
        assert not list(output.rglob("*.tmp"))
    print("PASS: SQLite order/coverage, actual SMATCH, original member preservation, backup, "
          "idempotent merge, conflict rejection, incomplete-source rejection, atomic cleanup")


if __name__ == "__main__":
    main()
