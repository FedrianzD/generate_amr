"""Convert document-level PENMAN .txt files to resumable SQLite storage.

Standalone: upload this file to Kaggle; no project modules or GPU are needed.
Accepts {doc_id}.txt documents or legacy {doc_id}_{sent_idx}.txt sentence files,
with ::id {doc_id}_{sent_idx} on every graph. Partial documents are allowed.
"""

import argparse
import json
import re
import sqlite3
import time
import zipfile
from pathlib import Path

import penman
from penman.models.noop import NoOpModel


MODEL = NoOpModel()
SENTENCE_ID = re.compile(r"^(?P<doc_id>.+)_(?P<sent_idx>\d+)$")
SCHEMA_VERSION = "amr-documents-v1"


def open_database(path):
    """Use rollback journaling so a closed export is a standalone file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(str(path), timeout=60)
    try:
        tables = {row[0] for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )}
        if tables:
            if "storage_metadata" not in tables:
                raise ValueError("Output is an unrelated database; choose a new output path")
            version = connection.execute(
                "SELECT value FROM storage_metadata WHERE key='schema_version'"
            ).fetchone()
            if version != (SCHEMA_VERSION,):
                raise ValueError("Unsupported SQLite storage schema")
        connection.execute("PRAGMA journal_mode=DELETE")
        connection.execute("PRAGMA synchronous=FULL")
        with connection:
            connection.execute("""CREATE TABLE IF NOT EXISTS storage_metadata (
                key TEXT PRIMARY KEY, value TEXT NOT NULL
            )""")
            connection.execute(
                "INSERT OR IGNORE INTO storage_metadata VALUES ('schema_version', ?)",
                (SCHEMA_VERSION,),
            )
            connection.execute("""CREATE TABLE IF NOT EXISTS graphs (
                split TEXT NOT NULL,
                doc_id TEXT NOT NULL,
                sent_idx INTEGER NOT NULL CHECK(sent_idx >= 0),
                sentence_id TEXT NOT NULL,
                amr_penman TEXT NOT NULL,
                PRIMARY KEY (split, doc_id, sent_idx),
                UNIQUE (split, sentence_id)
            )""")
        return connection
    except BaseException:
        connection.close()
        raise


def same_graph(left, right):
    # Triple order, top, and every metadata entry are checked, not just IDs.
    return (left.top == right.top and left.triples == right.triples
            and left.metadata == right.metadata)


def read_document(path, text=None):
    graphs = penman.loads(path.read_text(encoding="utf-8-sig") if text is None else text,
                          model=MODEL)
    if not graphs:
        raise ValueError(f"Empty document: {path}")
    seen = set()
    rows = []
    for graph in graphs:
        sentence_id = graph.metadata.get("id", "")
        match = SENTENCE_ID.fullmatch(sentence_id)
        if match is None or not (
                match.group("doc_id") == path.stem
                or (len(graphs) == 1 and sentence_id == path.stem)):
            raise ValueError(f"Missing/invalid sentence ID {sentence_id!r} in {path}")
        index = int(match.group("sent_idx"))
        if index in seen:
            raise ValueError(f"Duplicate sentence index {index} in {path}")
        seen.add(index)
        serialized = penman.encode(graph, model=MODEL) + "\n"
        if not same_graph(graph, penman.decode(serialized, model=MODEL)):
            raise ValueError(f"PENMAN round-trip mismatch: {path}, {sentence_id}")
        rows.append((match.group("doc_id"), index, sentence_id, serialized, graph))
    return sorted(rows, key=lambda row: row[1])


def import_document(connection, path, split, text=None):
    rows = read_document(path, text=text)
    added = 0
    for doc_id, index, sentence_id, serialized, graph in rows:
        existing = connection.execute(
            "SELECT sentence_id, amr_penman FROM graphs "
            "WHERE split=? AND doc_id=? AND sent_idx=?",
            (split, doc_id, index),
        ).fetchone()
        if existing is not None:
            if (existing[0] != sentence_id or not same_graph(
                    graph, penman.decode(existing[1], model=MODEL))):
                raise ValueError(f"Conflicting stored graph: {split}/{sentence_id} ({path})")
            continue
        connection.execute(
            "INSERT INTO graphs VALUES (?, ?, ?, ?, ?)",
            (split, doc_id, index, sentence_id, serialized),
        )
        stored = connection.execute(
            "SELECT amr_penman FROM graphs WHERE split=? AND doc_id=? AND sent_idx=?",
            (split, doc_id, index),
        ).fetchone()[0]
        if stored != serialized:
            raise ValueError(f"SQLite read-back mismatch: {sentence_id}")
        added += 1
    return added, len(rows) - added


def convert(input_dir, output, split="train", commit_every=100):
    """Re-runs retain identical rows, add missing sentences, reject conflicts.

    A failed batch rolls back completely; earlier committed batches survive.
    Existing rows absent from the input are retained. No completeness is inferred.
    Read stable input snapshots (e.g. Kaggle datasets), not files being rewritten.
    """
    input_dir, output = Path(input_dir), Path(output)
    if split not in {"train", "dev", "test"}:
        raise ValueError("split must be train, dev, or test")
    is_archive = input_dir.is_file() and input_dir.suffix.lower() == ".zip"
    if not input_dir.is_dir() and not is_archive:
        raise FileNotFoundError(f"Document input folder missing: {input_dir}")
    if commit_every < 1:
        raise ValueError("commit_every must be positive")
    if output.suffix.lower() not in {".sqlite", ".sqlite3", ".db"}:
        raise ValueError("Output must have a .sqlite, .sqlite3, or .db suffix")
    print(f"Indexing {split}: {input_dir}", flush=True)
    archive = zipfile.ZipFile(input_dir) if is_archive else None
    paths = (sorted(member.filename for member in archive.infolist()
                    if not member.is_dir() and member.filename.endswith(".txt"))
             if archive is not None else sorted(input_dir.rglob("*.txt")))
    if not paths:
        raise ValueError(f"No document .txt files found in {input_dir}")
    print(f"Found {len(paths):,} input files; output: {output}", flush=True)
    started = time.monotonic()
    added = unchanged = processed = 0
    connection = open_database(output)
    try:
        for start in range(0, len(paths), commit_every):
            with connection:
                for path in paths[start:start + commit_every]:
                    try:
                        text = (archive.read(path).decode("utf-8-sig")
                                if archive is not None else None)
                        new, old = import_document(connection, Path(path), split, text=text)
                    except Exception as exc:
                        raise ValueError(f"Failed importing {path}: {exc}") from exc
                    added += new
                    unchanged += old
                    processed += 1
            print(f"Committed {processed:,}/{len(paths):,} files | "
                  f"new graphs={added:,} | unchanged={unchanged:,}", flush=True)
        check = connection.execute("PRAGMA quick_check").fetchall()
        if check != [("ok",)]:
            raise ValueError(f"SQLite integrity check failed: {check}")
        documents, graphs = connection.execute(
            "SELECT COUNT(DISTINCT doc_id), COUNT(*) FROM graphs WHERE split=?", (split,)
        ).fetchone()
    finally:
        connection.close()
        if archive is not None:
            archive.close()
    result = dict(output=str(output), split=split, input_files=processed,
                  added_graphs=added, unchanged_graphs=unchanged,
                  stored_documents=documents, stored_graphs=graphs,
                  seconds=round(time.monotonic() - started, 2),
                  database_bytes=output.stat().st_size)
    print(json.dumps(result, indent=2), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("/kaggle/working/train.sqlite"))
    parser.add_argument("--split", choices=("train", "dev", "test"), default="train")
    parser.add_argument("--commit-every", type=int, default=100, help="documents per transaction")
    args = parser.parse_args()
    convert(args.input_dir, args.output, args.split, args.commit_every)


if __name__ == "__main__":
    main()
