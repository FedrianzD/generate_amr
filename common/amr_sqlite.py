"""SQLite storage shared by parser notebooks; compatible with documents_to_sqlite.

The notebooks embed this module to remain usable with older code-module bundles.
Connections are short-lived: each generation batch is committed and closed.
"""

import os
import re
import sqlite3
import time
from contextlib import closing
from pathlib import Path

import penman
from penman.models.noop import NoOpModel


SCHEMA_VERSION = "amr-documents-v1"
STORAGE_MODEL = NoOpModel()
SENTENCE_ID_RE = re.compile(r"^(?P<doc_id>.+)_(?P<sent_idx>\d+)$")


def sentence_key(sentence_id):
    match = SENTENCE_ID_RE.fullmatch(str(sentence_id))
    if match is None:
        raise ValueError(f"Invalid sentence ID: {sentence_id!r}")
    return match.group("doc_id"), int(match.group("sent_idx"))


def validate_schema(connection):
    tables = {row[0] for row in connection.execute(
        "SELECT name FROM sqlite_master WHERE type='table'")}
    if not {"storage_metadata", "graphs"}.issubset(tables):
        raise ValueError("Not an AMR SQLite database")
    version = connection.execute(
        "SELECT value FROM storage_metadata WHERE key='schema_version'"
    ).fetchone()
    if version != (SCHEMA_VERSION,):
        raise ValueError(f"Unsupported AMR SQLite schema: {version}")
    columns = [row[1] for row in connection.execute("PRAGMA table_info(graphs)")]
    if columns != ["split", "doc_id", "sent_idx", "sentence_id", "amr_penman"]:
        raise ValueError("Unexpected graphs table columns")


def readonly_database(path):
    return sqlite3.connect(Path(path).resolve().as_uri() + "?mode=ro", uri=True, timeout=60)


def same_graph(left, right):
    return (left.top == right.top and left.triples == right.triples
            and left.metadata == right.metadata)


class AMRSQLiteStore:
    def __init__(self, path, split):
        self.path = Path(path)
        self.split = str(split)
        if self.split not in {"train", "dev", "test"}:
            raise ValueError("Select train, dev, or test explicitly")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with closing(self.connect()) as connection:
            tables = connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table'").fetchall()
            if tables:
                validate_schema(connection)
            else:
                with connection:
                    connection.execute("""CREATE TABLE storage_metadata (
                        key TEXT PRIMARY KEY, value TEXT NOT NULL)""")
                    connection.execute("INSERT INTO storage_metadata VALUES (?, ?)",
                                       ("schema_version", SCHEMA_VERSION))
                    connection.execute("""CREATE TABLE graphs (
                        split TEXT NOT NULL, doc_id TEXT NOT NULL,
                        sent_idx INTEGER NOT NULL CHECK(sent_idx >= 0),
                        sentence_id TEXT NOT NULL, amr_penman TEXT NOT NULL,
                        PRIMARY KEY (split, doc_id, sent_idx),
                        UNIQUE (split, sentence_id))""")

    def connect(self):
        connection = sqlite3.connect(str(self.path), timeout=60)
        try:
            connection.execute("PRAGMA journal_mode=DELETE")
            connection.execute("PRAGMA synchronous=FULL")
        except BaseException:
            connection.close()
            raise
        return connection

    def parsed_ids(self):
        started = time.monotonic()
        print(f"Loading resume IDs for {self.split}...", flush=True)
        with closing(readonly_database(self.path)) as connection:
            result = {row[0] for row in connection.execute(
                "SELECT sentence_id FROM graphs WHERE split=?", (self.split,))}
        print(f"Loaded {len(result):,} resume IDs in {time.monotonic() - started:.1f}s", flush=True)
        return result

    def write_graphs(self, graphs):
        """Validate and commit an entire batch; never overwrite conflicting rows."""
        rows = []
        for graph in graphs:
            sentence_id = str(graph.metadata.get("id", ""))
            doc_id, index = sentence_key(sentence_id)
            if not graph.triples or graph.top is None:
                raise ValueError(f"Empty AMR graph: {sentence_id}")
            serialized = penman.encode(graph, model=STORAGE_MODEL) + "\n"
            if not same_graph(graph, penman.decode(serialized, model=STORAGE_MODEL)):
                raise ValueError(f"PENMAN round-trip mismatch: {sentence_id}")
            rows.append((doc_id, index, sentence_id, serialized, graph))
        with closing(self.connect()) as connection, connection:
            for doc_id, index, sentence_id, serialized, graph in rows:
                existing = connection.execute(
                    "SELECT sentence_id, amr_penman FROM graphs "
                    "WHERE split=? AND doc_id=? AND sent_idx=?",
                    (self.split, doc_id, index),
                ).fetchone()
                if existing is not None:
                    if existing[0] != sentence_id or not same_graph(
                            graph, penman.decode(existing[1], model=STORAGE_MODEL)):
                        raise ValueError(f"Conflicting stored graph: {self.split}/{sentence_id}")
                else:
                    connection.execute("INSERT INTO graphs VALUES (?, ?, ?, ?, ?)",
                                       (self.split, doc_id, index, sentence_id, serialized))
        # Callers update their resume cache only after this transaction commits.
        return {row[2] for row in rows}

    def import_database(self, source, batch_size=1000):
        source = Path(source)
        if source.resolve() == self.path.resolve():
            return
        started = time.monotonic()
        print(f"Checking previous SQLite database: {source}", flush=True)
        with closing(readonly_database(source)) as connection:
            validate_schema(connection)
            if connection.execute("PRAGMA quick_check").fetchall() != [("ok",)]:
                raise ValueError(f"SQLite integrity check failed: {source}")
            print(f"Integrity check finished in {time.monotonic() - started:.1f}s; "
                  "merging new or differing rows...", flush=True)
            # Fetch existing text in batches before paying the PENMAN parsing cost.
            # Differing serializations still receive semantic conflict validation.
            cursor = connection.execute(
                "SELECT doc_id, sent_idx, sentence_id, amr_penman FROM graphs "
                "WHERE split=? ORDER BY doc_id, sent_idx", (self.split,))
            count = 0
            changed = 0
            while True:
                rows = cursor.fetchmany(batch_size)
                if not rows:
                    break
                existing = {}
                with closing(readonly_database(self.path)) as destination:
                    for offset in range(0, len(rows), 400):
                        ids = [row[2] for row in rows[offset:offset + 400]]
                        placeholders = ",".join("?" for _ in ids)
                        for doc_id, index, sentence_id, text in destination.execute(
                                "SELECT doc_id, sent_idx, sentence_id, amr_penman FROM graphs "
                                f"WHERE split=? AND sentence_id IN ({placeholders})",
                                [self.split, *ids]):
                            existing[sentence_id] = (doc_id, index, text)
                graphs = []
                for doc_id, index, sentence_id, text in rows:
                    if existing.get(sentence_id) == (doc_id, index, text):
                        continue
                    graph = penman.decode(text, model=STORAGE_MODEL)
                    if (sentence_key(sentence_id) != (doc_id, index)
                            or graph.metadata.get("id") != sentence_id):
                        raise ValueError(f"Misaligned database row: {sentence_id}")
                    graphs.append(graph)
                if graphs:
                    self.write_graphs(graphs)
                count += len(rows)
                changed += len(graphs)
                if count % 10_000 < len(rows):
                    print(f"Checked {count:,} rows; merged {changed:,} new or differing rows "
                          f"in {time.monotonic() - started:.1f}s", flush=True)
        print(f"SQLite merge finished in {time.monotonic() - started:.1f}s", flush=True)

    def import_directory(self, source, log_every=1000):
        source = Path(source)
        if not source.is_dir():
            raise FileNotFoundError(f"Previous graph folder missing: {source}")
        count = 0
        pending = []
        for path in sorted(source.rglob("*.txt")):
            graphs = penman.loads(path.read_text(encoding="utf-8-sig"), model=STORAGE_MODEL)
            if not graphs:
                raise ValueError(f"Empty AMR file: {path}")
            indices = set()
            for graph in graphs:
                sentence_id = graph.metadata.get("id", "")
                doc_id, index = sentence_key(sentence_id)
                if not (path.stem == doc_id or (len(graphs) == 1 and path.stem == sentence_id)):
                    raise ValueError(f"Graph ID does not match filename: {path}, {sentence_id}")
                if index in indices:
                    raise ValueError(f"Duplicate sentence index in {path}")
                indices.add(index)
            pending.extend(graphs)
            count += 1
            if count % log_every == 0:
                self.write_graphs(pending)
                pending = []
                print(f"Imported {count:,} previous text files", flush=True)
        if not count:
            raise ValueError(f"No .txt graphs found in {source}")
        self.write_graphs(pending)
        print(f"Imported {count:,} previous text files", flush=True)

    def document_indices(self):
        with closing(readonly_database(self.path)) as connection:
            result = {}
            for doc_id, index in connection.execute(
                    "SELECT doc_id, sent_idx FROM graphs WHERE split=?", (self.split,)):
                result.setdefault(doc_id, set()).add(index)
            return result

    def read_document(self, doc_id):
        with closing(readonly_database(self.path)) as connection:
            return [penman.decode(row[0], model=STORAGE_MODEL) for row in connection.execute(
                "SELECT amr_penman FROM graphs WHERE split=? AND doc_id=? ORDER BY sent_idx",
                (self.split, str(doc_id)))]


def prepare_sqlite_store(output, split, previous_db="", previous_dir=""):
    """Use a converter database or import legacy files, then resume by sentence."""
    output = Path(output)
    if previous_db and previous_dir:
        raise ValueError("Set only one of PREV_DB and PREV_DIR")
    if previous_db:
        source = Path(previous_db)
        if not source.is_file():
            raise FileNotFoundError(f"Previous SQLite database missing: {source}")
        # Fast path: consistent SQLite backup into a new working database.
        if not output.exists():
            started = time.monotonic()
            print(f"Checking previous SQLite database: {source}", flush=True)
            output.parent.mkdir(parents=True, exist_ok=True)
            temporary = output.with_name(output.name + ".importing")
            try:
                with closing(readonly_database(source)) as origin:
                    validate_schema(origin)
                    if origin.execute("PRAGMA quick_check").fetchall() != [("ok",)]:
                        raise ValueError(f"SQLite integrity check failed: {source}")
                    print(f"Integrity check finished in {time.monotonic() - started:.1f}s; "
                          "copying database...", flush=True)
                    last_report = [time.monotonic()]
                    def report_copy(status, remaining, total):
                        now = time.monotonic()
                        if remaining == 0 or now - last_report[0] >= 10:
                            print(f"SQLite copy: {total - remaining:,}/{total:,} pages "
                                  f"({now - started:.1f}s including integrity check)", flush=True)
                            last_report[0] = now
                    with closing(sqlite3.connect(str(temporary))) as target:
                        origin.backup(target, pages=4096, progress=report_copy)
                os.replace(temporary, output)
            finally:
                temporary.unlink(missing_ok=True)
            return AMRSQLiteStore(output, split)
    store = AMRSQLiteStore(output, split)
    if previous_db:
        store.import_database(previous_db)
    if previous_dir:
        store.import_directory(previous_dir)
    return store
