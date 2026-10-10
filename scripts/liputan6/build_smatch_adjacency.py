"""Build document-level SMATCH adjacency matrices for Liputan 6 AMR output.

The Liputan 6 parser writes one AMR graph per sentence, named
``{doc_id}_{sent_idx}.txt``. This script reads either extracted ``.txt`` files
or a ZIP archive, groups sentences by document, orders them by ``sent_idx``,
and saves one float32 ``n x n`` matrix per document. Entry ``[i, j]`` is the
SMATCH F-score between sentences i and j. The diagonal is zero to preserve the
convention used by the original XLSum script.

Examples::

    conda run --name generate_amr python scripts/liputan6/build_smatch_adjacency.py

    conda run --name generate_amr python scripts/liputan6/build_smatch_adjacency.py \
        --split dev --dev-dir path/to/extracted/dev \
        --output data/liputan6/adjacency_matrices --max-documents 10
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import sqlite3
import time
import zipfile
from collections import defaultdict
from contextlib import closing
from importlib.metadata import version
from pathlib import Path, PurePosixPath

import numpy as np
import penman
import smatch
from tqdm import tqdm


SENTENCE_FILE_RE = re.compile(r"^(?P<doc_id>.+)_(?P<sent_idx>\d+)\.txt$")
PROJECT_ROOT = Path(__file__).resolve().parents[2]
LIPUTAN6_DATA_DIR = PROJECT_ROOT / "data" / "liputan6"


def graph_to_oneline(graph: penman.Graph) -> str:
    """Serialize a PENMAN graph on one line without metadata comments."""
    formatted = penman.format(penman.configure(graph))
    return " ".join(
        line.strip()
        for line in formatted.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    )


def parse_one_graph(content: str, source: str) -> str:
    """Parse exactly one AMR graph and return its SMATCH-ready representation."""
    graphs = penman.loads(content)
    if len(graphs) != 1:
        raise ValueError(f"{source}: expected 1 AMR graph, found {len(graphs)}")
    return graph_to_oneline(graphs[0])


def smatch_f_score(first: str, second: str) -> float:
    """Compute a pairwise SMATCH F-score and clear smatch's global cache."""
    try:
        best_match, first_num, second_num = smatch.get_amr_match(
            first, second, sent_num=1
        )
        _, _, f_score = smatch.compute_f(best_match, first_num, second_num)
        return float(f_score)
    finally:
        # smatch keeps pair-specific state in this module-level dictionary.
        smatch.match_triple_dict.clear()


def smatch_details(first: str, second: str) -> dict:
    """Return the best alignment, matching triples, and score components."""
    first_amr = smatch.amr.AMR.parse_AMR_line(first)
    second_amr = smatch.amr.AMR.parse_AMR_line(second)
    first_amr.rename_node("a")
    second_amr.rename_node("b")
    first_groups = first_amr.get_triples()
    second_groups = second_amr.get_triples()

    # SMATCH uses randomized hill climbing. Keep the best of several complete
    # searches so the explanatory table does not show a weaker local optimum.
    mapping = None
    match_count = -1
    try:
        for _ in range(20):
            smatch.match_triple_dict.clear()
            candidate_mapping, candidate_count = smatch.get_best_match(
                *first_groups, *second_groups, "a", "b"
            )
            if candidate_count > match_count:
                mapping = candidate_mapping
                match_count = candidate_count
    finally:
        smatch.match_triple_dict.clear()

    variable_mapping = {
        f"a{first_index}": f"b{second_index}"
        for first_index, second_index in enumerate(mapping)
        if second_index >= 0
    }
    matching_triples = []
    for category, first_triples, second_triples in zip(
        ("instance", "attribute", "relation"), first_groups, second_groups
    ):
        second_set = set(second_triples)
        for relation, source, target in first_triples:
            mapped_source = variable_mapping.get(source)
            if mapped_source is None:
                continue
            mapped_target = (
                variable_mapping.get(target) if category == "relation" else target
            )
            if mapped_target is None:
                continue
            first_triple = (relation, source, target)
            second_triple = (relation, mapped_source, mapped_target)
            if second_triple in second_set:
                matching_triples.append(
                    {
                        "category": category,
                        "first_triple": first_triple,
                        "second_triple": second_triple,
                    }
                )

    first_count = sum(len(group) for group in first_groups)
    second_count = sum(len(group) for group in second_groups)
    precision, recall, f_score = smatch.compute_f(
        match_count, first_count, second_count
    )
    return {
        "match_count": match_count,
        "first_count": first_count,
        "second_count": second_count,
        "precision": precision,
        "recall": recall,
        "f_score": f_score,
        "variable_mapping": variable_mapping,
        "matching_triples": matching_triples,
    }


def compute_smatch_adjacency(amr_strings: list[str]) -> np.ndarray:
    """Return the symmetric pairwise SMATCH matrix for ordered AMR strings."""
    count = len(amr_strings)
    adjacency = np.zeros((count, count), dtype=np.float32)
    for i in range(count):
        for j in range(i + 1, count):
            score = smatch_f_score(amr_strings[i], amr_strings[j])
            adjacency[i, j] = score
            adjacency[j, i] = score
    return adjacency


def process_sqlite(database, split, ids_csv, source_root, output_root, merge_path=None):
    """Build only requested documents with exact canonical index/ID coverage."""
    from npy_to_npz import load_matrix, merge_npz

    with Path(ids_csv).open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    ids = [row["id"] for row in rows]
    if not ids or len(ids) != len(set(ids)):
        raise ValueError("Document CSV must contain unique IDs in an id column")
    if any(not i or any(c in i for c in ("/", "\\", ":")) or i in {".", ".."} for i in ids):
        raise ValueError("Unsafe document ID in CSV")
    folder = Path(output_root) / split
    folder.mkdir(parents=True, exist_ok=True)
    additions, details = {}, []
    started = time.monotonic()
    with closing(sqlite3.connect(Path(database).resolve().as_uri() + "?mode=ro", uri=True)) as db:
        if db.execute("SELECT value FROM storage_metadata WHERE key='schema_version'").fetchone() != ("amr-documents-v1",):
            raise ValueError("Unsupported AMR SQLite schema")
        for number, doc_id in enumerate(ids, 1):
            source = json.loads((Path(source_root) / split / f"{doc_id}.json").read_text(encoding="utf-8-sig"))
            article = source["clean_article"]
            if not isinstance(article, list) or not article:
                raise ValueError(f"Invalid clean_article: {doc_id}")
            count = len(article)
            stored = db.execute(
                "SELECT sent_idx, sentence_id, amr_penman FROM graphs "
                "WHERE split=? AND doc_id=? ORDER BY sent_idx", (split, doc_id)).fetchall()
            if [row[0] for row in stored] != list(range(count)):
                raise ValueError(f"Incomplete or misaligned AMR indices: {doc_id}")
            strings = []
            for index, sentence_id, content in stored:
                graphs = penman.loads(content)
                if (sentence_id != f"{doc_id}_{index}" or len(graphs) != 1
                        or graphs[0].metadata.get("id") != sentence_id
                        or not graphs[0].triples or graphs[0].top is None):
                    raise ValueError(f"Invalid stored AMR: {doc_id}_{index}")
                strings.append(graph_to_oneline(graphs[0]))
            path = folder / f"{doc_id}.npy"
            reused = path.exists()
            if reused:
                matrix = load_matrix(path)
                if matrix.shape != (count, count):
                    raise ValueError(f"Existing matrix has wrong shape: {path}")
            else:
                matrix = compute_smatch_adjacency(strings)
                temporary = path.with_suffix(".npy.tmp")
                try:
                    with temporary.open("wb") as handle:
                        np.save(handle, matrix, allow_pickle=False)
                    checked = load_matrix(temporary)
                    if checked.shape != (count, count) or not np.array_equal(checked, matrix):
                        raise ValueError(f"Saved matrix mismatch: {doc_id}")
                    os.replace(temporary, path)
                finally:
                    temporary.unlink(missing_ok=True)
            additions[doc_id] = path
            details.append(dict(id=doc_id, sentences=count, reused=reused,
                                npy_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
            print(f"SQLITE SMATCH {number}/{len(ids)} | id={doc_id} sentences={count} "
                  f"reused={reused} elapsed={time.monotonic()-started:.1f}s", flush=True)
    report = dict(split=split, database=str(database), ids_csv=str(ids_csv),
                  ids_csv_sha256=hashlib.sha256(Path(ids_csv).read_bytes()).hexdigest(),
                  smatch_version=version("smatch"), iteration_num=smatch.iteration_num,
                  search="existing randomized SMATCH search; no deterministic seed guarantee",
                  dtype="float32", diagonal=0, sentence_order="canonical sent_idx 0..n-1",
                  documents=details)
    if merge_path is not None:
        report["merge"] = merge_npz(merge_path, additions)
    report_path = Path(output_root) / f"{split}_sqlite_smatch_generation.json"
    temporary = report_path.with_suffix(".json.tmp")
    try:
        temporary.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        os.replace(temporary, report_path)
    finally:
        temporary.unlink(missing_ok=True)
    return report


def index_archive(archive: zipfile.ZipFile) -> dict[str, list[tuple[int, str]]]:
    """Return ``doc_id -> [(sent_idx, member_name), ...]`` for a parser ZIP."""
    grouped: dict[str, list[tuple[int, str]]] = defaultdict(list)
    seen: set[tuple[str, int]] = set()

    for info in archive.infolist():
        if info.is_dir():
            continue
        basename = PurePosixPath(info.filename).name
        match = SENTENCE_FILE_RE.fullmatch(basename)
        if not match:
            continue

        doc_id = match.group("doc_id")
        sent_idx = int(match.group("sent_idx"))
        key = (doc_id, sent_idx)
        if key in seen:
            raise ValueError(
                f"duplicate sentence {doc_id}_{sent_idx} in {archive.filename}"
            )
        seen.add(key)
        grouped[doc_id].append((sent_idx, info.filename))

    for members in grouped.values():
        members.sort(key=lambda item: item[0])
    return dict(grouped)


def index_directory(directory: Path) -> dict[str, list[tuple[int, Path]]]:
    """Return ``doc_id -> [(sent_idx, path), ...]`` for extracted AMR files."""
    grouped: dict[str, list[tuple[int, Path]]] = defaultdict(list)
    seen: set[tuple[str, int]] = set()

    for graph_path in directory.rglob("*.txt"):
        match = SENTENCE_FILE_RE.fullmatch(graph_path.name)
        if not match:
            continue

        doc_id = match.group("doc_id")
        sent_idx = int(match.group("sent_idx"))
        key = (doc_id, sent_idx)
        if key in seen:
            raise ValueError(
                f"duplicate sentence {doc_id}_{sent_idx} below {directory}"
            )
        seen.add(key)
        grouped[doc_id].append((sent_idx, graph_path))

    for members in grouped.values():
        members.sort(key=lambda item: item[0])
    return dict(grouped)


def document_sort_key(value: str) -> tuple[bool, int | str]:
    """Sort numeric document IDs numerically and other IDs lexically."""
    return (not value.isdigit(), int(value) if value.isdigit() else value)


def process_archive(
    zip_path: Path,
    split: str,
    output_root: Path,
    overwrite: bool = False,
    max_documents: int | None = None,
) -> dict[str, int]:
    """Process one split archive and return summary counts."""
    output_dir = output_root / split
    output_dir.mkdir(parents=True, exist_ok=True)
    counts = {
        "documents": 0,
        "sentences": 0,
        "written": 0,
        "skipped": 0,
        "errors": 0,
    }
    error_details: list[tuple[str, str]] = []

    with zipfile.ZipFile(zip_path) as archive:
        grouped = index_archive(archive)
        doc_ids = sorted(grouped, key=document_sort_key)
        if max_documents is not None:
            doc_ids = doc_ids[:max_documents]

        counts["documents"] = len(doc_ids)
        counts["sentences"] = sum(len(grouped[doc_id]) for doc_id in doc_ids)
        print(
            f"{split}: {counts['documents']:,} documents / "
            f"{counts['sentences']:,} sentence graphs in {zip_path}"
        )

        for doc_id in tqdm(doc_ids, desc=f"SMATCH {split}", unit="doc"):
            output_path = output_dir / f"{doc_id}.npy"
            if output_path.exists() and not overwrite:
                counts["skipped"] += 1
                continue

            try:
                amr_strings = []
                for _, member_name in grouped[doc_id]:
                    content = archive.read(member_name).decode("utf-8-sig")
                    amr_strings.append(parse_one_graph(content, member_name))
                np.save(output_path, compute_smatch_adjacency(amr_strings))
                counts["written"] += 1
            except Exception as exc:  # one bad document should not lose a long run
                counts["errors"] += 1
                error_details.append((doc_id, str(exc)))

    if error_details:
        print(f"{split}: first errors:")
        for doc_id, message in error_details[:10]:
            print(f"  {doc_id}: {message}")
    return counts


def process_directory(
    input_dir: Path,
    split: str,
    output_root: Path,
    overwrite: bool = False,
    max_documents: int | None = None,
) -> dict[str, int]:
    """Process one directory of extracted sentence AMRs."""
    output_dir = output_root / split
    output_dir.mkdir(parents=True, exist_ok=True)
    grouped = index_directory(input_dir)
    doc_ids = sorted(grouped, key=document_sort_key)
    if max_documents is not None:
        doc_ids = doc_ids[:max_documents]

    counts = {
        "documents": len(doc_ids),
        "sentences": sum(len(grouped[doc_id]) for doc_id in doc_ids),
        "written": 0,
        "skipped": 0,
        "errors": 0,
    }
    error_details: list[tuple[str, str]] = []
    print(
        f"{split}: {counts['documents']:,} documents / "
        f"{counts['sentences']:,} sentence graphs below {input_dir}"
    )

    for doc_id in tqdm(doc_ids, desc=f"SMATCH {split}", unit="doc"):
        output_path = output_dir / f"{doc_id}.npy"
        if output_path.exists() and not overwrite:
            counts["skipped"] += 1
            continue

        try:
            amr_strings = []
            for _, graph_path in grouped[doc_id]:
                content = graph_path.read_text(encoding="utf-8-sig")
                amr_strings.append(parse_one_graph(content, str(graph_path)))
            np.save(output_path, compute_smatch_adjacency(amr_strings))
            counts["written"] += 1
        except Exception as exc:
            counts["errors"] += 1
            error_details.append((doc_id, str(exc)))

    if error_details:
        print(f"{split}: first errors:")
        for doc_id, message in error_details[:10]:
            print(f"  {doc_id}: {message}")
    return counts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--train-zip",
        type=Path,
        default=LIPUTAN6_DATA_DIR / "amr_graphs" / "archives" / "train.zip",
        help="ZIP containing train sentence AMRs",
    )
    parser.add_argument(
        "--dev-zip",
        type=Path,
        default=LIPUTAN6_DATA_DIR / "amr_graphs" / "archives" / "dev.zip",
        help="ZIP containing dev sentence AMRs (fallback when --dev-dir is absent)",
    )
    parser.add_argument(
        "--test-zip",
        type=Path,
        default=LIPUTAN6_DATA_DIR / "amr_graphs" / "archives" / "test.zip",
        help="ZIP containing test sentence AMRs",
    )
    parser.add_argument(
        "--train-dir",
        type=Path,
        default=LIPUTAN6_DATA_DIR / "amr_graphs" / "extracted" / "train",
        help="directory containing extracted train .txt AMRs (preferred over ZIP)",
    )
    parser.add_argument(
        "--dev-dir",
        type=Path,
        default=LIPUTAN6_DATA_DIR / "amr_graphs" / "extracted" / "dev",
        help="directory containing extracted dev .txt AMRs (preferred over ZIP)",
    )
    parser.add_argument(
        "--test-dir",
        type=Path,
        default=LIPUTAN6_DATA_DIR / "amr_graphs" / "extracted" / "test",
        help="directory containing extracted test .txt AMRs (preferred over ZIP)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=LIPUTAN6_DATA_DIR / "adjacency_matrices",
        help="output root; one directory is created below it per split",
    )
    parser.add_argument(
        "--split",
        choices=("train", "dev", "test", "both", "all"),
        default="both",
        help="split to process; both means train+test, all includes dev",
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="replace existing .npy files"
    )
    parser.add_argument(
        "--max-documents",
        type=int,
        help="process only the first N documents per split (useful for testing)",
    )
    parser.add_argument("--sqlite", type=Path, help="read targeted document AMRs from SQLite")
    parser.add_argument("--document-ids-csv", type=Path, help="requested IDs in an id column")
    parser.add_argument("--source-root", type=Path,
                        default=LIPUTAN6_DATA_DIR / "source/Liputan6/liputan6/liputan6_data/canonical",
                        help="canonical split folders for exact sentence coverage checks")
    parser.add_argument("--merge-npz", type=Path, help="atomically merge generated matrices into this NPZ")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.sqlite is not None:
        if args.split not in {"train", "dev", "test"} or args.document_ids_csv is None:
            raise SystemExit("--sqlite requires one explicit split and --document-ids-csv")
        if args.overwrite or args.max_documents is not None:
            raise SystemExit("SQLite targeted mode reuses existing valid matrices; no overwrite or max-documents")
        process_sqlite(args.sqlite, args.split, args.document_ids_csv,
                       args.source_root, args.output, args.merge_npz)
        return
    if args.document_ids_csv is not None or args.merge_npz is not None:
        raise SystemExit("--document-ids-csv and --merge-npz require --sqlite")
    if args.max_documents is not None and args.max_documents < 1:
        raise SystemExit("--max-documents must be at least 1")

    directories_by_split = {
        "train": args.train_dir,
        "dev": args.dev_dir,
        "test": args.test_dir,
    }
    archives_by_split = {
        "train": args.train_zip,
        "dev": args.dev_zip,
        "test": args.test_zip,
    }
    if args.split == "both":
        selected_splits = ("train", "test")
    elif args.split == "all":
        selected_splits = ("train", "dev", "test")
    else:
        selected_splits = (args.split,)

    summaries = {}
    for split in selected_splits:
        input_dir = directories_by_split[split]
        zip_path = archives_by_split[split]
        has_extracted_graphs = input_dir.is_dir() and next(
            input_dir.rglob("*.txt"), None
        ) is not None
        if has_extracted_graphs:
            summaries[split] = process_directory(
                input_dir=input_dir,
                split=split,
                output_root=args.output,
                overwrite=args.overwrite,
                max_documents=args.max_documents,
            )
        elif zip_path.is_file():
            summaries[split] = process_archive(
                zip_path=zip_path,
                split=split,
                output_root=args.output,
                overwrite=args.overwrite,
                max_documents=args.max_documents,
            )
        else:
            raise SystemExit(
                f"Missing {split} AMRs: neither extracted directory "
                f"{input_dir} nor archive {zip_path} is available"
            )

    print("\nSUMMARY")
    for split, counts in summaries.items():
        print(
            f"  {split:5s}: {counts['written']:,} written, "
            f"{counts['skipped']:,} skipped, {counts['errors']:,} errors "
            f"({counts['documents']:,} docs / {counts['sentences']:,} sentences)"
        )
    print(f"  output: {args.output.resolve()}")


if __name__ == "__main__":
    main()
