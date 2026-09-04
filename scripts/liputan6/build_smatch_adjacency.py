"""Build document-level SMATCH adjacency matrices for Liputan 6 AMR output.

The Liputan 6 parser writes one AMR graph per sentence, named
``{doc_id}_{sent_idx}.txt``, and packages the files below an ``amr_graphs/``
directory in a ZIP archive. This script groups those sentence files by
document, orders them by ``sent_idx``, and saves one float32 ``n x n`` matrix
per document. Entry ``[i, j]`` is the SMATCH F-score between sentences i and
j. The diagonal is zero to preserve the convention used by the original
XLSum script (and to avoid self-loops unless the GCN adds them explicitly).

The ZIP files are read directly; they do not need to be extracted.

Examples::

    python scripts/liputan6/build_smatch_adjacency.py

    python scripts/liputan6/build_smatch_adjacency.py \
        --train-zip path/to/train.zip --test-zip path/to/test.zip \
        --output data/liputan6/adjacency_matrix_amr --max-documents 10
"""

from __future__ import annotations

import argparse
import re
import zipfile
from collections import defaultdict
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--train-zip",
        type=Path,
        default=LIPUTAN6_DATA_DIR / "amr_graphs" / "archives" / "train.zip",
        help="ZIP containing train sentence AMRs",
    )
    parser.add_argument(
        "--test-zip",
        type=Path,
        default=LIPUTAN6_DATA_DIR / "amr_graphs" / "archives" / "test.zip",
        help="ZIP containing test sentence AMRs",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=LIPUTAN6_DATA_DIR / "adjacency_matrices",
        help="output root; train/ and test/ are created below it",
    )
    parser.add_argument(
        "--split",
        choices=("train", "test", "both"),
        default="both",
        help="split to process (default: both)",
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="replace existing .npy files"
    )
    parser.add_argument(
        "--max-documents",
        type=int,
        help="process only the first N documents per split (useful for testing)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.max_documents is not None and args.max_documents < 1:
        raise SystemExit("--max-documents must be at least 1")

    archives_by_split = {"train": args.train_zip, "test": args.test_zip}
    selected_splits = ("train", "test") if args.split == "both" else (args.split,)
    archives = [(split, archives_by_split[split]) for split in selected_splits]
    missing = [str(path) for _, path in archives if not path.is_file()]
    if missing:
        raise SystemExit("Missing input archive(s): " + ", ".join(missing))

    summaries = {}
    for split, zip_path in archives:
        summaries[split] = process_archive(
            zip_path=zip_path,
            split=split,
            output_root=args.output,
            overwrite=args.overwrite,
            max_documents=args.max_documents,
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
