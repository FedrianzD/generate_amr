"""Compare Liputan6 extractive-oracle indices with SMATCH graph statistics.

For every document-level adjacency matrix, this script reads the document's
``extractive_summary`` indices and reports:

* every sentence-index pair tied for the highest off-diagonal SMATCH score;
* whether a strongest pair contains one or two extractive-oracle sentences;
* every sentence tied for the highest weighted degree (matrix row sum); and
* whether a highest-degree sentence is an extractive-oracle sentence.

The sentence indices are recovered from extracted AMR filenames when available,
with the AMR ZIP archive retained as a fallback. This avoids assuming that
matrix row ``i`` is always source sentence ``i``.

Example::

    conda run --name generate_amr python \
        scripts/liputan6/check_extractive_vs_smatch.py --split train

    conda run --name generate_amr python \
        scripts/liputan6/check_extractive_vs_smatch.py \
        --split both --output data/liputan6/processed/extractive_vs_smatch.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import zipfile
from pathlib import Path

import numpy as np

from build_smatch_adjacency import document_sort_key, index_archive, index_directory


PROJECT_ROOT = Path(__file__).resolve().parents[2]
LIPUTAN6_DATA_DIR = PROJECT_ROOT / "data" / "liputan6"
DEFAULT_SOURCE_DIR = (
    LIPUTAN6_DATA_DIR
    / "source"
    / "Liputan6"
    / "liputan6"
    / "liputan6_data"
    / "canonical"
)

OUTPUT_FIELDS = [
    "split",
    "doc_id",
    "status",
    "error",
    "num_sentences",
    "sentence_indices",
    "extractive_summary_indices",
    "max_smatch_score",
    "max_smatch_pairs",
    "max_smatch_pair_count",
    "max_pair_any_endpoint_in_ground_truth",
    "max_pair_both_endpoints_in_ground_truth",
    "max_weighted_degree",
    "max_weighted_degree_sentence_indices",
    "max_degree_any_sentence_in_ground_truth",
]


def as_json(value: object) -> str:
    """Serialize lists compactly so they remain unambiguous inside the CSV."""
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def load_ground_truth(source_path: Path) -> tuple[list[int], int]:
    """Return zero-based extractive indices and source article sentence count."""
    with source_path.open("r", encoding="utf-8") as source_file:
        document = json.load(source_file)

    if "extractive_summary" not in document:
        raise ValueError("source JSON has no extractive_summary field")
    if "clean_article" not in document:
        raise ValueError("source JSON has no clean_article field")

    indices = document["extractive_summary"]
    if not isinstance(indices, list) or not all(
        isinstance(index, int) and not isinstance(index, bool) for index in indices
    ):
        raise ValueError("extractive_summary must be a list of integer indices")

    article_size = len(document["clean_article"])
    invalid = [index for index in indices if index < 0 or index >= article_size]
    if invalid:
        raise ValueError(
            f"extractive_summary contains out-of-range indices: {invalid} "
            f"for {article_size} source sentences"
        )
    return indices, article_size


def validate_adjacency(matrix: np.ndarray, expected_size: int) -> None:
    """Fail with a useful message if an adjacency matrix cannot be compared."""
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"adjacency matrix must be square, found shape {matrix.shape}")
    if matrix.shape[0] != expected_size:
        raise ValueError(
            f"matrix has {matrix.shape[0]} rows but archive has "
            f"{expected_size} sentence graphs"
        )
    if not np.isfinite(matrix).all():
        raise ValueError("adjacency matrix contains NaN or infinite values")
    if not np.allclose(matrix, matrix.T, rtol=1e-5, atol=1e-7):
        raise ValueError("adjacency matrix is not symmetric")


def strongest_pairs(
    matrix: np.ndarray,
    sentence_indices: list[int],
    tolerance: float,
) -> tuple[float | None, list[list[int]]]:
    """Return the maximum off-diagonal value and all pairs tied for it."""
    size = matrix.shape[0]
    if size < 2:
        return None, []

    first_rows, second_rows = np.triu_indices(size, k=1)
    scores = matrix[first_rows, second_rows]
    maximum = float(scores.max())
    tied_positions = np.flatnonzero(
        np.isclose(scores, maximum, rtol=0.0, atol=tolerance)
    )
    pairs = [
        [
            sentence_indices[int(first_rows[position])],
            sentence_indices[int(second_rows[position])],
        ]
        for position in tied_positions
    ]
    return maximum, pairs


def highest_weighted_degree(
    matrix: np.ndarray,
    sentence_indices: list[int],
    tolerance: float,
) -> tuple[float | None, list[int]]:
    """Return the largest row sum and all sentence indices tied for it."""
    if matrix.shape[0] == 0:
        return None, []

    degrees = matrix.sum(axis=1, dtype=np.float64)
    maximum = float(degrees.max())
    tied_rows = np.flatnonzero(
        np.isclose(degrees, maximum, rtol=0.0, atol=tolerance)
    )
    return maximum, [sentence_indices[int(row)] for row in tied_rows]


def analyze_document(
    matrix_path: Path,
    source_path: Path,
    archive_members: list[tuple[int, str | Path]],
    split: str,
    tolerance: float,
) -> dict[str, object]:
    """Build one output row for a document, raising on invalid alignment."""
    doc_id = matrix_path.stem
    sentence_indices = [sentence_index for sentence_index, _ in archive_members]
    if len(sentence_indices) != len(set(sentence_indices)):
        raise ValueError("archive contains duplicate sentence indices")

    matrix = np.load(matrix_path, allow_pickle=False)
    validate_adjacency(matrix, len(sentence_indices))
    ground_truth, article_size = load_ground_truth(source_path)

    invalid_archive_indices = [
        index for index in sentence_indices if index < 0 or index >= article_size
    ]
    if invalid_archive_indices:
        raise ValueError(
            "archive contains sentence indices outside clean_article: "
            f"{invalid_archive_indices}"
        )

    maximum_score, pairs = strongest_pairs(matrix, sentence_indices, tolerance)
    maximum_degree, degree_indices = highest_weighted_degree(
        matrix, sentence_indices, tolerance
    )
    ground_truth_set = set(ground_truth)

    any_endpoint = any(
        first in ground_truth_set or second in ground_truth_set
        for first, second in pairs
    )
    both_endpoints = any(
        first in ground_truth_set and second in ground_truth_set
        for first, second in pairs
    )
    any_degree_sentence = any(
        index in ground_truth_set for index in degree_indices
    )

    return {
        "split": split,
        "doc_id": doc_id,
        "status": "ok",
        "error": "",
        "num_sentences": len(sentence_indices),
        "sentence_indices": as_json(sentence_indices),
        "extractive_summary_indices": as_json(ground_truth),
        "max_smatch_score": "" if maximum_score is None else maximum_score,
        "max_smatch_pairs": as_json(pairs),
        "max_smatch_pair_count": len(pairs),
        "max_pair_any_endpoint_in_ground_truth": any_endpoint,
        "max_pair_both_endpoints_in_ground_truth": both_endpoints,
        "max_weighted_degree": "" if maximum_degree is None else maximum_degree,
        "max_weighted_degree_sentence_indices": as_json(degree_indices),
        "max_degree_any_sentence_in_ground_truth": any_degree_sentence,
    }


def error_row(split: str, doc_id: str, message: str) -> dict[str, object]:
    """Return a CSV-compatible row for a document that could not be checked."""
    row: dict[str, object] = {field: "" for field in OUTPUT_FIELDS}
    row.update(
        {
            "split": split,
            "doc_id": doc_id,
            "status": "error",
            "error": message,
        }
    )
    return row


def analyze_split(
    split: str,
    adjacency_root: Path,
    source_root: Path,
    amr_root: Path,
    archive_root: Path,
    tolerance: float,
    max_documents: int | None,
) -> list[dict[str, object]]:
    """Analyze all available matrices for one split."""
    matrix_dir = adjacency_root / split
    amr_dir = amr_root / split
    archive_path = archive_root / f"{split}.zip"
    source_dir = source_root / split

    if not matrix_dir.is_dir():
        raise FileNotFoundError(f"missing adjacency directory: {matrix_dir}")
    if not source_dir.is_dir():
        raise FileNotFoundError(f"missing source directory: {source_dir}")

    matrix_paths = sorted(
        matrix_dir.glob("*.npy"), key=lambda path: document_sort_key(path.stem)
    )
    if max_documents is not None:
        matrix_paths = matrix_paths[:max_documents]

    has_extracted_graphs = amr_dir.is_dir() and next(
        amr_dir.rglob("*.txt"), None
    ) is not None
    if has_extracted_graphs:
        amr_index = index_directory(amr_dir)
        source_description = str(amr_dir)
    elif archive_path.is_file():
        with zipfile.ZipFile(archive_path) as archive:
            amr_index = index_archive(archive)
        source_description = str(archive_path)
    else:
        raise FileNotFoundError(
            f"missing AMRs: neither extracted directory {amr_dir} nor "
            f"archive {archive_path} is available"
        )
    print(f"{split}: sentence-index mapping from {source_description}")

    rows = []
    for matrix_path in matrix_paths:
        doc_id = matrix_path.stem
        try:
            if doc_id not in amr_index:
                raise ValueError("document is absent from the AMR inputs")
            source_path = source_dir / f"{doc_id}.json"
            if not source_path.is_file():
                raise FileNotFoundError(f"missing source JSON: {source_path}")
            row = analyze_document(
                matrix_path=matrix_path,
                source_path=source_path,
                archive_members=amr_index[doc_id],
                split=split,
                tolerance=tolerance,
            )
        except Exception as exc:
            row = error_row(split, doc_id, str(exc))
        rows.append(row)
    return rows


def print_summary(rows: list[dict[str, object]], output_path: Path) -> None:
    """Print compact aggregate results after writing the detailed CSV."""
    valid = [row for row in rows if row["status"] == "ok"]
    errors = len(rows) - len(valid)
    pair_any = sum(
        bool(row["max_pair_any_endpoint_in_ground_truth"]) for row in valid
    )
    pair_both = sum(
        bool(row["max_pair_both_endpoints_in_ground_truth"]) for row in valid
    )
    degree_any = sum(
        bool(row["max_degree_any_sentence_in_ground_truth"]) for row in valid
    )

    print(f"Checked {len(rows):,} documents: {len(valid):,} OK, {errors:,} errors")
    if valid:
        denominator = len(valid)
        print(
            "Strongest pair touches >=1 ground-truth sentence: "
            f"{pair_any:,}/{denominator:,} ({pair_any / denominator:.1%})"
        )
        print(
            "Strongest pair has 2 ground-truth sentences: "
            f"{pair_both:,}/{denominator:,} ({pair_both / denominator:.1%})"
        )
        print(
            "Highest-degree sentence is ground truth: "
            f"{degree_any:,}/{denominator:,} ({degree_any / denominator:.1%})"
        )
    print(f"Detailed CSV: {output_path.resolve()}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--adjacency-root",
        type=Path,
        default=LIPUTAN6_DATA_DIR / "adjacency_matrices",
        help="root containing train/dev/test .npy matrix directories",
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        default=DEFAULT_SOURCE_DIR,
        help="Liputan6 canonical root containing split JSON directories",
    )
    parser.add_argument(
        "--amr-root",
        type=Path,
        default=LIPUTAN6_DATA_DIR / "amr_graphs" / "extracted",
        help="root containing extracted train/dev/test .txt AMR directories",
    )
    parser.add_argument(
        "--archive-root",
        type=Path,
        default=LIPUTAN6_DATA_DIR / "amr_graphs" / "archives",
        help="ZIP fallback directory containing split AMR archives",
    )
    parser.add_argument(
        "--split",
        choices=("train", "dev", "test", "both", "all"),
        default="train",
        help="split to analyze; both means train+test, all includes dev",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=LIPUTAN6_DATA_DIR / "processed" / "extractive_vs_smatch.csv",
        help="destination CSV",
    )
    parser.add_argument(
        "--tie-tolerance",
        type=float,
        default=1e-8,
        help="absolute tolerance for retaining tied maximum scores",
    )
    parser.add_argument(
        "--max-documents",
        type=int,
        help="analyze only the first N matrices per selected split",
    )
    args = parser.parse_args()
    if args.tie_tolerance < 0:
        parser.error("--tie-tolerance must be non-negative")
    if args.max_documents is not None and args.max_documents < 1:
        parser.error("--max-documents must be at least 1")
    return args


def main() -> None:
    args = parse_args()
    if args.split == "both":
        splits = ("train", "test")
    elif args.split == "all":
        splits = ("train", "dev", "test")
    else:
        splits = (args.split,)
    rows: list[dict[str, object]] = []
    for split in splits:
        rows.extend(
            analyze_split(
                split=split,
                adjacency_root=args.adjacency_root,
                source_root=args.source_root,
                amr_root=args.amr_root,
                archive_root=args.archive_root,
                tolerance=args.tie_tolerance,
                max_documents=args.max_documents,
            )
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print_summary(rows, args.output)


if __name__ == "__main__":
    main()
