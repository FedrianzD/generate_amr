"""Pack completed adjacency matrices into one lossless compressed NPZ per split.

Default inputs: data/liputan6/adjacency_matrices/{dev,test}/**/{doc_id}.npy
Default outputs: data/liputan6/adjacency_matrices/{dev,test}.npz
Source NPY files are deleted only after a verified archive is published.
Use --keep-originals to retain them. Only one document is loaded at a time.
"""

import argparse
import hashlib
import io
import os
import shutil
import tempfile
import zipfile
from pathlib import Path

import numpy as np


DEFAULT_ROOT = Path(__file__).resolve().parents[2] / "data/liputan6/adjacency_matrices"


def index_split(folder):
    folder = Path(folder)
    if not folder.is_dir():
        raise FileNotFoundError(f"Split folder missing: {folder}")
    files = {}
    for path in sorted(folder.rglob("*.npy")):
        if path.stem in files:
            raise ValueError(f"Duplicate document ID {path.stem}: {files[path.stem]} and {path}")
        files[path.stem] = path
    if not files:
        raise ValueError(f"No .npy files found in {folder}")
    return files


def load_matrix(path):
    matrix = np.load(path, allow_pickle=False)
    if not isinstance(matrix, np.ndarray):
        matrix.close()
        raise ValueError(f"Expected a single NPY array: {path}")
    return validate_matrix(matrix, path)


def validate_matrix(matrix, label):
    path = label
    if (matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]
            or matrix.shape[0] == 0):
        raise ValueError(f"Expected nonempty square adjacency, got {matrix.shape}: {path}")
    if matrix.dtype.kind not in "fiu":
        raise ValueError(f"Expected real numeric adjacency, got {matrix.dtype}: {path}")
    if not np.isfinite(matrix).all():
        raise ValueError(f"Nonfinite adjacency: {path}")
    if np.any(matrix < 0) or np.any(matrix > 1):
        raise ValueError(f"SMATCH scores outside [0, 1]: {path}")
    if not np.allclose(matrix, matrix.T, rtol=0, atol=1e-6):
        raise ValueError(f"Asymmetric adjacency: {path}")
    if not np.allclose(np.diag(matrix), 0, rtol=0, atol=1e-6):
        raise ValueError(f"Expected zero adjacency diagonal: {path}")
    return matrix


def stream_hash(handle):
    digest = hashlib.sha256()
    for block in iter(lambda: handle.read(1024 * 1024), b""):
        digest.update(block)
    return digest.digest()


def delete_archived_sources(folder, files, output):
    """Delete only in-folder files whose complete bytes match the saved archive."""
    root = Path(folder).resolve(strict=True)
    hashes = {}
    with zipfile.ZipFile(output, "r") as archive:
        # Check every source before deleting any, including NPY headers.
        for doc_id, path in files.items():
            resolved = path.resolve(strict=True)
            if not resolved.is_relative_to(root) or resolved == root or path.is_symlink():
                raise ValueError(f"Refusing cleanup outside the input folder: {path}")
            with archive.open(f"{doc_id}.npy") as member, path.open("rb") as source:
                digest = stream_hash(member)
                if stream_hash(source) != digest:
                    raise ValueError(f"Source changed; archive retained and cleanup stopped: {path}")
            hashes[doc_id] = digest
    for doc_id, path in files.items():
        # Recheck immediately before unlinking in case input changed during cleanup.
        resolved = path.resolve(strict=True)
        if not resolved.is_relative_to(root) or path.is_symlink():
            raise ValueError(f"Source path changed; cleanup stopped: {path}")
        with path.open("rb") as source:
            if stream_hash(source) != hashes[doc_id]:
                raise ValueError(f"Source changed; archive retained and cleanup stopped: {path}")
        path.unlink()
    return len(files)


def convert_split(folder, output, overwrite=False, compression_level=6, log_every=1000,
                  keep_originals=False):
    """Write to a temporary archive, verify every array, then publish atomically."""
    if not 0 <= compression_level <= 9:
        raise ValueError("compression_level must be between 0 and 9")
    if log_every < 1:
        raise ValueError("log_every must be positive")
    files = index_split(folder)
    output = Path(output)
    if output.suffix.lower() != ".npz":
        raise ValueError("Output must end in .npz")
    if output.exists() and not overwrite:
        raise FileExistsError(f"Output already exists: {output}; use --overwrite to replace it")
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=output.stem + ".", suffix=".tmp", dir=output.parent)
    os.close(descriptor)
    temporary = Path(name)
    try:
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED,
                             compresslevel=compression_level, allowZip64=True) as archive:
            for number, (doc_id, path) in enumerate(files.items(), 1):
                # Keep original NPY bytes: no casting, reshaping, or precision loss.
                matrix = load_matrix(path)
                del matrix
                archive.write(path, arcname=f"{doc_id}.npy")
                if number % log_every == 0 or number == len(files):
                    print(f"Packed {number:,}/{len(files):,} documents: {output.name}", flush=True)
        # NpzFile loads only the requested member, keeping RAM bounded by document size.
        with np.load(temporary, allow_pickle=False) as archive:
            if set(archive.files) != set(files) or len(archive.files) != len(files):
                raise ValueError("Archive document keys do not match input files")
            for number, (doc_id, path) in enumerate(files.items(), 1):
                original = load_matrix(path)
                stored = archive[doc_id]
                if (original.dtype != stored.dtype or original.shape != stored.shape
                        or original.tobytes(order="C") != stored.tobytes(order="C")):
                    raise ValueError(f"Lossless verification failed: {path}")
                del original, stored
                if number % log_every == 0 or number == len(files):
                    print(f"Verified {number:,}/{len(files):,} documents: {output.name}", flush=True)
        if output.exists() and not overwrite:
            raise FileExistsError(f"Output appeared during conversion: {output}")
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    source_bytes = sum(path.stat().st_size for path in files.values())
    output_bytes = output.stat().st_size
    print(f"Saved {output} | {len(files):,} documents | "
          f"{source_bytes / 2**20:.2f} MiB -> {output_bytes / 2**20:.2f} MiB", flush=True)
    removed = 0 if keep_originals else delete_archived_sources(folder, files, output)
    print(f"Removed {removed:,} original .npy files", flush=True)
    return dict(documents=len(files), source_bytes=source_bytes, output_bytes=output_bytes,
                removed_files=removed)


def convert_archive(source, output, compression_level=6, log_every=10000):
    """Repack ZIP members to flat NPZ keys, validating arrays and exact NPY bytes.

    Never extracts, deletes source files, or overwrites an existing output.
    Suitable for archives with a train/ prefix; no paths are written from members.
    """
    source, output = Path(source), Path(output)
    if output.suffix.lower() != ".npz" or output.exists():
        raise ValueError("Choose a new output path ending in .npz")
    if not 0 <= compression_level <= 9 or log_every < 1:
        raise ValueError("Invalid compression_level or log_every")
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=output.stem + ".", suffix=".tmp", dir=output.parent)
    os.close(descriptor)
    temporary = Path(name)
    try:
        with zipfile.ZipFile(source) as origin:
            files = {}
            for member in origin.infolist():
                if member.is_dir() or not member.filename.endswith(".npy"):
                    continue
                doc_id = Path(member.filename).stem
                if not doc_id or doc_id in files:
                    raise ValueError(f"Duplicate/empty document ID: {doc_id}")
                files[doc_id] = member
            if not files:
                raise ValueError("No NPY matrices in source archive")
            with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED,
                                 compresslevel=compression_level, allowZip64=True) as target:
                for number, (doc_id, member) in enumerate(sorted(files.items()), 1):
                    data = origin.read(member)
                    validate_matrix(np.load(io.BytesIO(data), allow_pickle=False), member.filename)
                    target.writestr(f"{doc_id}.npy", data)
                    if number % log_every == 0 or number == len(files):
                        print(f"Packed {number:,}/{len(files):,} matrices", flush=True)
            with zipfile.ZipFile(temporary) as target:
                if set(target.namelist()) != {f"{doc}.npy" for doc in files}:
                    raise ValueError("NPZ key mismatch")
                for number, (doc_id, member) in enumerate(files.items(), 1):
                    with origin.open(member) as left, target.open(f"{doc_id}.npy") as right:
                        if stream_hash(left) != stream_hash(right):
                            raise ValueError(f"Lossless verification failed: {doc_id}")
                    if number % log_every == 0 or number == len(files):
                        print(f"Verified {number:,}/{len(files):,} matrices", flush=True)
            if output.exists():
                raise FileExistsError(output)
            os.replace(temporary, output)
        report = dict(documents=len(files), output=str(output), output_bytes=output.stat().st_size,
                      removed_files=0)
        print(report, flush=True)
        return report
    finally:
        temporary.unlink(missing_ok=True)


def merge_npz(output, additions):
    """Atomically add NPY files; verify all original member bytes and keep a backup.

    Existing keys may only be supplied again when the complete NPY bytes match.
    Requires exclusive access: do not run another writer against this NPZ.
    """
    output = Path(output)
    with output.open("rb") as handle:
        original_hash = stream_hash(handle).hex()
    backup = output.with_name(f"{output.stem}.before-merge-{original_hash[:12]}.npz")
    descriptor, name = tempfile.mkstemp(prefix=output.stem + ".merge.", suffix=".tmp", dir=output.parent)
    os.close(descriptor)
    temporary = Path(name)
    expected = {}
    added = 0
    try:
        with zipfile.ZipFile(output) as origin, zipfile.ZipFile(
                temporary, "w", compression=zipfile.ZIP_DEFLATED, allowZip64=True) as target:
            members = origin.namelist()
            if len(members) != len(set(members)) or any(
                    not n.endswith(".npy") or Path(n).name != n for n in members):
                raise ValueError("Expected unique flat NPY keys in existing NPZ")
            for number, member in enumerate(members, 1):
                data = origin.read(member)
                expected[member] = hashlib.sha256(data).digest()
                target.writestr(member, data)
                if number % 10000 == 0 or number == len(members):
                    print(f"MERGE preserved {number:,}/{len(members):,} existing matrices", flush=True)
            for doc_id, path in additions.items():
                if not doc_id or Path(doc_id).name != doc_id or any(c in doc_id for c in ("/", "\\", ":")):
                    raise ValueError(f"Unsafe document ID: {doc_id}")
                load_matrix(path)
                data = Path(path).read_bytes()
                key = f"{doc_id}.npy"
                digest = hashlib.sha256(data).digest()
                if key in expected:
                    if digest != expected[key]:
                        raise ValueError(f"Conflicting existing matrix: {doc_id}")
                    continue
                target.writestr(key, data)
                expected[key] = digest
                added += 1
        with zipfile.ZipFile(temporary) as merged:
            if set(merged.namelist()) != set(expected) or len(merged.namelist()) != len(expected):
                raise ValueError("Merged NPZ keys differ")
            for member, digest in expected.items():
                with merged.open(member) as handle:
                    if stream_hash(handle) != digest:
                        raise ValueError(f"Merged bytes changed: {member}")
        with output.open("rb") as handle:
            if stream_hash(handle).hex() != original_hash:
                raise ValueError("Source NPZ changed during merge; refusing replacement")
        if added:
            if not backup.exists():
                shutil.copy2(output, backup)
            with backup.open("rb") as handle:
                if stream_hash(handle).hex() != original_hash:
                    raise ValueError("Backup verification failed")
            os.replace(temporary, output)
        report = dict(added=added, documents=len(expected), output=str(output),
                      backup=str(backup) if added else None,
                      previous_sha256=original_hash)
        print(report, flush=True)
        return report
    finally:
        temporary.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=DEFAULT_ROOT,
                        help="folder containing dev/ and test/ subfolders")
    parser.add_argument("--output-dir", type=Path,
                        help="default: input root; outputs are named dev.npz and test.npz")
    parser.add_argument("--splits", nargs="+", choices=("dev", "test", "train"),
                        default=["dev", "test"])
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--keep-originals", action="store_true",
                        help="retain source .npy files instead of deleting after verification")
    parser.add_argument("--compression-level", type=int, choices=range(10), default=6)
    parser.add_argument("--source-archive", type=Path,
                        help="pack directly from a ZIP; requires exactly one --splits value; retains sources")
    args = parser.parse_args()
    output_dir = args.output_dir or args.input_root
    splits = list(dict.fromkeys(args.splits))
    if args.source_archive:
        if len(splits) != 1 or args.overwrite:
            parser.error("--source-archive requires one split and does not support --overwrite")
        convert_archive(args.source_archive, output_dir / f"{splits[0]}.npz",
                        compression_level=args.compression_level)
        return
    # Catch missing inputs and existing outputs before starting either split.
    for split in splits:
        index_split(args.input_root / split)
        target = output_dir / f"{split}.npz"
        if target.exists() and not args.overwrite:
            raise FileExistsError(f"Output already exists: {target}; use --overwrite")
    for split in splits:
        convert_split(args.input_root / split, output_dir / f"{split}.npz",
                      overwrite=args.overwrite, compression_level=args.compression_level,
                      keep_originals=args.keep_originals)


if __name__ == "__main__":
    main()
