"""Check lossless NPZ conversion, document keys, and failed-output protection."""
import importlib.util
import tempfile
import zipfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("npy_to_npz", ROOT / "scripts/liputan6/npy_to_npz.py")
converter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(converter)


def expect_error(error, function, *args, **kwargs):
    try:
        function(*args, **kwargs)
    except error:
        return
    raise AssertionError(f"Expected {error.__name__}")


def main():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        source = root / "dev"
        source.mkdir()
        nested = source / "nested"
        nested.mkdir()
        matrices = {
            "001": np.array([[0, 0.25], [0.25, 0]], dtype=np.float32),
            "100": np.zeros((1, 1), dtype=np.float64),
            "200": np.asfortranarray(np.array(
                [[0, 0.5, 0.75], [0.5, 0, 1], [0.75, 1, 0]], dtype=">f4")),
        }
        for doc_id, matrix in matrices.items():
            np.save(nested / f"{doc_id}.npy", matrix)
        original_bytes = {path: path.read_bytes() for path in source.rglob("*.npy")}
        output = root / "dev.npz"
        report = converter.convert_split(source, output, keep_originals=True)
        assert report["documents"] == 3
        assert report["removed_files"] == 0
        with np.load(output, allow_pickle=False) as archive:
            assert archive.files == sorted(matrices)
            for doc_id, expected in matrices.items():
                actual = archive[doc_id]
                assert actual.dtype == expected.dtype and actual.shape == expected.shape
                assert actual.tobytes() == expected.tobytes()
        assert all(path.read_bytes() == content for path, content in original_bytes.items())
        zipped = root / "source.zip"
        with zipfile.ZipFile(zipped, "w") as archive:
            for path, content in original_bytes.items():
                archive.writestr(f"train/{path.name}", content)
        zip_output = root / "from_zip.npz"
        assert converter.convert_archive(zipped, zip_output)["documents"] == 3
        with zipfile.ZipFile(zip_output) as archive:
            assert all(archive.read(path.name) == content
                       for path, content in original_bytes.items())
        expect_error(ValueError, converter.convert_archive, zipped, zip_output)
        duplicate_zip = root / "duplicate.zip"
        with zipfile.ZipFile(duplicate_zip, "w") as archive:
            archive.writestr("a/001.npy", next(iter(original_bytes.values())))
            archive.writestr("b/001.npy", next(iter(original_bytes.values())))
        expect_error(ValueError, converter.convert_archive, duplicate_zip, root / "duplicate.npz")
        assert not (root / "duplicate.npz").exists()
        assert not list(root.glob("*.tmp"))
        expect_error(FileExistsError, converter.convert_split, source, output)
        archive_bytes = output.read_bytes()
        invalid = source / "bad.npy"
        for value in (np.ones((2, 3)), np.array([[float("nan")]]),
                      np.array([[0, 0.5], [0, 0]]), np.ones((2, 2)),
                      np.array([[0, 2], [2, 0]]), np.array([[object()]], dtype=object)):
            np.save(invalid, value)
            expect_error(ValueError, converter.convert_split, source, output, overwrite=True)
            assert output.read_bytes() == archive_bytes
            assert all(path.read_bytes() == content for path, content in original_bytes.items())
            assert not list(root.glob("*.tmp"))
        invalid.unlink()
        np.save(source / "001.npy", matrices["001"])
        expect_error(ValueError, converter.convert_split, source, output, overwrite=True)
        (source / "001.npy").unlink()
        # Cleanup refuses even a header-only change, retaining every original.
        changed = nested / "001.npy"
        changed.write_bytes(original_bytes[changed] + b"changed")
        expect_error(ValueError, converter.delete_archived_sources, source,
                     converter.index_split(source), output)
        assert len(list(source.rglob("*.npy"))) == 3
        changed.write_bytes(original_bytes[changed])
        report = converter.convert_split(source, output, overwrite=True)
        assert report["removed_files"] == 3
        assert not list(source.rglob("*.npy"))
        with np.load(output, allow_pickle=False) as archive:
            assert set(archive.files) == set(matrices)
    print("PASS: variable sizes, dtypes/endian, document IDs, exact values, source preservation, "
          "duplicate rejection, invalid input, atomic overwrite, verified source deletion, "
          "changed-source protection and temporary-file cleanup")


if __name__ == "__main__":
    main()
