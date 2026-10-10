"""Check source/notebook syntax, documentation links, and module packaging offline."""

import ast
import importlib.util
import re
import tempfile
import zipfile
from pathlib import Path

import nbformat
from IPython.core.inputtransformer2 import TransformerManager


ROOT = Path(__file__).resolve().parents[1]


def main():
    sources = [path for folder in ("scripts", "tests", "archive/scripts")
               for path in (ROOT / folder).rglob("*.py")]
    for path in sources:
        ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
    print(f"PASS: {len(sources)} Python files parse")

    transformer = TransformerManager()
    notebooks = [*list((ROOT / "notebooks").rglob("*.ipynb")),
                 *list((ROOT / "archive/notebooks").rglob("*.ipynb"))]
    for path in notebooks:
        notebook = nbformat.read(path, as_version=4)
        nbformat.validate(notebook)
        if ROOT / "archive" not in path.parents:
            for index, cell in enumerate(notebook.cells):
                if cell.cell_type == "code":
                    compile(transformer.transform_cell(cell.source),
                            f"{path}:cell-{index}", "exec")
    print(f"PASS: {len(notebooks)} notebook schemas; active code cells compile")

    missing = []
    documents = [ROOT / "README.md", ROOT / "HANDOVER.md",
                 *list((ROOT / "notebooks").rglob("*.md")),
                 ROOT / "scripts/README.md", ROOT / "tests/README.md",
                 ROOT / "archive/README.md"]
    for path in documents:
        for target in re.findall(r"\]\(([^)]+)\)", path.read_text(encoding="utf-8")):
            if "://" in target or target.startswith("#"):
                continue
            if not (path.parent / target.split("#")[0]).exists():
                missing.append((str(path), target))
    if missing:
        raise AssertionError(f"Broken local documentation links: {missing}")
    print("PASS: local Markdown links resolve")

    spec = importlib.util.spec_from_file_location(
        "packaging_check", ROOT / "scripts/prepare_kaggle_datasets.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    with tempfile.TemporaryDirectory() as directory:
        module.OUTPUT_DIR = Path(directory)
        bundle = module.create_code_modules_zip()
        with zipfile.ZipFile(bundle) as archive:
            names = archive.namelist()
            assert "common/constant.py" in names
            assert "common/additional-tokens.json" in names
            assert "model_interface/modeling_bart.py" in names
            assert all("__pycache__" not in name for name in names)
            assert archive.testzip() is None
        print(f"PASS: module ZIP preserves import paths ({len(names)} files)")


if __name__ == "__main__":
    main()
