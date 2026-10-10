"""Package custom AMR modules for the maintained Liputan6 Kaggle workflow.

Optionally package an explicitly selected local model with
--create-model --model-dir PATH. Normally the notebook downloads its checkpoint.
"""

import argparse
import zipfile
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "artifacts/kaggle"


def create_code_modules_zip():
    """Preserve common/ and model_interface/ imports in the uploaded archive."""
    folders = [PROJECT_ROOT / name for name in ("common", "model_interface")]
    for folder in folders:
        if not folder.is_dir():
            raise FileNotFoundError(f"Required module directory missing: {folder}")
    path = OUTPUT_DIR / "amr-code-modules.zip"
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
        for folder in folders:
            for source in sorted(folder.rglob("*")):
                if (source.is_file() and "__pycache__" not in source.parts
                        and source.suffix not in {".pyc", ".pyo"}):
                    archive.write(source, source.relative_to(PROJECT_ROOT).as_posix())
    print(f"Created: {path}")
    return path


def create_model_zip(model_dir):
    """Package the supplied model, including nested tokenizer/config files."""
    path = OUTPUT_DIR / "amr-model.zip"
    sources = sorted(source for source in model_dir.rglob("*") if source.is_file())
    if not sources:
        raise ValueError(f"Model directory has no files: {model_dir}")
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
        for source in sources:
            archive.write(source, source.relative_to(model_dir).as_posix())
    print(f"Created: {path}")
    return path


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--create-model", action="store_true", help="also package a local model")
    parser.add_argument("--model-dir", type=Path, help="explicit local checkpoint directory")
    args = parser.parse_args()
    if args.create_model and (args.model_dir is None or not args.model_dir.is_dir()):
        parser.error("--create-model requires --model-dir pointing to an existing directory")
    if args.model_dir is not None and not args.create_model:
        parser.error("--model-dir requires --create-model")
    if args.create_model:
        model_dir = args.model_dir.resolve()
        if model_dir == OUTPUT_DIR.resolve() or model_dir in OUTPUT_DIR.resolve().parents:
            parser.error("model directory must not contain the ZIP output directory")
    return args


def main():
    args = parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    create_code_modules_zip()
    if args.create_model:
        create_model_zip(args.model_dir)
    print("\nLiputan6 Kaggle setup:")
    print("1. Upload amr-code-modules.zip and your Liputan6 sentence table.")
    print("2. Attach previous SQLite output and set PREV_DB (or migrate text via PREV_DIR).")
    print("3. Upload notebooks/liputan6/amr_parse_id2id_abdi_kaggle_documents.ipynb.")
    print("4. Set explicit paths, enable GPU/internet, and run the reference smoke tests.")
    print("The checkpoint is downloaded at runtime unless you configure a local copy.")


if __name__ == "__main__":
    main()
