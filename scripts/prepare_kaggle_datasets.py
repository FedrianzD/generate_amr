"""
prepare_kaggle_datasets.py

This script prepares zip files for uploading to Kaggle as datasets.
It creates the following:
  1. amr-code-modules.zip  — contains common/ and model_interface/ folders
  2. amr-model.zip         — contains the finetuned model files
  3. (Instructions)        — for xlsum data (too large to zip here)

Usage:
    python scripts/prepare_kaggle_datasets.py

After running, upload the generated zip files to Kaggle:
  - Go to https://www.kaggle.com/datasets
  - Click "New Dataset"
  - Upload each zip file as a separate dataset
"""

import argparse
import os
import zipfile
from pathlib import Path

# ─── Paths ───────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

COMMON_DIR = PROJECT_ROOT / "common"
MODEL_INTERFACE_DIR = PROJECT_ROOT / "model_interface"
MODEL_DIR = PROJECT_ROOT.parent / "models" / "mbart-en-id-smaller-concat-finetuned" / "mbart-en-id-smaller-concat-finetuned"
XLSUM_DIR = PROJECT_ROOT.parent / "data" / "xlsum"

OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "kaggle"


def create_code_modules_zip():
    """Zip common/ and model_interface/ folders (excluding __pycache__)."""
    zip_path = OUTPUT_DIR / "amr-code-modules.zip"
    print(f"\n[1/2] Creating code modules zip: {zip_path}")

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for folder_name, folder_path in [("common", COMMON_DIR), ("model_interface", MODEL_INTERFACE_DIR)]:
            if not folder_path.exists():
                print(f"  WARNING: {folder_path} does not exist, skipping.")
                continue

            for root, dirs, files in os.walk(folder_path):
                # Skip __pycache__ directories
                dirs[:] = [d for d in dirs if d != "__pycache__"]

                for file in files:
                    file_path = Path(root) / file
                    # Archive name preserves the folder structure
                    arcname = os.path.join(folder_name, os.path.relpath(file_path, folder_path))
                    zf.write(file_path, arcname)
                    print(f"  Added: {arcname}")

    size_mb = zip_path.stat().st_size / (1024 * 1024)
    print(f"  ✓ Created {zip_path.name} ({size_mb:.1f} MB)")
    return zip_path


def create_model_zip():
    """Zip the finetuned model files."""
    zip_path = OUTPUT_DIR / "amr-model.zip"
    print(f"\n[2/2] Creating model zip: {zip_path}")

    if not MODEL_DIR.exists():
        print(f"  ERROR: Model directory not found: {MODEL_DIR}")
        print(f"  Please check the path and try again.")
        return None

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for file in MODEL_DIR.iterdir():
            if file.is_file():
                zf.write(file, file.name)
                size_mb = file.stat().st_size / (1024 * 1024)
                print(f"  Added: {file.name} ({size_mb:.1f} MB)")

    size_mb = zip_path.stat().st_size / (1024 * 1024)
    print(f"  ✓ Created {zip_path.name} ({size_mb:.1f} MB)")
    return zip_path


def print_data_instructions():
    """Print instructions for uploading the xlsum data."""
    print("\n" + "=" * 70)
    print("INSTRUCTIONS FOR UPLOADING XLSUM DATA TO KAGGLE")
    print("=" * 70)
    print(f"""
The XLSum data directory is at:
  {XLSUM_DIR}

It contains:
  - analysis_data.csv  (~170 MB)
  - translate/         (47,802 .txt files)

OPTIONS for uploading:

  Option A (Recommended): Upload the existing zip if available
    File: {XLSUM_DIR / 'translation_xlsum.zip'} 
    Plus upload analysis_data.csv separately

  Option B: Create a new zip with both files
    This may take a while due to the large number of files.
    You can run:
      cd {XLSUM_DIR}
      zip -r xlsum-data.zip analysis_data.csv translate/

KAGGLE DATASET NAMING:
  When uploading to Kaggle, use these dataset names (slug):
    1. amr-code-modules    → for the code zip
    2. amr-model           → for the model zip  
    3. xlsum-translate-data → for the data files

  These names will be referenced in the notebook as:
    /kaggle/input/amr-code-modules/
    /kaggle/input/amr-model/
    /kaggle/input/xlsum-translate-data/
""")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--create-model",
        action="store_true",
        help="create the large model ZIP without an interactive prompt",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    print("=" * 70)
    print("KAGGLE DATASET PREPARATION SCRIPT")
    print("=" * 70)

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {OUTPUT_DIR}")

    # Step 1: Code modules
    code_zip = create_code_modules_zip()

    # Step 2: Model (this will be large ~1.5GB)
    print("\n" + "-" * 70)
    print("NOTE: The model zip will be ~1.5GB. This may take a few minutes.")
    if args.create_model:
        create_model_zip()
    else:
        print("  Skipped model zip creation (pass --create-model to include it).")
        print(f"  You can manually upload the model directory from:")
        print(f"    {MODEL_DIR}")

    # Step 3: Data instructions
    print_data_instructions()

    print("\n" + "=" * 70)
    print("NEXT STEPS")
    print("=" * 70)
    print(f"""
1. Go to https://www.kaggle.com/datasets and create datasets:

   Dataset 1: "amr-code-modules"
     Upload: {OUTPUT_DIR / 'amr-code-modules.zip'}

   Dataset 2: "amr-model" 
     Upload: {OUTPUT_DIR / 'amr-model.zip'}
     (or upload the model directory directly)

   Dataset 3: "xlsum-translate-data"
     Upload: analysis_data.csv + translate/ folder
     (see instructions above)

2. Create a new Kaggle Notebook
3. Add all 3 datasets as data sources
4. Enable GPU (T4 x2 or P100)
5. Copy cells from parser_kaggle.ipynb

Files are ready in: {OUTPUT_DIR}
""")


if __name__ == "__main__":
    main()
