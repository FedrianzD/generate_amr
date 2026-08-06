"""
liputan6_to_csv.py

Convert the Liputan6 canonical JSON files into an `analysis_data.csv`
with the SAME format the XLSum parser expects: columns `id` and `text`,
one row per article (matching the old XLSum pipeline granularity).

`text` is reconstructed from `clean_article` (a list of sentences, each a
list of tokens) and lightly de-tokenized so it reads naturally for the
NLLB translator and the AMR parser.

Usage:
    python liputan6_to_csv.py \
        --input  Liputan6/liputan6/liputan6_data/canonical \
        --output analysis_data.csv

Then upload `analysis_data.csv` to Kaggle as part of the `liputan6-data`
dataset (alongside the `translate/` folder produced by the translation step).
"""

import os
import re
import json
import argparse
import csv
from pathlib import Path


def detokenize(tokens):
    """Join a list of tokens into a readable string.

    clean_article stores pre-tokenized text (punctuation split off), e.g.
    ["Liputan6", ".", "com", ",", "Jakarta", ":"]. A plain space-join yields
    "Liputan6 . com , Jakarta :". We fix the spacing around punctuation so it
    reads naturally: "Liputan6.com, Jakarta:".
    """
    text = " ".join(tokens)
    # No space before closing punctuation
    text = re.sub(r"\s+([.,;:!?%)\]}])", r"\1", text)
    # No space after opening brackets
    text = re.sub(r"([(\[{])\s+", r"\1", text)
    # Glue back things like "Liputan6 . com" -> "Liputan6.com"
    text = re.sub(r"\s+\.\s+(com|co|id)\b", r".\1", text)
    return text.strip()


def article_to_text(clean_article):
    """clean_article -> single text block (all sentences joined)."""
    sentences = [detokenize(sent) for sent in clean_article]
    return " ".join(s for s in sentences if s)


def convert(input_dir, output_csv):
    input_dir = Path(input_dir)
    json_files = sorted(input_dir.rglob("*.json"))
    print(f"Found {len(json_files)} JSON files under {input_dir}")

    rows = []
    skipped = 0
    for jf in json_files:
        try:
            with open(jf, "r", encoding="utf-8") as f:
                data = json.load(f)
            text = article_to_text(data["clean_article"])
            if not text:
                skipped += 1
                continue
            # id kept as string so downstream filename handling is uniform
            rows.append({"id": str(data["id"]), "text": text})
        except Exception as e:
            skipped += 1
            print(f"  Skipped {jf.name}: {e}")

    with open(output_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "text"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {output_csv} (skipped {skipped})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--input",
        default="Liputan6/liputan6/liputan6_data/canonical",
        help="Directory containing Liputan6 canonical JSON (searched recursively)",
    )
    ap.add_argument("--output", default="analysis_data.csv")
    args = ap.parse_args()
    convert(args.input, args.output)
