"""
liputan6_to_csv.py

Convert the Liputan6 canonical JSON files into an `analysis_data.csv` for the
AMR pipeline. Columns: `id`, `doc_id`, `sent_idx`, `text`.

Granularity (matches the Aimar/Fajri AMR-GNN research pipeline):
  --mode sentence  (DEFAULT): one row per SENTENCE. AMR is parsed per sentence
                   and later merged into one graph per document; the sentence
                   nodes feed the NxN smatch adjacency matrix. `id` is
                   "{doc_id}_{sent_idx}".
  --mode article : one row per article (whole clean_article joined). `id` is
                   the document id; sent_idx is -1.

`clean_article` is a list of sentences, each a list of tokens. It is already
sentence-split, so no SpaCy is needed. Tokens are lightly de-tokenized so the
text reads naturally for the NLLB translator and the AMR parser.

Usage:
    python liputan6_to_csv.py \
        --input  Liputan6/liputan6/liputan6_data/canonical \
        --output analysis_data.csv \
        --mode   sentence

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


def convert(input_dir, output_csv, mode="sentence"):
    input_dir = Path(input_dir)
    json_files = sorted(input_dir.rglob("*.json"))
    print(f"Found {len(json_files)} JSON files under {input_dir} (mode={mode})")

    rows = []
    skipped_sents = 0
    skipped_docs = 0
    for jf in json_files:
        try:
            with open(jf, "r", encoding="utf-8") as f:
                data = json.load(f)
            doc_id = str(data["id"])
            clean_article = data["clean_article"]

            if mode == "article":
                text = article_to_text(clean_article)
                if not text:
                    skipped_docs += 1
                    continue
                rows.append({"id": doc_id, "doc_id": doc_id, "sent_idx": -1, "text": text})
            else:  # sentence
                for i, sent_tokens in enumerate(clean_article):
                    text = detokenize(sent_tokens)
                    if not text:
                        skipped_sents += 1
                        continue
                    # id = "{doc_id}_{sent_idx}" -> unique filename per sentence
                    rows.append({"id": f"{doc_id}_{i}", "doc_id": doc_id,
                                 "sent_idx": i, "text": text})
        except Exception as e:
            skipped_docs += 1
            print(f"  Skipped {jf.name}: {e}")

    with open(output_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "doc_id", "sent_idx", "text"])
        writer.writeheader()
        writer.writerows(rows)

    n_docs = len(set(r["doc_id"] for r in rows))
    print(f"Wrote {len(rows)} rows from {n_docs} documents to {output_csv}")
    print(f"  (skipped {skipped_sents} empty sentences, {skipped_docs} bad/empty docs)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--input",
        default="Liputan6/liputan6/liputan6_data/canonical",
        help="Directory containing Liputan6 canonical JSON (searched recursively)",
    )
    ap.add_argument("--output", default="analysis_data.csv")
    ap.add_argument("--mode", choices=["sentence", "article"], default="sentence",
                    help="sentence: one AMR per sentence (default, matches AMR-GNN pipeline); "
                         "article: one AMR per whole article")
    args = ap.parse_args()
    convert(args.input, args.output, args.mode)
