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

Sentence 0 additionally has the "Liputan6.com, <city>:" dateline stripped - it
is masthead boilerplate, not article content, and it otherwise turns up as a
publication/location predicate in the AMR of every single document.

Usage:
    python scripts/liputan6/liputan6_to_csv.py \
        --input  data/liputan6/source/Liputan6/liputan6/liputan6_data/canonical \
        --output data/liputan6/processed/analysis_data.csv \
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


PROJECT_ROOT = Path(__file__).resolve().parents[2]
LIPUTAN6_DATA_DIR = PROJECT_ROOT / "data" / "liputan6"


def detokenize(tokens):
    """Join a list of tokens into a readable string.

    clean_article stores pre-tokenized text (punctuation split off), e.g.
    ["Liputan6", ".", "com", ",", "Jakarta", ":"]. A plain space-join yields
    "Liputan6 . com , Jakarta :". We fix the spacing around punctuation so it
    reads naturally: "Liputan6.com, Jakarta:".
    """
    text = " ".join(tokens)
    # No space before closing punctuation: "Jakarta :" -> "Jakarta:"
    text = re.sub(r"\s+([.,;:!?%)\]}])", r"\1", text)
    # No space after opening brackets: "( KAA )" -> "(KAA)"
    text = re.sub(r"([(\[{])\s+", r"\1", text)
    # Domains: "Liputan6. com" -> "Liputan6.com". Must run AFTER the rule above,
    # which has already removed the space before the dot.
    text = re.sub(r"\.\s+(com|co|id|net|org)\b", r".\1", text)
    # Numbers split around punctuation: "14. 00" -> "14.00", "1, 5" -> "1,5"
    text = re.sub(r"(\d)([.,])\s+(?=\d)", r"\1\2", text)
    return text.strip()


# The masthead as it survives detokenize(), tolerating "Liputan 6 . com" spacing
# and the lowercase spellings that appear in a few hundred documents.
_MASTHEAD = r"Liputan\s?6\s*\.\s*com"

# Leading dateline: "Liputan6.com, Jakarta:" and its damaged variants. The
# leading .{0,120}? absorbs scrape debris that got glued on in front of the
# masthead - a stray ")" or "B", and in ~30 documents the article HEADLINE plus
# leftover HTML attributes, e.g.
#   'Jenazah ... Dimakamkanfont color= " ffff00 " Liputan6.com, Yogyakarta: '
# None of that is article text, so removing it with the dateline is the point.
DATELINE = re.compile(
    r"""^.{0,120}?            # scrape debris / glued-in headline + markup
        """ + _MASTHEAD + r"""
        \s*,?\s*              # comma before the city
        [^:]{0,40}?           # dateline city
        \s*:\s*               # the colon that terminates the dateline
    """, re.X | re.I)

# ~4 documents write the dateline with no colon ("Liputan6.com, Mimika Sepasang
# cahaya ..."), where the city cannot be told apart from the first word of the
# sentence. Drop only the masthead there and leave the city in the text.
LEAD_MASTHEAD = re.compile(r"^\W{0,4}" + _MASTHEAD + r"\s*,?\s*", re.I)


def strip_dateline(text):
    """Remove the "Liputan6.com, <city>:" boilerplate from a lede sentence.

    Applied ONLY to sentence 0, which is where every dateline lives. The 45
    other mentions in the corpus are ordinary content - "berdialog dengan
    netters Liputan6.com", "Tim liputan6.com SCTV", "www. liputan6.com" - and
    deleting those would leave ungrammatical sentences for the parser.
    """
    prev = None
    while prev != text:              # a few ledes carry two stacked datelines
        prev = text
        text = DATELINE.sub("", text, count=1)
    return LEAD_MASTHEAD.sub("", text).strip()


def article_to_text(clean_article):
    """clean_article -> single text block (all sentences joined)."""
    sentences = [detokenize(sent) for sent in clean_article]
    if sentences:
        sentences[0] = strip_dateline(sentences[0])
    return " ".join(s for s in sentences if s)


def split_of(json_path, input_dir):
    """Split name = the directory under `input_dir` holding the file.

    Liputan6 canonical is laid out as canonical/<split>/<id>.json, so the parent
    directory name is the split ("train" / "test" / "dev"). Recorded per row so
    downstream steps can select a split instead of silently mixing them - note
    that sorted() puts canonical/test before canonical/train, so "the first N
    rows" is NOT the training set.
    """
    rel = json_path.relative_to(input_dir)
    return rel.parts[0] if len(rel.parts) > 1 else "unknown"


def convert(input_dir, output_csv, mode="sentence"):
    input_dir = Path(input_dir)
    output_csv = Path(output_csv)
    json_files = sorted(input_dir.rglob("*.json"))
    print(f"Found {len(json_files)} JSON files under {input_dir} (mode={mode})")

    rows = []
    skipped_sents = 0
    skipped_docs = 0
    n_datelines = 0
    for jf in json_files:
        try:
            with open(jf, "r", encoding="utf-8") as f:
                data = json.load(f)
            doc_id = str(data["id"])
            split = split_of(jf, input_dir)
            clean_article = data["clean_article"]

            if mode == "article":
                text = article_to_text(clean_article)
                if not text:
                    skipped_docs += 1
                    continue
                rows.append({"id": doc_id, "doc_id": doc_id, "split": split,
                             "sent_idx": -1, "text": text})
            else:  # sentence
                for i, sent_tokens in enumerate(clean_article):
                    text = detokenize(sent_tokens)
                    if i == 0:
                        stripped = strip_dateline(text)
                        if stripped != text:
                            n_datelines += 1
                        text = stripped
                    if not text:
                        skipped_sents += 1
                        continue
                    # id = "{doc_id}_{sent_idx}" -> unique filename per sentence
                    rows.append({"id": f"{doc_id}_{i}", "doc_id": doc_id,
                                 "split": split, "sent_idx": i, "text": text})
        except Exception as e:
            skipped_docs += 1
            print(f"  Skipped {jf.name}: {e}")

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(output_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "doc_id", "split", "sent_idx", "text"])
        writer.writeheader()
        writer.writerows(rows)

    n_docs = len(set(r["doc_id"] for r in rows))
    print(f"Wrote {len(rows)} rows from {n_docs} documents to {output_csv}")
    print(f"  (skipped {skipped_sents} empty sentences, {skipped_docs} bad/empty docs)")
    if mode == "sentence":
        print(f"  (stripped the Liputan6.com dateline from {n_datelines} lede sentences)")

    by_split = {}
    for r in rows:
        s = by_split.setdefault(r["split"], {"sents": 0, "docs": set()})
        s["sents"] += 1
        s["docs"].add(r["doc_id"])
    print("  Per split:")
    for s, v in sorted(by_split.items()):
        print(f"    {s:8s}: {v['sents']:>7} sentences / {len(v['docs']):>6} docs")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--input",
        default=LIPUTAN6_DATA_DIR / "source" / "Liputan6" / "liputan6" / "liputan6_data" / "canonical",
        type=Path,
        help="Directory containing Liputan6 canonical JSON (searched recursively)",
    )
    ap.add_argument(
        "--output",
        default=LIPUTAN6_DATA_DIR / "processed" / "analysis_data.csv",
        type=Path,
    )
    ap.add_argument("--mode", choices=["sentence", "article"], default="sentence",
                    help="sentence: one AMR per sentence (default, matches AMR-GNN pipeline); "
                         "article: one AMR per whole article")
    args = ap.parse_args()
    convert(args.input, args.output, args.mode)
