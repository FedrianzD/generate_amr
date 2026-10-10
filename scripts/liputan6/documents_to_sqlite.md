# Convert AMR text files to SQLite

Local dev and test use the same `amr-documents-v1` schema as train. Convert
legacy sentence files directly, without making intermediate document files:

```powershell
conda run --name generate_amr python scripts/liputan6/documents_to_sqlite.py --input-dir data/liputan6/amr_graphs/extracted/dev --output data/liputan6/amr_graphs/sqlite/dev.sqlite --split dev --commit-every 5000
conda run --name generate_amr python scripts/liputan6/documents_to_sqlite.py --input-dir data/liputan6/amr_graphs/extracted/test --output data/liputan6/amr_graphs/sqlite/test.sqlite --split test --commit-every 5000
```

Original text files and adjacency matrices are not changed. SQLite stores one
row per sentence graph, with document IDs and numeric sentence indices, just
like the train parser. Query a document with `ORDER BY sent_idx`.

`--input-dir` also accepts a `.zip` archive directly. For the original local
dev/test archives, replace `data/liputan6/amr_graphs/extracted/dev` with
`data/liputan6/amr_graphs/archives/dev.zip` (and similarly for test). This
avoids opening hundreds of thousands of separate files. No extraction or
temporary text files are needed. Existing SQLite rows are checked for equality.

Upload `notebooks/liputan6/documents_to_sqlite_kaggle.ipynb` to Kaggle and
attach the dataset containing your graph files. Edit `INPUT_DIR`, `SPLIT`, and
`OUTPUT_DB` for one split at a time, then run all cells. No GPU or
model download is required. The notebook contains the converter itself.
It requires `penman` (already installed in the AMR parsing notebook).
If it is missing, install it in your Kaggle session before running.

Alternatively, attach `documents_to_sqlite.py` as a Kaggle dataset and run:

```python
%run /kaggle/input/YOUR-SCRIPT-DATASET/documents_to_sqlite.py --input-dir /kaggle/input/YOUR-AMR-DATASET/amr_documents/train --output /kaggle/working/train.sqlite --split train
```

The input must be a stable snapshot, with `{doc_id}.txt` files containing
one or more independent PENMAN sentence graphs, each with an
`# ::id {doc_id}_{sent_idx}` entry. Subfolders are searched recursively.
Choose one split folder specifically; the converter cannot infer splits
from document filenames. Legacy `{doc_id}_{sent_idx}.txt` files are also
accepted when they contain exactly one graph and its metadata ID matches
the filename. Missing graph IDs are rejected, not guessed.

All graph triples, the top node, and PENMAN metadata are preserved and checked
by decoding the serialized graph again. Whitespace is normalized; this is
not a byte-for-byte copy of the source document. Input files are not modified.
Rows have `split`, `doc_id`, `sent_idx`, `sentence_id`, and `amr_penman` columns.
Partial documents and nonconsecutive sentence indices are valid.

Each transaction contains 100 document files by default (`--commit-every`).
An interrupted run retains committed batches. Re-run against the same writable
database: identical graphs are retained, missing sentences are inserted, and
conflicting graphs stop the conversion. Rows absent from the new input are
retained. Malformed documents cause the current batch to roll back. Progress
and final counts are printed; success means the supplied files were imported,
not that the entire training split has been parsed.

Download `/kaggle/working/train.sqlite` after conversion, or save a Kaggle
notebook version with outputs. For a later Kaggle session, attach the prior
database as a dataset and copy it into `/kaggle/working` before running the
converter against the newer document snapshot. Kaggle input datasets are
read-only. Copy only a closed database; this converter closes its connection
before printing the final report and uses rollback journaling.

To retrieve one document's sentence graphs in numeric sentence order:

```python
import sqlite3

with sqlite3.connect("file:/kaggle/working/train.sqlite?mode=ro", uri=True) as db:
    graphs = db.execute(
        "SELECT sentence_id, amr_penman FROM graphs "
        "WHERE split=? AND doc_id=? ORDER BY sent_idx",
        ("train", "100"),
    ).fetchall()
```

Both maintained AMR parsers now read and write this SQLite schema. Set their
`PREV_DB` to the converted database and leave `PREV_DIR` empty. On Kaggle,
the parser creates a writable copy under `/kaggle/working` and continues
with only the missing sentence IDs for the selected split.
Existing SMATCH consumers also require an adapter to read this database.

Local CLI and checks (from the repository root):

```powershell
conda run --name generate_amr python scripts/liputan6/documents_to_sqlite.py --input-dir path/to/train --output artifacts/train.sqlite
conda run --name generate_amr python tests/verify_sqlite_conversion.py
```
