# Project checks

Run from the project root using the existing `generate_amr` environment:

```powershell
conda run --name generate_amr python tests/check_project_layout.py
conda run --name generate_amr python tests/verify_document_notebook_storage.py
conda run --name generate_amr python tests/verify_sqlite_parser_storage.py
conda run --name generate_amr python tests/verify_sqlite_conversion.py
conda run --name generate_amr python tests/verify_npz_conversion.py
conda run --name generate_amr python tests/verify_sqlite_smatch_generation.py
conda run --name generate_amr python tests/check_training_notebook.py
```

The layout check verifies Python syntax, notebook schemas and active code-cell
syntax, local documentation links, and import paths in a temporary module ZIP.
It does not execute cloud notebook cells or download a model.

The parser storage check uses synthetic graphs in a temporary directory (no
model/GPU download). It executes both notebooks' actual storage and dataset
code, testing converter compatibility, text migration, metadata, split isolation,
partial-document resume, and batch rollback. The old document-storage command
is a compatibility entry point for the same check; running either is sufficient.

The SQLite conversion check covers metadata/triple preservation, numeric
sentence order, partial documents, incremental reruns, conflicts, malformed
inputs, and rollback of a failed batch. It uses only synthetic graphs.

The NPZ conversion check covers variable matrix sizes, dtypes and byte order,
document IDs, exact preservation, optional source retention, verified deletion,
changed-source protection, duplicate/invalid input rejection, and protection of
existing output when conversion fails.

The SQLite SMATCH check computes actual scores on tiny synthetic graphs,
checks exact canonical index coverage and sentence ordering, and verifies that
NPZ merges preserve old member bytes, retain a backup, reject conflicting keys,
and safely reuse identical additions. It does not access real dataset files.

The training check needs PyTorch Geometric but no real dataset. It executes
the training notebook on synthetic train/dev/test fixtures in a temporary
directory for one epoch. It verifies continuous factuality ranking and ties,
exclusion of higher-scored incomplete and equal-count misaligned AMR documents,
sequential feature shards and NPZ loading (with NPY fallback),
split isolation, greedy labels and cache invalidation, graph minibatching,
trigram blocking, dev-ROUGE checkpoint selection, test prediction output,
missing-selected-input failure, and atomic checkpoints. It does not train on
the real train/dev/test datasets or write permanent run outputs.
