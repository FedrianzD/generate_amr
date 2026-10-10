# Retired workflows (recoverable)

These files were removed from active directories during cleanup on 2026-10-10.
They are preserved, not recommended as production entry points. No datasets,
AMR graphs, matrices, feature files, or checkpoints were deleted.

| Location | Contents / reason |
| --- | --- |
| `finetune/` | Legacy AMRBART training, evaluation, and inference; unused by the active Liputan6 parsers and GCN pipeline. |
| `notebooks/xlsum/` | All three upstream XLSum notebooks; not part of the active Liputan6 pipeline. |
| `notebooks/liputan6/amr_parse_id2id_abdi.ipynb` | Earlier sentence-file Kaggle parser, superseded for the document-output workflow. |
| `notebooks/liputan6/*nafkhan.ipynb`, `*nllb.ipynb` | Parked concat/translation route. |
| `notebooks/liputan6/temporary_*.ipynb` | Exploratory SMATCH/NumPy scratch notebooks. |
| `notebooks/example_amrbart_inference.ipynb` | Upstream demo. |
| `scripts/build_training_notebook.py` | Stale generator; disabled to avoid replacing the maintained PyG notebook with dense definitions. |
| `scripts/check_gcn_equivalence.py` | Disabled stale checker expecting the retired PyG CLI. |
| `scripts/train_gcn_dense.py` | Retired dense training CLI; preserved for historical checkpoints. Training now uses the maintained notebook. |

Archived notebook data/model paths remain historical external paths; they were
not rewritten to guess locations on your machine. Reconfigure them before reuse.
Shared modules in `common/` and `model_interface/` remain active because the
Liputan6 parsers use them. The separate AMRBART training stack was subsequently
moved to `archive/finetune/` at the user's request. Its original wrapper paths
are historical; see [its README](finetune/README.md) before reuse.

The exact duplicate `scripts/liputan6/train_gcn_dense_legacy.py` was removed;
its byte-identical implementation is retained at `archive/scripts/train_gcn_dense.py`.
The active `scripts/liputan6/train_gcn.py` was subsequently removed at the user's
request; the notebook's project-root check no longer depends on that file.
The untracked `hybrid_tadaptive.zip` was moved to the Git-ignored
`artifacts/imports/hybrid_tadaptive.zip` (preserve that file separately if needed
on another machine). Diagnostic checks moved to `tests/`, and single-document
inspection moved to `notebooks/diagnostics/`.

Restore a notebook by moving it back and checking its configured paths first.
See the [active notebook index](../notebooks/README.md) for current workflows.
