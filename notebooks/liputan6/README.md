# Liputan6 workflows

Only maintained pipeline notebooks are kept here. Choose the environment you
actually use; Kaggle and GCP variants are separate because their paths and
execution contracts differ, not because one is a backup of the other.

| Notebook | Environment | Purpose |
| --- | --- | --- |
| `amr_parse_id2id_abdi_kaggle_documents.ipynb` | Kaggle GPU | Parse into SQLite and resume by committed sentence IDs. |
| `amr_parse_id2id_abdi_gcp_fixed.ipynb` | GCP/local GPU | SQLite parser with persistent output, encoder bounds and truncation logging. |
| `documents_to_sqlite_kaggle.ipynb` | Kaggle CPU | Convert existing document text files to the parser's SQLite schema. |
| `build_smatch_adjacency_kaggle.ipynb` | Kaggle CPU | Parallel SMATCH adjacency from ZIP/folder input, resumable output. |
| `build_smatch_adjacency_gcp.ipynb` | GCP CPU | Adjacency with completeness checks against the parser's Parquet. |
| `check_extractive_vs_smatch_colab.ipynb` | Colab/local | Compare oracle indices, strongest pair, and highest weighted degree. |
| `train_gcn.ipynb` | Local | PyG GCNConv pilot training and optional test evaluation. |

## Parser setup

Model: `abdiharyadi/mbart-en-id-smaller-indo-amr-parsing-translated-nafkhan`.
It receives Indonesian text directly; translation is not needed.

Build the custom module bundle locally:

```powershell
conda run --name generate_amr python scripts/prepare_kaggle_datasets.py
conda run --name generate_amr python scripts/liputan6/liputan6_to_csv.py --mode sentence
```

Attach `artifacts/kaggle/amr-code-modules.zip`, the sentence table, and previous
graph output if resuming. Set the notebook's explicit paths to the actual mounts.
Use the corrected Parquet when configured for Parquet; the table builder above
produces CSV, not Parquet. Kaggle outputs must be under `/kaggle/working`, not
the read-only `/kaggle/input` mount.

For Save Version, dependencies must be installed before importing them. The
recent Python 3.13 runtime needs compatibility testing with the custom model;
successful package installation alone does not prove correct graph decoding.
Run reference encode/decode checks and a small generation test **before**
scanning/copying a large dataset. A graph `()` is not a valid successful parse.

Checkpoint metadata is inconsistent: do not blindly trust `len(tokenizer)`,
padding IDs, or the standard EOS. Verify vocabulary rows (38025), the encoder
suffix, AMR BOS (38023), AMR EOS (38024), and the model's actual position limit.
Keep encoder truncation and reference checks intact.

## Output and alignment

Both parsers write `OUTPUT_DB`, with one row per sentence graph in the `graphs`
table. The schema matches the document-to-SQLite converter. Set `PREV_DB` to
your converted database or previous parser output and leave `PREV_DIR` empty.
On Kaggle, a fresh session creates a writable copy under
`/kaggle/working/amr_documents/train/train.sqlite`. Save the output and attach
it as a dataset on the next run. On GCP, keep `OUTPUT_DB` on persistent storage;
leave both previous-input settings empty to resume directly from it.

For legacy migration, set only `PREV_DIR` to the selected split's text folder.
Both sentence files and document files are supported. Source files are never
changed. Each generation batch commits atomically and closes its connection.
Resume queries stored IDs for `SPLIT`, so partial documents can receive missing
sentences. Conflicting duplicate graphs stop the run rather than overwrite data.
PENMAN metadata, including truncation flags, stays inside `amr_penman`.

The existing adjacency/analysis consumers still expect text files and do not
read SQLite. Adapt or export their inputs before running them.
Preserve original zero-based indices, filter the split explicitly, and sample
whole documents. Missing/misaligned documents must not silently become training
examples. Start adjacency runs with `MAX_DOCUMENTS = 10`.

## Training

```powershell
conda run --name generate_amr jupyter notebook notebooks/liputan6/train_gcn.ipynb
```

The notebook is the only maintained training entry point and uses PyG
`GCNConv`. The earlier dense CLI is preserved at
`archive/scripts/train_gcn_dense.py`; its checkpoints are not interchangeable
with PyG checkpoints. Do not regenerate this notebook with the archived generator.

Current experiment: top 10,000 AMR-complete train documents ranked by `prob_yes_factuality`
in `data/liputan6/faktualitas_lip6_v2.csv`; canonical dev is validation, canonical
test is held out. The CSV's `factuality` column is binary and is not the ranking
score. Ties use lexicographic document ID. `amr_graphs/sqlite/train.sqlite`
is checked against canonical `clean_article`: exactly one graph for each index
`0..n-1`, no extra indices, and matching sentence IDs. Incomplete/misaligned
documents are excluded before selection. This checks coverage, not semantic AMR
quality. Candidates are checked in score order until 10,000 pass; lower-ranked
candidates after that cutoff are not audited. Missing selected features/matrices
still stop the run rather than silently changing the cohort.

Configure all paths in the notebook's configuration cell. Defaults are trusted
`features/dev.pt` and `test.pt` are lists of `id`/`embeddings` records; train
defaults to the four downloaded `train_1.pt`–`train_4.pt` shards, read sequentially.
Feature paths can also point to a single record-list file or directories of
`{doc_id}.pt` tensors. Adjacency loads `adjacency_matrices/{split}.npz` when
available, with individual `{split}/{doc_id}.npy` as fallback.

The notebook generates greedy ROUGE-1 F1 labels from `clean_summary` (not the
existing `extractive_summary`), with resumable content-keyed oracle caches.
The current E002 default caps oracle labels at two sentences (`ORACLE_MAX_SENTENCES=2`),
stopping earlier when no improvement is possible. Set the cap to `None` only
with a separate output directory to reproduce the unlimited-label E001 policy.
It trains `768→256→256` GCN layers with a projected outer residual and an MLP
`256→128→1`. Weighted BCE uses class counts from the selected train data only.
Dev macro ROUGE-1 F1 selects `best.pt`; `last.pt` is also saved. Prediction is
ranked top-2 with trigram blocking, without a positive-score threshold.

E001 artifacts are retained under `runs/pyg-gcn-top10000-factuality/`. Current
E002 outputs use `runs/pyg-gcn-top10000-factuality-oracle2/`: selection/split manifests,
configuration, input errors, oracle caches, history, and checkpoints. The files
`train_amr_eligibility_checked.csv` and `train_amr_excluded.csv` record the
checked candidates and exclusion reasons. Activate
`RUN_TEST_EVALUATION` only after model choices are fixed. The final cell writes
test ROUGE and predictions directly, without requiring the separate ROUGE CLI.
Test evaluation can run independently after the imports/config/helpers/model
cells. Older PyG/dense checkpoint architectures are not compatible.

See [EXPERIMENTS.md](../../EXPERIMENTS.md) for the exact comparison protocol and results.

Older sentence-only Kaggle parsing, concat/translation notebooks, temporary
experiments, and XLSum workflows are in [`../../archive/`](../../archive/README.md).
