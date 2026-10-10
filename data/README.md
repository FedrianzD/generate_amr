# Local data layout

The contents of `data/liputan6/` are local and Git-ignored because they are
large or reproducible. The expected pipeline layout is:

```text
data/liputan6/
|-- source/Liputan6/                     # original Liputan6 corpus
|   `-- liputan6/liputan6_data/canonical/{train,dev,test}/*.json
|-- processed/
|   |-- analysis_data.csv                # generated sentence table
|   `-- extractive_vs_smatch.csv         # generated comparison report
|-- amr_graphs/
|   |-- archives/{train,dev,test}.zip    # downloaded parser archives
|   |-- extracted/{train,dev,test}/      # original AMR .txt files (retained)
|   `-- sqlite/{train,dev,test}.sqlite   # shared amr-documents-v1 graph storage
|-- features/{train,dev,test}.pt          # trusted per-article id/embeddings records
|-- features/drive-download-*/train_*.pt # downloaded train record-list shards
|-- faktualitas_lip6_v2.csv               # factuality scores; rank prob_yes_factuality within train
|-- adjacency_matrices/{train,dev,test}/ # one {doc_id}.npy per document, when retained
`-- adjacency_matrices/{train,dev,test}.npz # compressed, document-keyed matrices
```

All contents below `data/liputan6/` are Git-ignored. The directory layout is a
local data contract, not source code.

Dev/test SQLite exports use the same sentence-row schema as the train parser.
They do not replace or regenerate adjacency matrices. For conversion and resume
commands see [the SQLite converter guide](../scripts/liputan6/documents_to_sqlite.md).
The train database may remain on Kaggle until downloaded to this layout.

Generate the sentence table with:

```powershell
conda run --name generate_amr python scripts/liputan6/liputan6_to_csv.py --mode sentence
```

Generate SMATCH adjacency matrices with:

```powershell
conda run --name generate_amr python scripts/liputan6/build_smatch_adjacency.py --split train
```

Compare the extractive-oracle sentence indices with each document's strongest
SMATCH pair and highest weighted-degree sentence with:

```powershell
conda run --name generate_amr python scripts/liputan6/check_extractive_vs_smatch.py --split train
```

The detailed result is written to
`data/liputan6/processed/extractive_vs_smatch.csv` by default.

For every `.npy` matrix, row order is derived from sorted sentence AMR filenames
in the source ZIP. Keep the archive used to build the matrix so analysis can map
matrix rows back to the original `sent_idx` safely.

See [`../scripts/README.md`](../scripts/README.md) for custom paths, test split,
sampling, overwrite behavior, and all other CLI options.

Generated Kaggle upload bundles are stored separately in `artifacts/kaggle/`.
