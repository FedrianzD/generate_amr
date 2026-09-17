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
|   `-- extracted/{train,dev,test}/      # extracted AMR .txt files
`-- adjacency_matrices/{train,dev,test}/ # one {doc_id}.npy per document
```

All contents below `data/liputan6/` are Git-ignored. The directory layout is a
local data contract, not source code.

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
