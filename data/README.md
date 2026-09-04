# Local data layout

The contents of `data/liputan6/` are local and Git-ignored because they are
large or reproducible. The expected pipeline layout is:

```text
data/liputan6/
├── source/Liputan6/                     # original Liputan 6 corpus
├── processed/analysis_data.csv          # generated sentence table
├── amr_graphs/archives/{train,test}.zip # parser output, one AMR per sentence
└── adjacency_matrices/{train,test}/     # generated document-level .npy files
```

Generate the sentence table with:

```powershell
python scripts/liputan6/liputan6_to_csv.py
```

Generate SMATCH adjacency matrices with:

```powershell
python scripts/liputan6/build_smatch_adjacency.py
```

Generated Kaggle upload bundles are stored separately in `artifacts/kaggle/`.
