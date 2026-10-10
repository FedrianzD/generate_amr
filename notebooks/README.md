# Notebooks

The six maintained workflow notebooks live in [`liputan6/`](liputan6/README.md).
They remain self-contained so they can be uploaded to Kaggle or Colab without
importing project scripts (the AMR parser still needs the custom model modules).

| Task | Notebook |
| --- | --- |
| Parse AMR, one file per document, Kaggle | [Document parser](liputan6/amr_parse_id2id_abdi_kaggle_documents.ipynb) |
| Parse AMR, GCP/local GPU | [GCP parser](liputan6/amr_parse_id2id_abdi_gcp_fixed.ipynb) |
| Build SMATCH adjacency, Kaggle | [Kaggle adjacency](liputan6/build_smatch_adjacency_kaggle.ipynb) |
| Build SMATCH adjacency, GCP | [GCP adjacency](liputan6/build_smatch_adjacency_gcp.ipynb) |
| Compare extractive oracle with SMATCH | [Analysis](liputan6/check_extractive_vs_smatch_colab.ipynb) |
| Pilot PyG GCN training | [Training](liputan6/train_gcn.ipynb) |

Single-document inspection is in
[`diagnostics/document_pairwise_smatch.ipynb`](diagnostics/document_pairwise_smatch.ipynb).
Older parsers, temporary experiments, the upstream demo, and all XLSum notebooks
are preserved in [`../archive/`](../archive/README.md), not active entry points.

Edit explicit input/output paths in each notebook before running. Do not assume
the GCP and Kaggle variants accept identical graph layouts. Test a few documents
before a full run. Local notebooks must use the `generate_amr` Conda kernel.

```powershell
conda run --name generate_amr jupyter notebook notebooks/diagnostics/document_pairwise_smatch.ipynb
```
