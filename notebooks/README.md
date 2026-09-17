# Notebooks

| Notebook | What it does |
| --- | --- |
| [`liputan6/amr_parse_id2id_abdi.ipynb`](liputan6/amr_parse_id2id_abdi.ipynb) | **Active.** Liputan6 → AMR with Abdi's monolingual ID→ID parser. No translation. |
| [`liputan6/amr_parse_id2id_abdi_gcp.ipynb`](liputan6/amr_parse_id2id_abdi_gcp.ipynb) | GCP/local-GPU variant of the active Liputan6 parser. |
| [`liputan6/document_pairwise_smatch.ipynb`](liputan6/document_pairwise_smatch.ipynb) | Inspect every sentence pair and SMATCH score within one Liputan 6 document. |
| [`liputan6/check_extractive_vs_smatch_colab.ipynb`](liputan6/check_extractive_vs_smatch_colab.ipynb) | Self-contained extractive-ground-truth vs SMATCH analysis for data on Colab Enterprise. |
| [`liputan6/temporary_external_smatch.ipynb`](liputan6/temporary_external_smatch.ipynb) | Temporary SMATCH diagnostic; not the main pipeline entry point. |
| [`liputan6/translate_id2en_nllb.ipynb`](liputan6/translate_id2en_nllb.ipynb) | Parked. Liputan6 id→en with NLLB-1.3B; only needed for the concat route. |
| [`liputan6/amr_parse_id2en_concat_nafkhan.ipynb`](liputan6/amr_parse_id2en_concat_nafkhan.ipynb) | Parked. Liputan6 → AMR with Nafkhan's concat parser (needs translations, weights not public). |
| [`xlsum/parser_local.ipynb`](xlsum/parser_local.ipynb) | Upstream (Aimar). XLSum → AMR, run locally with the concat model. |
| [`xlsum/parser_kaggle.ipynb`](xlsum/parser_kaggle.ipynb) | Upstream (Aimar). The same, adapted for Kaggle. |
| [`xlsum/translate_nllb.ipynb`](xlsum/translate_nllb.ipynb) | Upstream (Aimar). XLSum id→en with NLLB. |
| [`example_amrbart_inference.ipynb`](example_amrbart_inference.ipynb) | Upstream AMRBART demo: parse a couple of sentences. |

> **Working on Liputan6?** Read [`liputan6/README.md`](liputan6/README.md) first.
> Abdi's checkpoint disagrees with its own tokenizer on nearly every special
> token, and three of the resulting bugs are completely silent.

## Which one to run

For Liputan6, use **`liputan6/amr_parse_id2id_abdi.ipynb`**. It parses Indonesian
sentences straight into AMR with
`abdiharyadi/mbart-en-id-smaller-indo-amr-parsing-translated-nafkhan`
(Smatch 0.8299), downloaded from HuggingFace at runtime.

There is no translation step, because that model takes plain Indonesian — verified
by section 4 of the notebook, which reproduces the reference `input_ids` from the
model repo's own `dummy_input.json`.

The two parked notebooks exist only for the alternative route through Nafkhan's
`mbart-en-id-smaller-concat-finetuned`, whose input is
`id_ID <indonesian> en_XX <english>` and therefore needs the corpus translated
first. Those weights are not published.

## Kaggle setup (the active notebook)

Upload two datasets:

| Dataset title | File |
| --- | --- |
| `amr-code-modules` | `artifacts/kaggle/amr-code-modules.zip` (see [`scripts/README.md`](../scripts/README.md)) |
| `liputan6-data` | `analysis_data.csv` (see [`scripts/README.md`](../scripts/README.md)) |

Then import the notebook, add both as Input, enable **GPU T4** and **Internet ON**
(the model is downloaded at runtime), run cell 1, switch the kernel to
"Python (3.10) AMR" as instructed, and run the rest.

## A note on paths

Every notebook uses explicit absolute paths for stable locations such as model,
dataset, module, archive, and output directories. None of them determines those
locations from the current working directory. Edit the path constants in the
notebook's setup/configuration cell when moving between local Windows, Kaggle,
and GCP/Colab Enterprise. Paths for individual files whose names depend on a
document or sentence ID are still assembled at runtime.

## Open a local notebook

From the repository root:

```powershell
conda run --name generate_amr jupyter notebook notebooks/liputan6/document_pairwise_smatch.ipynb
```
