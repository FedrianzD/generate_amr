# Notebooks

| Notebook | What it does |
| --- | --- |
| [`liputan6/amr_parse_id2id_abdi.ipynb`](liputan6/amr_parse_id2id_abdi.ipynb) | **Active.** Liputan6 → AMR with Abdi's monolingual ID→ID parser. No translation. |
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
| `amr-code-modules` | `kaggle_uploads/amr-code-modules.zip` (run `prepare_kaggle_datasets.py`) |
| `liputan6-data` | `analysis_data.csv` (run `liputan6_to_csv.py`) |

Then import the notebook, add both as Input, enable **GPU T4** and **Internet ON**
(the model is downloaded at runtime), run cell 1, switch the kernel to
"Python (3.10) AMR" as instructed, and run the rest.

## A note on paths

The `liputan6/` notebooks and `xlsum/parser_kaggle.ipynb` use absolute
`/kaggle/input/...` paths, so their location in this repo does not matter.

The other upstream notebooks use paths relative to the notebook's own directory
and were written to sit at the repo root; they were adjusted when moved here, and
still expect `models/` and `data/` to live one level *above* the repo.
