# Handover — Liputan6 → AMR

Status as of 2026-08-12. Branch `liputan6-amr-pipeline` (19 commits ahead of
`main`, pushed to `origin` = FedrianzD/generate_amr; `upstream` is Aimar's repo,
never pushed to).

## The task

Thesis work on GNN extractive summarisation for Indonesian. Assigned 8 Aug 2026:
*"Fedrianz: Liputan 6 + parser Abdi (ID→ID), target 10rb"*. Aimar did the XLSum
equivalent; nobody had run Liputan6 through AMR before, so there is no prior art.

Full pipeline: sentences → AMR graphs → merge per document → smatch adjacency
matrix → GCN. **This repo covers the AMR-parsing stage only.**

## Done

AMR parsing is **complete** for the assigned scope:

- **9,994 graphs / 767 documents**, Liputan6 `train` split, 100% `OK` status
- Ran on Kaggle T4 in two sessions (8.0 h then ~3 h) via
  `notebooks/liputan6/amr_parse_id2id_abdi.ipynb`
- Output lives in the Kaggle dataset **`liputan6-amr-graphs`**, one
  `{doc_id}_{sent_idx}.txt` per sentence

Model: `abdiharyadi/mbart-en-id-smaller-indo-amr-parsing-translated-nafkhan`
(Smatch 0.8299), pulled from HuggingFace at runtime. Takes **plain Indonesian** —
no translation step, confirmed by Aimar and by the checkpoint's own
`dummy_input.json`.

## Next

1. Verify the final set: 767 documents, no document missing sentences (a partial
   document breaks the per-document merge).
2. Merge sentence graphs per document, build the smatch adjacency matrix
   (`n x n` for `n` sentences), feature matrix from IndoSBERT.
3. Feed the GCN. Lina's architecture is the current baseline:
   `SBERT(768) + pos enc → GCN(2 layers, hidden 256) → MLP scoring head →
   sentence selection`.

## Read before touching the notebook

**`notebooks/liputan6/README.md`** — the checkpoint disagrees with its own
tokenizer on `pad` (3 vs 1), `mask` (4 vs 35107), `len` (0 vs 38025) and the
generation stop token (`eos=2` vs `</AMR>`=38024). Three bugs came from trusting
those, and **none raised an error**: a zeroed embedding matrix, meaningless
graphs, and a 19x decode slowdown. Sections 4 and 7b verify encode and decode
against the checkpoint's reference file — keep them.

## Files

| Path | What |
| --- | --- |
| `notebooks/liputan6/amr_parse_id2id_abdi.ipynb` | the parser that produced the output |
| `notebooks/liputan6/README.md` | gotchas, Kaggle setup, pipeline |
| `scripts/liputan6/liputan6_to_csv.py` | Liputan6 JSON → `analysis_data.csv`, per sentence, with `split` |
| `scripts/liputan6/build_smatch_adjacency.py` | Sentence AMR archives → document-level SMATCH matrices |
| `data/liputan6/processed/analysis_data.csv` | 195,779 sentences / 16,350 docs (git-ignored, regenerate) |
| `notebooks/liputan6/smatch_adjacency.ipynb` | One-document SMATCH adjacency demonstration |
| `notebooks/liputan6/*_nafkhan.ipynb`, `*_nllb.ipynb` | parked concat route — needs translation, weights not public |
| `notebooks/xlsum/` | Aimar's upstream XLSum notebooks, reference only |

Regenerate the CSV with:

```bash
python scripts/liputan6/liputan6_to_csv.py --mode sentence
```

## Open questions

- **Scope.** "10rb" was read as ~10,000 *sentences*, giving 767 documents. If it
  meant 10,000 *documents*, more data is needed: the local canonical copy has
  only 5,378 train docs (67,321 sentences) against Liputan6's real ~193k.
- **Model choice.** The 8 Aug notes link the `...indo-amr-**generation**...`
  model, which is AMR→text (BLEU 50.42). The parsing model (Smatch 0.8299) is
  the right direction and is what was used — worth confirming with Abdi/MLK.
- **Test split.** Only `train` was parsed. Test (128,458 sentences / 10,972 docs)
  is untouched; evaluation will need it.
- The local Liputan6 copy is partial — `canonical/train` has 5,378 of ~193k docs.

## Data traps

- `sorted(rglob)` lists `canonical/test` before `canonical/train`, so "the first
  N documents" silently yields the **test** set. Hence the `split` column and an
  explicit `SPLIT` filter.
- Caps must take **whole documents**; a partial document is useless downstream.
- Liputan6 is already sentence-segmented (no SpaCy needed — this avoids the
  misalignment that forced truncation on XLSum) but is word-tokenized with
  punctuation split off, which `scripts/liputan6/liputan6_to_csv.py` undoes.
