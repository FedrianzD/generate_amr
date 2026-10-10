# Liputan6 SBERT–GCN experiment results

This is the experiment record. Scores below are **dev**, not test results.
ROUGE values are macro-average F1 multiplied by 100. No test evaluation has
been performed for either experiment. Do not tune using the test split.
Linked run artifacts are local, Git-ignored outputs; this Markdown record is
version-controlled, but checkpoints, datasets, caches, and logs are not.

## Shared protocol

- Train: 10,000 AMR-complete documents, ranked by `prob_yes_factuality` within
  train. Completeness requires exactly canonical sentence indices `0..n-1` in
  `train.sqlite`, with matching sentence IDs. Missing/incomplete/misaligned
  documents are excluded. The selected cohort contains 116,073 sentences.
- Validation: all 10,972 canonical dev documents (136,199 sentences).
- Held-out test: 10,972 canonical documents; not evaluated.
- Features: supplied 768-dimensional sentence tensors. Position information is
  assumed included; feature provenance/order is not independently established.
- Graph: existing weighted SMATCH adjacency, unchanged between experiments.
  PyG adds self-loops and applies GCN normalization.
- Architecture: residual GCN `768→256→256`, projected outer residual,
  dropout 0.3, MLP `256→128→1`.
- Training: Adam, learning rate `0.001`, weight decay `0`, batch size `16`,
  gradient clipping `1`, 10 epochs, no early stopping, seed `42`.
- Loss: weighted binary cross-entropy. Positive weight is recomputed as the
  negative/positive training-label ratio for each label definition.
- Prediction: ranked top-2 with trigram blocking, restored to source order;
  no positive-score threshold. At most two sentences, not necessarily two.
- Checkpoint selection: highest dev ROUGE-1 F1.
- Environment: Conda `generate_amr`; NVIDIA RTX 3050 Laptop GPU; PyTorch
  `2.11.0+cu128`, PyG `2.8.0.post1`, NumPy `2.2.6`, rouge-score `0.1.2`.
- Same seed does not guarantee bitwise-identical CUDA graph operations.

## Results overview

| Experiment | Oracle-label sentence cap | Status | Best epoch | Dev R1 | Dev R2 | Dev RL | Sentence micro-F1 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| E001 — baseline | Unlimited | Completed | 7 | 29.4242 | 14.3966 | 23.9066 | 32.6295 |
| E002 — capped labels | At most 2 | Completed | 7 | 29.6389 | 14.4274 | 23.9971 | 33.0850 |

Sentence micro-F1 uses the experiment's own pseudo-labels. When the oracle
changes, its micro-F1 and weighted BCE are **not directly comparable** between
experiments. ROUGE against the unchanged abstractive dev references is the
primary comparable outcome. Both oracle searches are greedy, not globally
optimal subset searches.

## E001 — unlimited greedy ROUGE-1 labels

- Status: completed; 10 epochs; no train/dev input errors.
- Oracle: start empty; repeatedly add the sentence producing the largest
  ROUGE-1 F1 improvement against `clean_summary`; stop when no addition improves
  the score. No sentence-count cap. Ties favor the earliest source index.
- Training positives: 25,859; negatives: 90,214; mean positives/document:
  2.5859; effective positive weight: 3.4886886577.
- Prediction remains capped at two sentences, creating a potential mismatch
  between oracle label budget and prediction budget.
- Best checkpoint: epoch 7. Dev R1 `0.2942419325`, R2 `0.1439658530`,
  RL `0.2390657743`; sentence micro-F1 `0.3262952326`.
- Final epoch 10: dev R1 29.1202. The final epoch is not the selected model.
- Outputs: [summary](runs/pyg-gcn-top10000-factuality/training_summary.json),
  [history](runs/pyg-gcn-top10000-factuality/history.json),
  [config](runs/pyg-gcn-top10000-factuality/config.json),
  [best checkpoint](runs/pyg-gcn-top10000-factuality/best.pt),
  [source snapshot](runs/pyg-gcn-top10000-factuality/train_gcn.source.ipynb),
  [log](runs/pyg-gcn-top10000-factuality/training.log).

### E001 epoch history

| Epoch | Train weighted BCE | Dev R1 | Dev R2 | Dev RL |
| --- | --- | --- | --- | --- |
| 1 | 0.974316 | 29.0804 | 14.2368 | 23.6351 |
| 2 | 0.940598 | 28.8407 | 14.1028 | 23.4309 |
| 3 | 0.921638 | 29.1412 | 14.2644 | 23.6960 |
| 4 | 0.908067 | 29.0163 | 14.0673 | 23.5632 |
| 5 | 0.894591 | 29.2934 | 14.3057 | 23.8482 |
| 6 | 0.882640 | 29.3011 | 14.3016 | 23.8189 |
| 7 | 0.872988 | 29.4242 | 14.3966 | 23.9066 |
| 8 | 0.861782 | 29.4151 | 14.3343 | 23.9044 |
| 9 | 0.849831 | 29.2459 | 14.1514 | 23.7647 |
| 10 | 0.839057 | 29.1202 | 13.9397 | 23.5716 |

## E002 — greedy ROUGE-1 labels capped at two sentences

- Status: completed; 10 epochs; no train/dev input errors. Synthetic end-to-end
  and cap/cache tests passed.
- Question: does matching the oracle budget to the fixed top-2 prediction
  budget improve dev ROUGE?
- Direct change from E001: `ORACLE_MAX_SENTENCES = 2`, rather than unlimited.
  Stop at two selected sentences **or earlier** if there is no ROUGE-1 gain.
- Unchanged: training IDs and source/feature/graph inputs, architecture, seed,
  optimizer settings, epochs, selection policy, and dev checkpoint metric.
- Expected consequences: pseudo-labels and automatic positive-class weight
  change; sentence-F1 evaluation labels change as well. This is an oracle-policy
  comparison including that automatic weighting consequence, not a fixed-weight
  loss ablation.
- Caches/checkpoints/logs use a separate run directory; baseline artifacts are
  retained. Cache signatures include the oracle cap.
- Output directory: `runs/pyg-gcn-top10000-factuality-oracle2/`.
- Training positives: 18,894; negatives: 97,179; mean positives/document:
  1.8894; effective positive weight: 5.1433788504.
- The selected split-manifest SHA-256 matches E001:
  `124fa1c4bb785badcc301005cbcc39ecac4bfb4fef5fda49ffad493bbc6a7239`.
- Acceptance/comparison: compare each run's best dev ROUGE-1 F1; report R2/RL
  too. A single-seed increase is preliminary evidence, not proof of superiority.
- Test evaluation remains disabled.
- Best checkpoint: epoch 7. Dev R1 `0.2963887354`, R2 `0.1442738322`,
  RL `0.2399707933`; sentence micro-F1 `0.3308499831`.
- Difference from E001's best checkpoint (ROUGE points): R1 **+0.2147**,
  R2 **+0.0308**, RL **+0.0905**.
- Final epoch 10: dev R1 29.1659. Use the best checkpoint, not the last one.
- Outputs: [summary](runs/pyg-gcn-top10000-factuality-oracle2/training_summary.json),
  [history](runs/pyg-gcn-top10000-factuality-oracle2/history.json),
  [config](runs/pyg-gcn-top10000-factuality-oracle2/config.json),
  [best checkpoint](runs/pyg-gcn-top10000-factuality-oracle2/best.pt),
  [source snapshot](runs/pyg-gcn-top10000-factuality-oracle2/train_gcn.source.ipynb),
  [log](runs/pyg-gcn-top10000-factuality-oracle2/training.log).

### E002 epoch history

| Epoch | Train weighted BCE | Dev R1 | Dev R2 | Dev RL |
| --- | --- | --- | --- | --- |
| 1 | 1.019629 | 29.1023 | 14.2169 | 23.5538 |
| 2 | 0.983856 | 29.1600 | 14.2379 | 23.6151 |
| 3 | 0.964987 | 29.0209 | 14.1050 | 23.4310 |
| 4 | 0.949154 | 29.3147 | 14.2149 | 23.7090 |
| 5 | 0.936892 | 28.8244 | 13.6279 | 23.2029 |
| 6 | 0.924563 | 29.2831 | 14.1714 | 23.6656 |
| 7 | 0.911623 | 29.6389 | 14.4274 | 23.9971 |
| 8 | 0.899156 | 29.3237 | 14.1071 | 23.6922 |
| 9 | 0.883243 | 28.6456 | 13.3964 | 23.0748 |
| 10 | 0.870480 | 29.1659 | 13.9068 | 23.5110 |

## Interpretation so far

E001's training loss continued decreasing after the best dev checkpoint, while
dev ROUGE did not improve. This does not establish that more epochs would help.
E002 improved all three dev ROUGE measures slightly in this seed-42 comparison.
The capped-label configuration is the better observed dev result, but this
single run does not establish statistical significance or general superiority.
The change also recomputes the automatic class weight, so the observed difference
cannot be attributed to the sentence cap independently of that loss-weight change.
Keep test held out while choosing further experiments; use multiple seeds and/or
paired document-level uncertainty estimates before making a stronger research claim.
