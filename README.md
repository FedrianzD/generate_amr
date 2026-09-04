# AMRBART — AMR Parsing & Text Generation

Fine-tuning and inference pipeline for **Abstract Meaning Representation (AMR)** using [AMRBART](https://github.com/goodbai-nlp/AMRBART), built on top of HuggingFace Transformers with a custom MBart-based architecture. (Check the [Paper](https://scholar.its.ac.id/en/publications/abstract-meaning-representation-parser-development-for-cross-ling/) also built by [nafkhanzam](https://github.com/nafkhanzam))

This project supports two tasks:

| Task                                    | Direction                    | Metric    |
| --------------------------------------- | ---------------------------- | --------- |
| **AMR Parsing** (`text2amr`)            | Natural language → AMR graph | Smatch    |
| **AMR-to-Text Generation** (`amr2text`) | AMR graph → Natural language | SacreBLEU |

---

## Table of Contents

- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Setup](#setup)
- [Data Format](#data-format)
- [Usage](#usage)
  - [Training](#training)
  - [Evaluation](#evaluation)
  - [Inference](#inference)
- [Notebooks](#notebooks)
- [Kaggle Deployment](#kaggle-deployment)
- [Key Components](#key-components)

---

## Project Structure

```text
generate_amr/
├── common/                  # Shared AMR utilities and configuration
├── model_interface/         # Custom models and tokenizers
├── finetune/                # Training, evaluation, and inference
├── notebooks/
│   ├── liputan6/            # Liputan 6 parsing and SMATCH workflows
│   └── xlsum/               # XLSum reference workflows
├── scripts/
│   ├── liputan6/            # Liputan 6 conversion and adjacency tools
│   └── prepare_kaggle_datasets.py
├── data/                    # Local datasets; see data/README.md
├── artifacts/               # Generated upload bundles (Git-ignored)
├── requirements.txt
└── .gitignore
```

---

## Requirements

- **Python** 3.10+
- **PyTorch** (with CUDA support recommended)
- **GPU**: NVIDIA GPU with ≥16 GB VRAM recommended for training

---

## Setup

1. **Clone the repository** and navigate to the project directory:

   ```bash
   cd generate_amr
   ```

2. **Create and activate a virtual environment**:

   ```bash
   python -m venv venv
   source venv/bin/activate        # Linux/macOS
   # or
   .\venv\Scripts\activate         # Windows
   ```

3. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

4. **Download NLTK data** (done automatically on first run, or manually):

   ```python
   import nltk
   nltk.download('punkt')
   ```

5. **Download the pretrained AMRBART model** from HuggingFace or use your own finetuned checkpoint.

---

## Data Format

Training data should be in **JSONL format** (one JSON object per line) placed in a data directory with the following splits:

```
../data/<Dataset>/
├── train.jsonl
├── val.jsonl
└── test.jsonl
```

---

## Usage

All training and inference scripts are located in the `finetune/` directory. Before running, update the `BasePath` variable in the shell scripts to point to your data storage directory.

---

## Notebooks

See `notebooks/README.md` for the notebook index. Liputan 6 workflows are in
`notebooks/liputan6/`; XLSum reference workflows are in `notebooks/xlsum/`.

---

## Kaggle Deployment

To run this project on Kaggle:

1. **Prepare upload packages**:

   ```bash
   python scripts/prepare_kaggle_datasets.py
   ```

   This creates zip files in `artifacts/kaggle/`:
   - `amr-code-modules.zip` — `common/` and `model_interface/` modules
   - `amr-model.zip` — Finetuned model weights

2. **Upload to Kaggle** as three separate datasets:
   - `amr-code-modules` → Code modules
   - `amr-model` → Model weights
   - `xlsum-translate-data` → XLSum data files

3. **Create a Kaggle Notebook**, add all three datasets, enable GPU, and use
   `notebooks/xlsum/parser_kaggle.ipynb` as a starting point.

---

## Key Components

### AMRBartTokenizer

Custom tokenizer (`model_interface/tokenization_bart.py`) extending MBart50 with:

- AMR-specific vocabulary (relations, concepts, special tokens)
- AMR graph decoding (`decode_amr`) with linearized graph reconstruction
- Entity recategorization support

### Custom BART Model

Modified MBart architecture (`model_interface/modeling_bart.py`) for conditional generation, supporting both AMR parsing and text generation tasks.

### Metrics

- **Smatch** — Semantic match score for AMR parsing quality
- **SacreBLEU** — BLEU score for text generation quality

### Training Features

- Polynomial / cosine / linear learning rate scheduling with warmup
- Smart embedding initialization for AMR tokens
- FP16 mixed-precision training
- Early stopping with configurable patience
- TensorBoard / WandB logging support
- Distributed training support (via `torchrun`)
