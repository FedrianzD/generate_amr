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

```
generate_amr/
├── common/                        # Shared utilities & configuration
│   ├── additional-tokens.json     # AMR-specific vocabulary tokens
│   ├── callbacks.py               # Training callbacks
│   ├── constant.py                # Constants, schedulers, tokenizer/model mappings
│   ├── options.py                 # Dataclass argument definitions (Model, Data, Training)
│   ├── penman_interface.py        # Penman graph encoding utilities
│   ├── postprocessing.py          # AMR postprocessing & graph normalization
│   ├── training_args.py           # Extended HuggingFace TrainingArguments
│   └── utils.py                   # Smart embedding init, Smatch calculation, etc.
│
├── model_interface/               # Custom model & tokenizer implementations
│   ├── modeling_bart.py           # MBart-based conditional generation model
│   ├── modeling_outputs.py        # Custom model output dataclasses
│   ├── tokenization_bart.py       # AMRBartTokenizer (AMR-aware tokenization)
│   └── tokenization_mbart50.py    # Base MBart50 tokenizer
│
├── finetune/                      # Training, evaluation & inference scripts
│   ├── main.py                    # Main training/eval entry point
│   ├── seq2seq_trainer.py         # Custom Seq2Seq trainer
│   ├── base_trainer.py            # Base trainer with extended functionality
│   ├── postprocess.py             # Output postprocessing
│   │
│   ├── data_interface/            # Standard dataset loading
│   │   ├── data.py
│   │   └── dataset.py
│   ├── data_interface_concat/     # Concatenated input format dataset loading
│   │
│   ├── evaluation/                # Evaluation scripts & tools
│   │   ├── eval_gen.py            # Text generation evaluation
│   │   ├── eval_gen.sh
│   │   ├── eval_smatch.py         # Smatch score evaluation
│   │   ├── eval_smatch.sh
│   │   └── cdec-corpus/           # Corpus processing tools
│   │
│   ├── metric/
│   │   └── sacrebleu.py           # SacreBLEU metric implementation
│   │
│   ├── train-AMRBART-large-AMRParsing.sh   # Train: Text → AMR
│   ├── train-AMRBART-large-AMR2Text.sh     # Train: AMR → Text
│   ├── Eval-AMRBART-large-AMRParsing.sh    # Evaluate: Text → AMR
│   ├── Eval-AMRBART-large-AMR2Text.sh      # Evaluate: AMR → Text
│   ├── inference-amr.sh                     # Inference: Text → AMR
│   ├── inference-text.sh                    # Inference: AMR → Text
│   ├── train.sh                             # Generic training script
│   ├── eval.sh                              # Generic evaluation script
│   └── inference.sh                         # Generic inference script
│
├── example.ipynb                  # Example notebook for AMR parsing
├── parser.ipynb                   # AMR parsing notebook (local)
├── parser_kaggle.ipynb            # AMR parsing notebook (Kaggle)
├── translation.ipynb              # Translation notebook
├── prepare_kaggle_datasets.py     # Prepare datasets for Kaggle upload
├── requirements.txt               # Python dependencies
└── .gitignore
```

---

## Requirements

- **Python** 3.8+
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

| Notebook              | Description                                        |
| --------------------- | -------------------------------------------------- |
| `example.ipynb`       | Walkthrough example for AMR parsing and generation |
| `parser.ipynb`        | Full AMR parsing pipeline (local execution)        |
| `parser_kaggle.ipynb` | AMR parsing adapted for Kaggle environment         |
| `translation.ipynb`   | Translation-related experiments                    |

---

## Kaggle Deployment

To run this project on Kaggle:

1. **Prepare upload packages**:

   ```bash
   python prepare_kaggle_datasets.py
   ```

   This creates zip files in `kaggle_uploads/`:
   - `amr-code-modules.zip` — `common/` and `model_interface/` modules
   - `amr-model.zip` — Finetuned model weights

2. **Upload to Kaggle** as three separate datasets:
   - `amr-code-modules` → Code modules
   - `amr-model` → Model weights
   - `xlsum-translate-data` → XLSum data files

3. **Create a Kaggle Notebook**, add all three datasets, enable GPU, and use `parser_kaggle.ipynb` as a starting point.

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
