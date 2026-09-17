# AMRBART training, evaluation, and inference

Direktori ini mempertahankan training stack AMRBART untuk dua task:

| Task | Input -> output | Metrik |
| --- | --- | --- |
| `text2amr` | teks -> graph AMR | SMATCH |
| `amr2text` | graph AMR -> teks | BLEU |

Pipeline ini terpisah dari pembuatan adjacency matrix Liputan6. Skrip Liputan6
yang aktif berada di `scripts/liputan6/`.

## Persyaratan

- Jalankan Python melalui Conda environment `generate_amr`.
- Pasang seluruh `requirements.txt` terlebih dahulu. Tanpa dependency lengkap,
  bahkan `finetune/main.py --help` dapat berhenti pada import dependency.
- Training praktis memerlukan GPU NVIDIA dengan VRAM yang memadai.
- Wrapper `.sh` memerlukan Bash, sehingga ditujukan untuk Linux, WSL, Git Bash,
  Kaggle, atau VM cloud.
- Dataset training harus berformat JSONL dan model harus tersedia pada path
  yang diberikan.

## Layout wrapper generik

Wrapper `train.sh`, `eval.sh`, dan `inference.sh` mengharapkan layout:

```text
generate_amr/
|-- ds/
|   |-- <train-dataset>/train.jsonl
|   `-- <eval-dataset>/
|       |-- dev.jsonl
|       `-- test.jsonl
|-- models/<model-name>/
`-- finetune/
```

`inference.sh` mengharapkan `ds/<dataset>/inference.jsonl`.

## Wrapper generik

Karena wrapper memanggil `main.py` relatif terhadap working directory, jalankan
dari `finetune/`. Dari root repositori, bentuk command yang aman adalah:

```bash
conda run --name generate_amr bash -lc 'cd finetune && bash train.sh <model-name> <train-dataset> <eval-dataset>'
```

Evaluation:

```bash
conda run --name generate_amr bash -lc 'cd finetune && bash eval.sh <model-name> <dataset>'
```

Inference:

```bash
conda run --name generate_amr bash -lc 'cd finetune && bash inference.sh <model-name> <dataset> <output-name>'
```

Contoh bentuk parameter, dengan asumsi direktori yang diperlukan memang ada:

```bash
conda run --name generate_amr bash -lc 'cd finetune && bash train.sh mbart-model liputan6-train liputan6-eval'
```

## Menjalankan `main.py` langsung

`main.py` menerima argumen Hugging Face di command line atau satu file JSON
konfigurasi.

```powershell
conda run --name generate_amr python finetune/main.py path\to\training_config.json
```

Untuk melihat seluruh argumen yang didukung:

```powershell
conda run --name generate_amr python finetune/main.py --help
```

File konfigurasi harus menyediakan setidaknya path model, file dataset, task,
output directory, serta flag train/eval/predict yang dibutuhkan.

## Wrapper upstream khusus LDC

File berikut merupakan template upstream:

```text
train-AMRBART-large-AMRParsing.sh
train-AMRBART-large-AMR2Text.sh
Eval-AMRBART-large-AMRParsing.sh
Eval-AMRBART-large-AMR2Text.sh
inference-amr.sh
inference-text.sh
```

Skrip tersebut memiliki path dataset/cache yang hard-coded seperti
`/mnt/nfs-storage/data` dan dapat meminta penghapusan output lama. Periksa dan
sesuaikan `BasePath`, `DataPath`, model, serta output directory sebelum
menjalankannya. Skrip ini tidak disarankan sebagai quick start Liputan6.

## Evaluation utilities

SMATCH:

```powershell
conda run --name generate_amr python finetune/evaluation/eval_smatch.py path\to\gold.amr path\to\prediction.amr
```

Text generation metrics:

```powershell
conda run --name generate_amr python finetune/evaluation/eval_gen.py --in-tokens path\to\prediction.txt --in-reference-tokens path\to\reference.txt
```

`postprocess.py` dan `sandbox.py` masih memiliki path/model contoh yang
hard-coded dan dipertahankan sebagai utilitas pengembangan, bukan CLI umum.
