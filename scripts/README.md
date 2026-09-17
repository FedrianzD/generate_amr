# Script command reference

Jalankan semua command dari root repositori. Seluruh command Python di bawah
memakai environment Conda proyek secara eksplisit.

## Bantuan umum

Setiap CLI aktif mendukung `--help`:

```powershell
conda run --name generate_amr python scripts/liputan6/liputan6_to_csv.py --help
conda run --name generate_amr python scripts/liputan6/build_smatch_adjacency.py --help
conda run --name generate_amr python scripts/liputan6/check_extractive_vs_smatch.py --help
conda run --name generate_amr python scripts/prepare_kaggle_datasets.py --help
```

## Pipeline Liputan6

### 1. `liputan6_to_csv.py`

Mengubah JSON canonical Liputan6 menjadi tabel input parser.

Default, satu baris per kalimat:

```powershell
conda run --name generate_amr python scripts/liputan6/liputan6_to_csv.py --mode sentence
```

Satu baris per artikel:

```powershell
conda run --name generate_amr python scripts/liputan6/liputan6_to_csv.py --mode article --output data/liputan6/processed/articles.csv
```

Input dan output custom:

```powershell
conda run --name generate_amr python scripts/liputan6/liputan6_to_csv.py --input D:\path\to\canonical --output D:\path\to\analysis_data.csv --mode sentence
```

Kolom mode `sentence` adalah `id`, `doc_id`, `split`, `sent_idx`, dan `text`.
Dateline `Liputan6.com, <kota>:` dibuang hanya dari kalimat pertama.

### 2. `build_smatch_adjacency.py`

Membaca graph AMR dari direktori `.txt` hasil ekstraksi (atau ZIP sebagai
fallback), mengelompokkan graph per dokumen, lalu menyimpan satu adjacency
matrix NumPy untuk setiap dokumen.

Train:

```powershell
conda run --name generate_amr python scripts/liputan6/build_smatch_adjacency.py --split train
```

Test:

```powershell
conda run --name generate_amr python scripts/liputan6/build_smatch_adjacency.py --split test
```

Dev:

```powershell
conda run --name generate_amr python scripts/liputan6/build_smatch_adjacency.py --split dev
```

Train dan test:

```powershell
conda run --name generate_amr python scripts/liputan6/build_smatch_adjacency.py --split both
```

Smoke test untuk 10 dokumen:

```powershell
conda run --name generate_amr python scripts/liputan6/build_smatch_adjacency.py --split train --max-documents 10
```

Sangat disarankan melakukan smoke test sebelum dev/test penuh:

```powershell
conda run --name generate_amr python scripts/liputan6/build_smatch_adjacency.py --split dev --max-documents 10
conda run --name generate_amr python scripts/liputan6/build_smatch_adjacency.py --split test --max-documents 10
```

Gunakan lokasi custom dan timpa matriks lama:

```powershell
conda run --name generate_amr python scripts/liputan6/build_smatch_adjacency.py --split train --train-zip D:\path\to\train.zip --output D:\path\to\adjacency_matrices --overwrite
```

Input default yang diprioritaskan:

```text
data/liputan6/amr_graphs/extracted/{train,dev,test}/**/*.txt
```

Jika direktori split tidak tersedia atau kosong, skrip memakai fallback:

```text
data/liputan6/amr_graphs/archives/{train,dev,test}.zip
```

Output default:

```text
data/liputan6/adjacency_matrices/{train,dev,test}/{doc_id}.npy
```

### 3. `check_extractive_vs_smatch.py`

Membandingkan ground-truth `extractive_summary` dengan pasangan SMATCH
tertinggi dan weighted degree setiap kalimat.

Train:

```powershell
conda run --name generate_amr python scripts/liputan6/check_extractive_vs_smatch.py --split train
```

Dev dan test:

```powershell
conda run --name generate_amr python scripts/liputan6/check_extractive_vs_smatch.py --split dev --output data/liputan6/processed/extractive_vs_smatch_dev.csv
conda run --name generate_amr python scripts/liputan6/check_extractive_vs_smatch.py --split test --output data/liputan6/processed/extractive_vs_smatch_test.csv
```

Batasi ke 100 dokumen:

```powershell
conda run --name generate_amr python scripts/liputan6/check_extractive_vs_smatch.py --split train --max-documents 100
```

Train dan test dengan output custom:

```powershell
conda run --name generate_amr python scripts/liputan6/check_extractive_vs_smatch.py --split both --output data/liputan6/processed/extractive_vs_smatch_all.csv
```

Lokasi input custom:

```powershell
conda run --name generate_amr python scripts/liputan6/check_extractive_vs_smatch.py --split dev --adjacency-root D:\path\to\adjacency_matrices --amr-root D:\path\to\extracted --source-root D:\path\to\canonical --output D:\path\to\result.csv
```

Jika beberapa skor maksimum hampir sama akibat presisi floating point, toleransi
tie dapat diatur:

```powershell
conda run --name generate_amr python scripts/liputan6/check_extractive_vs_smatch.py --split train --tie-tolerance 1e-6
```

Output default:

```text
data/liputan6/processed/extractive_vs_smatch.csv
```

Untuk Colab Enterprise, tersedia notebook self-contained
[`check_extractive_vs_smatch_colab.ipynb`](../notebooks/liputan6/check_extractive_vs_smatch_colab.ipynb).
Notebook tersebut mengimplementasikan logika analisis langsung di dalam cell
dan tidak memerlukan import skrip ini.

## Kaggle packaging

### `prepare_kaggle_datasets.py`

Membuat ZIP `common/` dan `model_interface/` untuk notebook Kaggle:

```powershell
conda run --name generate_amr python scripts/prepare_kaggle_datasets.py
```

Output:

```text
artifacts/kaggle/amr-code-modules.zip
```

Untuk sekaligus mencoba membuat ZIP model dari lokasi model yang dikonfigurasi
di dalam skrip:

```powershell
conda run --name generate_amr python scripts/prepare_kaggle_datasets.py --create-model
```

Opsi `--create-model` memerlukan direktori model lokal yang sesuai dan dapat
menghasilkan file sekitar beberapa GB.

## Evaluation utilities

Menghitung SMATCH untuk file gold dan prediksi AMR:

```powershell
conda run --name generate_amr python finetune/evaluation/eval_smatch.py path\to\gold.amr path\to\prediction.amr
```

Menghitung metrik generasi teks:

```powershell
conda run --name generate_amr python finetune/evaluation/eval_gen.py --in-tokens path\to\prediction.txt --in-reference-tokens path\to\reference.txt
```

Utilitas pada `finetune/evaluation/cdec-corpus/` berasal dari toolchain upstream
dan memerlukan shell/Perl. Utilitas tersebut bukan bagian dari pipeline aktif
Liputan6.

## Training dan inference AMRBART

Wrapper Bash dan command training didokumentasikan di
[`finetune/README.md`](../finetune/README.md). Skrip tersebut memerlukan model,
dataset JSONL, GPU, dan shell Bash; jangan menganggapnya sebagai bagian dari
pipeline adjacency Liputan6.
