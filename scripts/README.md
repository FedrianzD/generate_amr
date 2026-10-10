# Script command reference

Jalankan semua command dari root repositori. Seluruh command Python di bawah
memakai environment Conda proyek secara eksplisit.

## Bantuan umum

Setiap CLI aktif mendukung `--help`:

```powershell
conda run --name generate_amr python scripts/liputan6/liputan6_to_csv.py --help
conda run --name generate_amr python scripts/liputan6/build_smatch_adjacency.py --help
conda run --name generate_amr python scripts/liputan6/check_extractive_vs_smatch.py --help
conda run --name generate_amr python scripts/liputan6/benchmark_npy_vs_smatch.py --help
conda run --name generate_amr python scripts/liputan6/evaluate_rouge.py --help
conda run --name generate_amr python scripts/prepare_kaggle_datasets.py --help
```

## Pipeline Liputan6

### Pack completed adjacency files into NPZ

Convert existing `dev/` and `test/` document matrices into one compressed archive
per split, then delete the original `.npy` files after verification:

```powershell
conda run --name generate_amr python scripts/liputan6/npy_to_npz.py
```

Defaults: inputs `data/liputan6/adjacency_matrices/{dev,test}/**/*.npy`;
outputs `data/liputan6/adjacency_matrices/dev.npz` and `test.npz`.
Custom paths (also suitable for Kaggle input/output mounts):

```powershell
conda run --name generate_amr python scripts/liputan6/npy_to_npz.py --input-root path/to/adjacency_matrices --output-dir path/to/packed
```

Use `--splits dev` for one split, `--keep-originals` to retain source files,
or `--overwrite` to replace existing archives.

To pack the train ZIP directly (retains the ZIP and all extracted NPY files):

```powershell
conda run --name generate_amr python scripts/liputan6/npy_to_npz.py --splits train --source-archive "data/liputan6/adjacency_matrices/train/archive (1).zip"
```

This mode validates every array, flattens keys to document IDs, and verifies
the exact original NPY bytes. It requires a new output path and one split.
The converter loads one document at a time, preserves full matrices and their
dtypes, verifies every saved array exactly, then publishes the archive atomically.
It rejects duplicate document IDs, object arrays, and invalid SMATCH matrices.

To fill a document-ID list from AMR SQLite and merge into an existing NPZ:

```powershell
conda run --name generate_amr python scripts/liputan6/build_smatch_adjacency.py --split train --sqlite data/liputan6/amr_graphs/sqlite/train.sqlite --document-ids-csv path/to/missing_ids.csv --merge-npz data/liputan6/adjacency_matrices/train.npz
```

The CSV needs an `id` column. Every requested document must have exactly the
canonical sentence indices `0..n-1`. New NPY matrices are saved atomically in
`adjacency_matrices/train/`; existing valid files are reused. The same pairwise
SMATCH search as the text/ZIP generator is used (randomized search, not guaranteed
bitwise reproducible). The NPZ merge verifies all original member bytes and all
additions, keeps a hash-named pre-merge backup, and publishes atomically. A
`train_sqlite_smatch_generation.json` report records IDs, settings, and checksums.
Do not run concurrent writers against the same NPZ.
After publishing the archive, cleanup compares each complete source file against
its archived bytes before deleting it. Changed sources stop cleanup; the archive
is retained. Directories and unrelated files are left in place. If cleanup is
interrupted, retain the verified archive; do not overwrite it using only the
remaining source files, which would omit already archived documents.
Use stable, finished input files. Conversion preserves existing matrix row order;
it cannot recover sentence IDs that were absent from the `.npy` files.

Read a document using its filename stem (including any leading zeroes):

```python
import numpy as np

with np.load("data/liputan6/adjacency_matrices/dev.npz", allow_pickle=False) as split:
    adjacency = split["100"]
```

Current training and analysis readers still expect individual `.npy` files;
their loaders need adaptation before switching to these archives.

### Pilot GCN setelah feature dan adjacency tersedia

Training hanya memakai notebook `notebooks/liputan6/train_gcn.ipynb`;
CLI `train_gcn.py` sudah dikeluarkan dari direktori aktif. Notebook memakai
`torch_geometric.nn.GCNConv` dan memerlukan PyTorch
Geometric pada environment `generate_amr`. File `.pt`
harus berasal dari sumber tepercaya. Instal dependensi di environment proyek
sesuai kebijakan `AGENTS.md`; jangan instal secara global.

```powershell
conda run --name generate_amr python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
```

Untuk mesin Windows dengan driver NVIDIA yang mendukung CUDA 12.8, gunakan
build GPU resmi (menggantikan build CPU pada environment yang sama):

```powershell
conda run --name generate_amr python -m pip install --upgrade torch==2.11.0+cu128 --index-url https://download.pytorch.org/whl/cu128
conda run --name generate_amr python -s -m pip install torch-geometric
```

Notebook memilih CUDA otomatis bila tersedia; ubah `DEVICE` untuk memaksa CPU.

```powershell
conda run --name generate_amr jupyter notebook notebooks/liputan6/train_gcn.ipynb
```

The notebook reads trusted feature record lists (including sequential train
shards), `adjacency_matrices/{split}.npz` (NPY-folder fallback), and canonical
source JSON. It ranks train rows in `faktualitas_lip6_v2.csv` by
`prob_yes_factuality`, excludes incomplete/misaligned graph coverage using
`amr_graphs/sqlite/train.sqlite`, then selects exactly 10,000 complete documents.
It uses full dev for validation and test only for final
evaluation. Missing selected inputs stop the run rather than replacing IDs.
Training uses greedy ROUGE-1 F1 labels capped at two sentences, a residual
two-layer GCN, and an MLP.
Prediction is ranked top-2 with trigram blocking; dev macro ROUGE-1 F1 selects
the checkpoint. The self-contained notebook is
[`notebooks/liputan6/train_gcn.ipynb`](../notebooks/liputan6/train_gcn.ipynb).
Current output is stored in `runs/pyg-gcn-top10000-factuality-oracle2/`.
The unlimited-label baseline remains in `runs/pyg-gcn-top10000-factuality/`.
See [the experiment record](../EXPERIMENTS.md) for results and differences.
Kode dense lama hanya disimpan di `archive/scripts/train_gcn_dense.py` untuk
referensi historis; checkpoint notebook PyG tidak kompatibel dengan state dict dense.
Validasi memastikan ID dan jumlah baris cocok, tetapi file feature tidak
menyimpan ID per kalimat. Urutan kalimat dalam tensor harus dikonfirmasi dari
kode pembuat feature sebelum hasil dipakai dalam laporan.

Untuk audit input, jalankan cell data loading tanpa cell training. Untuk
pembanding tanpa graph, set `BASELINE = True` dan gunakan folder `OUTPUT` terpisah.
For a data-free smoke test, run `tests/check_training_notebook.py` in the project
Conda environment. It uses temporary synthetic train/dev/test inputs.

Setelah konfigurasi dipilih pada validation, set `RUN_TEST_EVALUATION = True`
pada cell terakhir. Cell itu membaca `OUTPUT/best.pt` dan menulis
`test_metrics.json` serta `test_predictions.jsonl` tanpa training ulang.

The notebook reports macro ROUGE-1/2/L F1 against `clean_summary` and supplementary
micro-F1 against the new oracle labels. Neither is a factuality metric. Test
must not select hyperparameters. ROUGE is already computed in the notebook;
the separate CLI below is an optional independent recheck.

Untuk mengukur ringkasan ekstraktif terhadap `clean_summary` setelah evaluasi
test menghasilkan `test_predictions.jsonl`:

```powershell
conda run --name generate_amr python -s scripts/liputan6/evaluate_rouge.py --predictions runs/pyg-gcn-top10000-factuality-oracle2/test_predictions.jsonl
```

Hasil ROUGE-1, ROUGE-2, dan ROUGE-L disimpan di
`runs/pyg-gcn-top10000-factuality-oracle2/test_rouge.json`. Skor adalah rata-rata F1 per artikel,
tanpa stemming. Ini mengukur overlap kata, bukan faktualitas.

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

### 4. `benchmark_npy_vs_smatch.py`

Membandingkan waktu memuat adjacency matrix `.npy` dengan waktu membaca AMR,
parse PENMAN, dan menghitung seluruh pasangan SMATCH secara langsung. Kedua
jalur memakai sampel dokumen yang sama. Waktu indexing direktori dilaporkan
terpisah agar tidak tercampur dengan waktu per dokumen.

Benchmark default untuk 1.000 dokumen dev:

```powershell
conda run --name generate_amr python scripts/liputan6/benchmark_npy_vs_smatch.py --split dev --documents 1000
```

Smoke test cepat untuk 10 dokumen:

```powershell
conda run --name generate_amr python scripts/liputan6/benchmark_npy_vs_smatch.py --split dev --documents 10
```

Sampel default dipilih secara acak tetapi deterministik dengan seed `42`.
Gunakan `--selection first` untuk memilih ID dokumen pertama, atau `--seed`
untuk mengganti sampel acak. Hasil per dokumen ditulis ke:

```text
data/liputan6/processed/benchmark_npy_vs_smatch.csv
```

Kolom hasil memisahkan `npy_load_seconds`, `amr_read_parse_seconds`, dan
`smatch_compute_seconds`. Skrip juga memeriksa kesamaan shape serta melaporkan
selisih nilai maksimum antara matriks tersimpan dan hasil hitung ulang. SMATCH
memakai randomized hill climbing, sehingga selisih kecil tidak otomatis berarti
file `.npy` rusak.

## Kaggle packaging

### Document AMRs to SQLite

Use the self-contained
[`documents_to_sqlite_kaggle.ipynb`](../notebooks/liputan6/documents_to_sqlite_kaggle.ipynb)
to convert one `.txt` per document into resumable SQLite storage. It preserves
sentence graph metadata and allows partial documents. See
[`documents_to_sqlite.md`](liputan6/documents_to_sqlite.md) for Kaggle setup,
incremental conversion, and the standalone CLI. Both maintained parsers use
this SQLite schema; set `PREV_DB` to the converted database to resume parsing.

### `prepare_kaggle_datasets.py`

Membuat ZIP `common/` dan `model_interface/` untuk notebook Kaggle:

```powershell
conda run --name generate_amr python scripts/prepare_kaggle_datasets.py
```

Output:

```text
artifacts/kaggle/amr-code-modules.zip
```

Opsional: buat ZIP checkpoint lokal dengan lokasi eksplisit:

```powershell
conda run --name generate_amr python scripts/prepare_kaggle_datasets.py --create-model --model-dir D:\path\to\checkpoint
```

Opsi `--create-model` wajib disertai `--model-dir` dan dapat menghasilkan file
beberapa GB. Default hanya membundel modul; tidak ada lagi path model/XLSum
hardcoded atau instruksi upload XLSum dalam workflow aktif.

## Pemeriksaan pengembang dan arsip

Helper validasi dipisahkan dari command pipeline ke [`../tests/`](../tests/README.md).

```powershell
conda run --name generate_amr python tests/check_project_layout.py
conda run --name generate_amr python tests/verify_document_notebook_storage.py
conda run --name generate_amr python tests/check_training_notebook.py
```

Generator notebook dan checker equivalence lama disimpan di
[`../archive/`](../archive/README.md), bukan entry point aktif.

## Evaluation utilities

Menghitung SMATCH untuk file gold dan prediksi AMR:

```powershell
conda run --name generate_amr python archive/finetune/evaluation/eval_smatch.py path\to\gold.amr path\to\prediction.amr
```

Menghitung metrik generasi teks:

```powershell
conda run --name generate_amr python archive/finetune/evaluation/eval_gen.py --in-tokens path\to\prediction.txt --in-reference-tokens path\to\reference.txt
```

Utilitas pada `archive/finetune/evaluation/cdec-corpus/` berasal dari toolchain upstream
dan memerlukan shell/Perl. Utilitas tersebut bukan bagian dari pipeline aktif
Liputan6.

## Training dan inference AMRBART

Wrapper Bash dan command training didokumentasikan di
[`archive/finetune/README.md`](../archive/finetune/README.md). Folder ini sudah
diarsipkan; wrapper training/inference perlu dikembalikan ke layout asli atau
path-nya disesuaikan sebelum dipakai. Skrip tersebut memerlukan model,
dataset JSONL, GPU, dan shell Bash; jangan menganggapnya sebagai bagian dari
pipeline adjacency Liputan6.
