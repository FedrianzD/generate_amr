# generate_amr

Training configurations, differences, and measured results are recorded in
[EXPERIMENTS.md](EXPERIMENTS.md).

Pipeline riset untuk mengubah kalimat berita Indonesia menjadi graph Abstract
Meaning Representation (AMR), membangun adjacency matrix berbasis SMATCH, dan
menganalisis hubungannya dengan ground truth extractive summarization.

Fokus pipeline aktif saat ini adalah dataset **Liputan6**:

```text
JSON Liputan6
    -> tabel kalimat (CSV)
    -> AMR per kalimat
    -> adjacency matrix SMATCH per dokumen
    -> analisis ground truth vs relasi SMATCH
```

Repositori juga masih menyimpan implementasi AMRBART upstream untuk training,
evaluation, dan inference `text2amr`/`amr2text`. Bagian tersebut dipertahankan
sebagai pipeline lanjutan/legacy dan didokumentasikan terpisah.

## Entry points aktif

Enam notebook workflow aktif tersedia di [`notebooks/README.md`](notebooks/README.md).
Notebook XLSum, parser lama, route concat/translation, dan eksperimen sementara
dipindahkan ke [`archive/`](archive/README.md) agar tidak bercampur dengan pipeline
utama. Pemeriksaan pengembang ada di [`tests/`](tests/README.md).

## Status pipeline Liputan6

- Pilot historis menghasilkan 9.994 graph dari 767 dokumen; angka ini bukan
  inventaris output terbaru.
- Parser Kaggle dan GCP menyimpan SQLite, satu baris per graph kalimat.
  Atur `PREV_DB` untuk resume dari database hasil konversi atau run sebelumnya;
  `PREV_DIR` hanya untuk migrasi folder `.txt` lama.
- Adjacency matrix per dokumen berukuran `n x n`; elemen `[i, j]` adalah
  SMATCH F-score antara AMR kalimat `i` dan `j`.
- Ground truth extractive tersedia pada field `extractive_summary` dalam JSON
  canonical Liputan6.
- Implementasi pilot GCN summarization tersedia untuk eksperimen awal.
- Pilot GCN extractive summarization tersedia di
  [`notebooks/liputan6/train_gcn.ipynb`](notebooks/liputan6/train_gcn.ipynb)
  as the only active training entry point: the top 10,000 factuality-scored
  train documents, full dev validation, and held-out test evaluation.

## Quick start

Semua command Python harus dijalankan melalui environment Conda
`generate_amr`.

```powershell
conda create --name generate_amr python=3.10 pip
conda run --name generate_amr python -m pip install -r requirements.txt
```

Jika environment sudah tersedia, command pertama tidak perlu dijalankan lagi.

### 1. Siapkan dataset

Letakkan dataset canonical Liputan6 pada struktur berikut:

```text
data/liputan6/source/Liputan6/liputan6/liputan6_data/canonical/
|-- train/*.json
|-- dev/*.json
`-- test/*.json
```

Data lokal dan output hasil proses tidak disimpan di Git. Detail layout ada di
[`data/README.md`](data/README.md).

### 2. Buat tabel kalimat

```powershell
conda run --name generate_amr python scripts/liputan6/liputan6_to_csv.py --mode sentence
```

Output default:

```text
data/liputan6/processed/analysis_data.csv
```

### 3. Parse kalimat menjadi AMR

Notebook aktif untuk Liputan6 adalah:

```text
notebooks/liputan6/amr_parse_id2id_abdi_kaggle_documents.ipynb
```

Notebook tersebut ditujukan untuk Kaggle dan menggunakan model:

```text
abdiharyadi/mbart-en-id-smaller-indo-amr-parsing-translated-nafkhan
```

Model menerima teks Indonesia secara langsung; tidak memerlukan terjemahan
Inggris. Petunjuk setup, notebook alternatif, dan peringatan special token ada
di [`notebooks/liputan6/README.md`](notebooks/liputan6/README.md).

Untuk membuka notebook lokal:

```powershell
conda run --name generate_amr jupyter notebook notebooks/diagnostics/document_pairwise_smatch.ipynb
```

Untuk menjalankan analisis ground truth vs SMATCH pada data yang berada di
Colab Enterprise, gunakan
[`check_extractive_vs_smatch_colab.ipynb`](notebooks/liputan6/check_extractive_vs_smatch_colab.ipynb).
Notebook tersebut self-contained dan tidak mengimpor skrip dari repository.
Versi Colab memindai file AMR `{doc_id}_{sent_idx}.txt` langsung dari direktori,
sedangkan `extractive_summary` tetap dibaca dari JSON canonical Liputan6.

### 4. Simpan arsip graph AMR

Letakkan output parser pada:

```text
data/liputan6/amr_graphs/archives/train.zip
data/liputan6/amr_graphs/archives/test.zip
```

Arsip boleh memiliki direktori internal, selama nama file graph mengikuti
`{doc_id}_{sent_idx}.txt`.

Kontrak CLI adjacency/analisis lama di bawah memakai file per kalimat. Jangan
langsung memberikan `{doc_id}.txt` hasil konsolidasi tanpa memastikan consumer
mendukung multi-graph. Lakukan smoke test sebelum mengubah layout input.

### 5. Bangun adjacency matrix SMATCH

Train saja:

```powershell
conda run --name generate_amr python scripts/liputan6/build_smatch_adjacency.py --split train
```

Train dan test:

```powershell
conda run --name generate_amr python scripts/liputan6/build_smatch_adjacency.py --split both
```

Output default:

```text
data/liputan6/adjacency_matrices/{split}/{doc_id}.npy
```

Untuk menjalankan tahap yang sama di Kaggle tanpa mengimpor skrip repository,
gunakan [`build_smatch_adjacency_kaggle.ipynb`](notebooks/liputan6/build_smatch_adjacency_kaggle.ipynb).
Notebook tersebut mendukung input ZIP/folder `.txt`, multiprocessing CPU,
resume, validasi hasil, dan pembuatan ZIP output.

### 6. Bandingkan ground truth dengan relasi SMATCH

```powershell
conda run --name generate_amr python scripts/liputan6/check_extractive_vs_smatch.py --split train
```

Skrip melaporkan pasangan dengan SMATCH maksimum, overlap pasangan tersebut
dengan `extractive_summary`, dan kalimat dengan weighted degree tertinggi.
Output default:

```text
data/liputan6/processed/extractive_vs_smatch.csv
```

Seluruh opsi dan variasi command tersedia di
[`scripts/README.md`](scripts/README.md).

### 7. Pilot GCN sentence selection

Configure trusted train feature shards (or one `train.pt`), `dev.pt`, and
`test.pt` in the notebook. Notebook memvalidasi jumlah baris feature, adjacency, dan kalimat
sumber untuk setiap dokumen. Jalankan setelah memasang PyTorch pada environment
Conda `generate_amr` (lihat `scripts/README.md`):

```powershell
conda run --name generate_amr jupyter notebook notebooks/liputan6/train_gcn.ipynb
```

The notebook ranks `train` rows in `data/liputan6/faktualitas_lip6_v2.csv` by
`prob_yes_factuality`, excludes documents without exact canonical sentence-index
coverage in `amr_graphs/sqlite/train.sqlite`, and selects the first 10,000 complete
documents. It saves checked-candidate and exclusion reports. Missing selected
features or adjacency still stop the run. Adjacency prefers `{split}.npz` with
individual NPY files as fallback. It generates greedy ROUGE-1 F1 oracle labels, trains a
two-layer residual GCN with an MLP, predicts ranked top-2 with trigram blocking,
and selects checkpoints by dev macro ROUGE-1 F1. Position encoding is assumed
already included in the supplied 768-dimensional features, not verified.

Current oracle labels are capped at two sentences. The notebook uses PyG and writes
to `runs/pyg-gcn-top10000-factuality-oracle2/`; baseline results are retained
separately. See [the experiment record](EXPERIMENTS.md). Aktifkan
`RUN_TEST_EVALUATION` pada cell terakhir setelah konfigurasi dipilih dari
validasi. CLI training lama telah dipindahkan ke `archive/scripts/` dan tidak
lagi menjadi entry point aktif; checkpoint dense lama tidak kompatibel dengan PyG.
Setelah itu, `scripts/liputan6/evaluate_rouge.py` membandingkan kalimat yang
terpilih dengan `clean_summary` dan menulis ROUGE-1/2/L.

## Struktur direktori

```text
generate_amr/
|-- common/                    # utilitas, konfigurasi, dan token AMR
|-- model_interface/           # tokenizer dan arsitektur AMRBART
|-- scripts/
|   |-- README.md              # indeks seluruh command CLI
|   |-- prepare_kaggle_datasets.py
|   `-- liputan6/
|       |-- liputan6_to_csv.py
|       |-- build_smatch_adjacency.py
|       |-- check_extractive_vs_smatch.py
|       |-- benchmark_npy_vs_smatch.py
|       `-- evaluate_rouge.py
|-- notebooks/
|   |-- README.md
|   |-- liputan6/              # enam notebook workflow aktif
|   `-- diagnostics/           # inspeksi satu dokumen
|-- tests/                     # pemeriksaan storage dan smoke test training
|-- archive/                   # XLSum, parser lama, dan eksperimen retired
|   `-- finetune/              # training/evaluation/inference AMRBART legacy
|-- data/
|   `-- README.md              # kontrak layout data lokal
|-- artifacts/                 # bundle generated untuk Kaggle, Git-ignored
|-- HANDOVER.md                # status dan keputusan riset
|-- requirements.txt
`-- AGENTS.md                  # kebijakan environment proyek
```

Source code, notebook, dan dokumentasi dipisahkan dari data/output generated.
File besar seperti dataset, model lokal, cache, adjacency matrix, dan bundle
Kaggle tidak perlu dimasukkan ke Git.

## Indeks dokumentasi

- [`scripts/README.md`](scripts/README.md): seluruh command yang dapat dipakai.
- [`notebooks/README.md`](notebooks/README.md): indeks notebook.
- [`notebooks/liputan6/README.md`](notebooks/liputan6/README.md): pipeline AMR
  Liputan6 dan special-token gotchas.
- [`data/README.md`](data/README.md): layout input dan output lokal.
- [`archive/finetune/README.md`](archive/finetune/README.md): arsip training,
  evaluation, dan inference AMRBART.
- [`HANDOVER.md`](HANDOVER.md): status pekerjaan dan keputusan riset.

## Komponen model

Repositori mempertahankan dua task AMRBART:

| Task | Arah | Metrik utama |
| --- | --- | --- |
| AMR Parsing (`text2amr`) | Teks -> graph AMR | SMATCH |
| AMR-to-Text (`amr2text`) | Graph AMR -> teks | BLEU |

Kode model utama berada pada `model_interface/`, sedangkan training loop dan
evaluation legacy diarsipkan pada `archive/finetune/`. Pipeline parser Liputan6
dan training GCN aktif tidak bergantung pada folder tersebut.

## Batasan

- Output AMR adalah prediksi parser, bukan gold AMR Liputan6.
- Nilai tinggi dalam adjacency matrix berarti dua graph kalimat mirip secara
  semantik; nilai tersebut bukan label kepentingan kalimat.
- `extractive_summary` adalah oracle extractive berdasarkan ringkasan referensi,
  bukan skor yang tersimpan dalam adjacency matrix.
- Dataset Liputan6 hanya boleh digunakan sesuai ketentuan pemilik dataset,
  terutama untuk keperluan riset/nonkomersial.
