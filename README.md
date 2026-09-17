# generate_amr

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

## Status pipeline Liputan6

- Parsing AMR train yang ditugaskan telah menghasilkan 9.994 graph dari 767
  dokumen.
- Satu graph disimpan untuk setiap kalimat dengan nama
  `{doc_id}_{sent_idx}.txt`.
- Adjacency matrix per dokumen berukuran `n x n`; elemen `[i, j]` adalah
  SMATCH F-score antara AMR kalimat `i` dan `j`.
- Ground truth extractive tersedia pada field `extractive_summary` dalam JSON
  canonical Liputan6.
- Implementasi GCN summarization belum menjadi bagian dari repositori ini.

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
notebooks/liputan6/amr_parse_id2id_abdi.ipynb
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
conda run --name generate_amr jupyter notebook notebooks/liputan6/document_pairwise_smatch.ipynb
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

## Struktur direktori

```text
generate_amr/
|-- common/                    # utilitas, konfigurasi, dan token AMR
|-- model_interface/           # tokenizer dan arsitektur AMRBART
|-- finetune/                  # training/evaluation/inference AMRBART
|   `-- README.md              # command dan catatan pipeline legacy
|-- scripts/
|   |-- README.md              # indeks seluruh command CLI
|   |-- prepare_kaggle_datasets.py
|   `-- liputan6/
|       |-- liputan6_to_csv.py
|       |-- build_smatch_adjacency.py
|       `-- check_extractive_vs_smatch.py
|-- notebooks/
|   |-- README.md
|   |-- liputan6/              # pipeline aktif dan notebook diagnostik
|   `-- xlsum/                 # notebook referensi upstream
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
- [`finetune/README.md`](finetune/README.md): training, evaluation, dan inference
  AMRBART.
- [`HANDOVER.md`](HANDOVER.md): status pekerjaan dan keputusan riset.

## Komponen model

Repositori mempertahankan dua task AMRBART:

| Task | Arah | Metrik utama |
| --- | --- | --- |
| AMR Parsing (`text2amr`) | Teks -> graph AMR | SMATCH |
| AMR-to-Text (`amr2text`) | Graph AMR -> teks | BLEU |

Kode model utama berada pada `model_interface/`, sedangkan training loop dan
evaluation berada pada `finetune/`.

## Batasan

- Output AMR adalah prediksi parser, bukan gold AMR Liputan6.
- Nilai tinggi dalam adjacency matrix berarti dua graph kalimat mirip secara
  semantik; nilai tersebut bukan label kepentingan kalimat.
- `extractive_summary` adalah oracle extractive berdasarkan ringkasan referensi,
  bukan skor yang tersimpan dalam adjacency matrix.
- Dataset Liputan6 hanya boleh digunakan sesuai ketentuan pemilik dataset,
  terutama untuk keperluan riset/nonkomersial.
