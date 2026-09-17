# Notebook Liputan6

Direktori ini berisi notebook pipeline Liputan6 dan notebook diagnostik.

## Notebook aktif

| Notebook | Lingkungan | Fungsi |
| --- | --- | --- |
| `amr_parse_id2id_abdi.ipynb` | Kaggle | Parse teks Indonesia langsung menjadi AMR. |
| `amr_parse_id2id_abdi_gcp.ipynb` | GCP/local GPU | Varian parser yang sama dengan setup path untuk VM. |
| `document_pairwise_smatch.ipynb` | Lokal | Tampilkan kalimat, pasangan AMR, detail triple, dan SMATCH satu dokumen. |
| `check_extractive_vs_smatch_colab.ipynb` | Colab Enterprise | Analisis ground truth vs SMATCH secara self-contained dari file AMR `.txt` yang sudah diekstrak; tidak memerlukan import dari repository. |

Model parser aktif:

```text
abdiharyadi/mbart-en-id-smaller-indo-amr-parsing-translated-nafkhan
```

Model menerima bahasa Indonesia secara langsung. Notebook
`translate_id2en_nllb.ipynb` tidak diperlukan untuk route aktif ini.

## Notebook alternatif/parked

| Notebook | Status |
| --- | --- |
| `translate_id2en_nllb.ipynb` | Hanya untuk route concat ID+EN. |
| `amr_parse_id2en_concat_nafkhan.ipynb` | Route concat; bobot model yang diperlukan tidak dipublikasikan. |
| `temporary_external_smatch.ipynb` | Notebook eksperimen/diagnostik, bukan entry point pipeline. |

## Membuka notebook lokal

Dari root repositori:

```powershell
conda run --name generate_amr jupyter notebook notebooks/liputan6/document_pairwise_smatch.ipynb
```

Notebook analisis Colab Enterprise:

```powershell
conda run --name generate_amr jupyter notebook notebooks/liputan6/check_extractive_vs_smatch_colab.ipynb
```

Untuk VM/local GPU:

```powershell
conda run --name generate_amr jupyter notebook notebooks/liputan6/amr_parse_id2id_abdi_gcp.ipynb
```

Notebook Kaggle umumnya diunggah melalui UI Kaggle, bukan dijalankan langsung
dari path lokal.

## Setup Kaggle

Siapkan bundle modul:

```powershell
conda run --name generate_amr python scripts/prepare_kaggle_datasets.py
```

Siapkan CSV kalimat:

```powershell
conda run --name generate_amr python scripts/liputan6/liputan6_to_csv.py --mode sentence
```

Tambahkan input berikut ke notebook Kaggle:

1. `amr-code-modules`: isi `artifacts/kaggle/amr-code-modules.zip`.
2. `liputan6-data`: isi `analysis_data.csv`.
3. `liputan6-amr-graphs`: output run sebelumnya, jika melanjutkan proses.

Aktifkan GPU T4 dan internet karena checkpoint diambil dari Hugging Face pada
runtime.

## Peringatan special token

Checkpoint Abdi dan metadata tokenizer tidak konsisten pada beberapa special
token. Nilai yang pernah menyebabkan bug diam-diam adalah:

| Token/konfigurasi | Nilai yang perlu diverifikasi |
| --- | --- |
| pad | checkpoint `3`, tokenizer dapat melaporkan `1` |
| mask | checkpoint `4`, tokenizer dapat melaporkan `35107` |
| panjang vocabulary | metadata dapat melaporkan `0` vs `38025` |
| stop generation | `eos=2` tidak sama dengan `</AMR>=38024` |

Jangan menghapus pengecekan encode/decode pada section 4 dan 7b notebook
parser. Pengecekan tersebut membandingkan perilaku tokenizer dengan
`dummy_input.json` milik checkpoint dan mencegah:

- embedding matrix terinisialisasi nol;
- graph yang dapat di-decode tetapi tidak bermakna; dan
- decoding yang jauh lebih lambat karena stop token salah.

## Output parser

Setiap kalimat menghasilkan satu file:

```text
{doc_id}_{sent_idx}.txt
```

Sebelum membangun adjacency matrix, arsipkan output per split sebagai:

```text
data/liputan6/amr_graphs/archives/train.zip
data/liputan6/amr_graphs/archives/test.zip
```

Kemudian jalankan:

```powershell
conda run --name generate_amr python scripts/liputan6/build_smatch_adjacency.py --split train
```

## Aturan sampling

- Filter split secara eksplisit; urutan file global dapat menempatkan `test`
  sebelum `train`.
- Ambil dokumen utuh. Jangan menghentikan sampling di tengah dokumen.
- Pertahankan `sent_idx` asli karena adjacency dan ground truth menggunakannya
  untuk alignment.
