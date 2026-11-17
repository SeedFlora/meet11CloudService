# 05 · Analisis sentimen ulasan berbahasa Indonesia (NLP)

**Masalah.** Pemilik UMKM punya ratusan ulasan pelanggan tetapi tidak sempat membaca semuanya.
**Pertanyaan.** Seberapa akurat model ringan mengelompokkan ulasan menjadi positif, netral, negatif,
dan apakah model embedding yang lebih berat memang lebih baik?

## Langkah (venv capstone aktif, perintah sama di Windows/macOS/Linux)

```bash
cd 05-nlp-sentimen-ulasan
python download_data.py          # IndoNLU SmSA: 11.000 ulasan latih + 1.260 validasi (dipakai sebagai uji)
python train.py                  # TF-IDF + Logistic Regression, < 1 menit; simpan model.joblib, metrics.json
python train.py --embedding      # opsional: + embedding multibahasa (paraphrase-multilingual-MiniLM-L12-v2), ±3 menit di CPU
python app.py                    # buka http://127.0.0.1:7875
```

## Hasil di mesin uji (1.260 ulasan uji)

| Model | Macro-F1 | Waktu latih |
|---|---|---|
| Tebak mayoritas (semua positif) | 0,246 | - |
| **TF-IDF unigram+bigram + Logistic Regression** | **0,851** | 1,3 s |
| Embedding MiniLM multibahasa + Logistic Regression | 0,817 | 190 s |

- F1 per kelas (TF-IDF): negatif 0,860 · netral 0,766 · positif 0,927. Kelas netral paling sulit.
- Model yang lebih berat tidak otomatis lebih baik: pada data ini kata kunci ("mengecewakan", "mantap") sudah sangat informatif.

## Tampilan web

- **Dashboard banyak ulasan**: tempel banyak ulasan (satu per baris), lihat grafik jumlah per sentimen dan tabel.
- **Analisis 1 ulasan**: label, probabilitas, dan kata yang paling mendorong keputusan model (bisa dijelaskan ke pengguna).
- **Evaluasi model**: angka di atas.

Coba juga ulasan bernada sarkas atau campur bahasa daerah: di situ model masih sering salah. Kumpulkan contoh salahnya
untuk bahan analisis kesalahan di laporan.

> **⚠ SESUAIKAN DI DEVICE-MU**
> - Port: `$env:PORT=7885; python app.py` (PowerShell) atau `PORT=7885 python app.py` (macOS/Linux).
> - `--embedding` mengunduh model ±470 MB ke cache Hugging Face (`HF_HOME` untuk memindahkan lokasi).
> - Data ulasan mitra: simpan sebagai TSV dua kolom (teks, label) seperti format SmSA, lalu ubah path di `train.py`.

## Batasan dan ide capstone

- SmSA berisi ulasan umum (makanan, tempat, layanan). Domain lain (aplikasi, politik, kesehatan) perlu label sendiri.
- Ide: aspek apa yang dikeluhkan (aspect-based), tren sentimen per minggu, balasan otomatis draf via LLM (Pertemuan 11),
  ambil ulasan dari Google Maps/marketplace dengan izin.
