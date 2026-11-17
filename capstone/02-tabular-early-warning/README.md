# 02 · Early warning siswa berisiko (ML data tabular)

**Masalah.** Wali kelas baru tahu siswa tidak lulus setelah nilai akhir keluar.
**Pertanyaan.** Setelah nilai periode 1 (G1) keluar, bisakah model menandai siswa yang berisiko nilai akhir
G3 < 10, lebih baik daripada aturan sederhana "G1 < 10"?

Data sumber: [UCI Student Performance, Cortez dan Silva](https://archive.ics.uci.edu/dataset/320/student%2Bperformance) ([CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)).

## Langkah (venv capstone aktif, perintah sama di Windows/macOS/Linux)

```bash
cd 02-tabular-early-warning
python download_data.py      # UCI Student Performance (CC BY 4.0), 395 siswa matematika, 33 kolom
python train.py              # latih + evaluasi, simpan model.joblib, metrics.json, evaluasi.png
python app.py                # buka http://127.0.0.1:7872
```

## Hasil di mesin uji (25 % data uji = 99 siswa, 33 berisiko)

| Model | Akurasi | Recall berisiko | Presisi berisiko | F1 berisiko |
|---|---|---|---|---|
| Tebak mayoritas (semua lulus) | 0,667 | 0,000 | 0,000 | 0,000 |
| Aturan G1 < 10 | 0,818 | 0,758 | 0,714 | 0,735 |
| **Random Forest** (400 pohon, class_weight balanced) | 0,818 | **0,818** | 0,692 | **0,750** |

- CV 5-fold F1 pada data latih: 0,769 ± 0,051.
- Fitur terpenting (permutation importance): G1 jauh di atas yang lain (0,378), lalu dukungan sekolah dan keluarga.
- Artinya jujur: aturan G1 < 10 sudah kuat. Random Forest menangkap 27 dari 33 siswa berisiko (aturan: 25),
  dengan alarm palsu 12 siswa (aturan: 10). Untuk peringatan dini konseling, recall lebih penting.
- G2 dan G3 sengaja dibuang dari fitur: saat peringatan dini dibuat, nilai itu belum ada (mencegah kebocoran data).

## Tampilan web

Tab **Prediksi**: isi 10 data penting siswa (nilai G1, absen, waktu belajar, dukungan, dll.), klik contoh,
dan lihat probabilitas berisiko. Kolom lain diisi nilai tipikal dari data. Tab **Evaluasi model**: grafik F1 vs
baseline dan fitur terpenting.

> **⚠ SESUAIKAN DI DEVICE-MU**
> - Port: `$env:PORT=7882; python app.py` (PowerShell) atau `PORT=7882 python app.py` (macOS/Linux).
> - Data sekolahmu sendiri: kolom harus disesuaikan di `train.py` (target, kolom yang dibuang) dan input di `app.py`.
>   Data siswa adalah data pribadi: pakai anonim/izin sekolah, jangan unggah ke repo publik.

## Batasan dan ide capstone

- 395 siswa dari 2 sekolah menengah di Portugal (Cortez & Silva, 2008): perlu diuji ulang di data lokal sebelum dipakai.
- Cek keadilan: bandingkan recall per kelompok (jenis kelamin, alamat desa/kota) sebelum dipakai untuk keputusan.
- Ide: dashboard wali kelas per kelas, penjelasan per siswa (SHAP), kirim pengingat ke konselor.
