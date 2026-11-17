# 06 · Prakiraan suhu per jam Kota Bandung (time series)

**Masalah.** Jadwal penyiraman, kegiatan luar ruang, dan beban AC bergantung pada suhu besok.
**Pertanyaan.** Apakah model ML bisa mengalahkan baseline "suhu kemarin di jam yang sama" untuk prakiraan 24 jam?

## Langkah (venv capstone aktif, perintah sama di Windows/macOS/Linux)

```bash
cd 06-timeseries-suhu-bandung
python download_data.py   # Open-Meteo Archive (gratis, tanpa API key, CC BY 4.0), per jam sejak 2023-01-01
python train.py           # latih + evaluasi; simpan model.joblib, metrics.json, prediksi_uji.csv, evaluasi.png
python app.py             # buka http://127.0.0.1:7876
```

## Desain evaluasi

- Prakiraan dibuat pukul 23.00 untuk jam 00.00–23.00 esok hari, hanya memakai data sampai 23.00 (tanpa bocoran masa depan).
- Uji = **120 hari terakhir**, latih = 1.282 hari sebelumnya (pembagian menurut waktu, bukan acak).
- Model: HistGradientBoosting memprediksi **koreksi** terhadap baseline musiman, dari fitur 24 jam terakhir
  (suhu, perubahan, kelembapan, hujan, maks/min).

## Hasil di mesin uji (MAE dalam °C, makin kecil makin baik)

| Metode | MAE jam 1–6 | MAE jam 1–24 |
|---|---|---|
| Persistensi (suhu 23.00 dipakai untuk semua jam) | 1,998 | 3,692 |
| Musiman: suhu kemarin di jam yang sama | 0,767 | 0,833 |
| **Model koreksi musiman** | **0,550** | **0,802** |

Model unggul jelas di jam-jam awal (1–8) dan sedikit kalah dari baseline musiman di horizon siang hari.
Tidak semua hari lebih baik: contoh 2026-10-05 MAE model 1,57 vs baseline 0,71 (cuaca berubah mendadak).
Hari seperti ini justru bagus untuk dibahas di laporan.

## Tampilan web

Pilih tanggal di periode uji: grafik prakiraan model vs baseline vs suhu aktual, MAE hari itu, dan tabel ringkasan.

> **⚠ SESUAIKAN DI DEVICE-MU**
> - Port: `$env:PORT=7886; python app.py` (PowerShell) atau `PORT=7886 python app.py` (macOS/Linux).
> - Kota lain: ubah `LAT, LON, KOTA` di `download_data.py` (koordinat dari Google Maps), lalu ulangi download dan train.
> - `HTTP 429 Too Many Requests` dari Open-Meteo: skrip menunggu dan mencoba ulang otomatis; bila tetap gagal, ulangi beberapa menit lagi.
> - Tanggal akhir data = hari ini − 7 hari, jadi angka MAE-mu akan sedikit berbeda dari tabel di atas.

## Batasan dan ide capstone

- Data Open-Meteo adalah reanalisis (model cuaca), bukan sensor di lokasimu. Bandingkan dengan sensor sendiri
  (mis. ESP32 proyek 01) bila ingin prakiraan mikro-iklim.
- Ide: prakiraan hujan (klasifikasi), gabungkan prakiraan cuaca numerik sebagai fitur, jadwal penyiraman otomatis.
