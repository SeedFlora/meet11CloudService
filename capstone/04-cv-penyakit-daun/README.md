# 04 · Deteksi penyakit daun tomat (computer vision, transfer learning)

**Masalah.** Petani sulit membedakan penyakit daun tomat sejak dini; salah diagnosis berarti salah obat.
**Pertanyaan.** Dengan ±100 foto per kelas dan tanpa GPU, seberapa akurat CNN pralatih (EfficientNet-B0)
dibandingkan fitur warna sederhana?

Data sumber: [PlantVillage, Hughes dan Salathé](https://arxiv.org/abs/1511.08060) ([CC BY-SA 3.0](https://creativecommons.org/licenses/by-sa/3.0/)), diunduh dari [mirror Hugging Face](https://huggingface.co/datasets/BrandonFors/Plant-Diseases-PlantVillage-Dataset). Gambar contoh untuk UI dibuat saat pelatihan.

## Langkah (venv capstone aktif, perintah sama di Windows/macOS/Linux)

```bash
cd 04-cv-penyakit-daun
python download_data.py   # 1 file parquet 170 MB (PlantVillage) -> maks 150 foto per kelas tomat di data/daun/
python train.py           # ekstrak fitur EfficientNet-B0 + Logistic Regression; simpan classifier.joblib, metrics.json, confusion.png
python app.py             # buka http://127.0.0.1:7874
```

## Hasil di mesin uji (10 kelas, 1.424 foto: latih 996, uji 428)

| Model | Akurasi | Macro-F1 |
|---|---|---|
| Histogram warna HSV 8×8×8 + Logistic Regression (baseline) | 0,614 | 0,602 |
| **EfficientNet-B0 ImageNet (dibekukan) + Logistic Regression** | **0,879** | **0,885** |

- Kecepatan ekstraksi fitur di CPU: 28–33 foto/detik. Tidak ada fine-tuning: `train.py` selesai ±1 menit di 2 vCPU.
- `confusion.png` menunjukkan kelas yang sering tertukar (mis. bercak yang mirip satu sama lain).

## Tampilan web

Unggah atau foto daun tomat (kamera HP bila dibuka dari HP), lihat 3 tebakan teratas, nama penyakit dalam
bahasa Indonesia, dan saran umum. Bila keyakinan < 60 %, aplikasi meminta foto ulang yang lebih jelas.
Tab **Evaluasi model** menampilkan confusion matrix.

> **⚠ SESUAIKAN DI DEVICE-MU**
> - Port: `$env:PORT=7884; python app.py` (PowerShell) atau `PORT=7884 python app.py` (macOS/Linux).
> - Unduhan pertama: parquet 170 MB + bobot EfficientNet-B0 ±21 MB ke cache Hugging Face. Butuh ±400 MB disk.
> - Buka dari HP di jaringan yang sama: jalankan dengan `server_name="0.0.0.0"` di `app.py`, lalu buka
>   `http://<IP-laptop>:7884`. Izinkan port di firewall. Kamera browser HP hanya aktif lewat HTTPS atau localhost;
>   untuk foto lewat HTTP, pakai tombol unggah dari galeri.
> - Linux: pasang torch CPU dari index PyTorch (lihat README induk).

## Batasan dan ide capstone

- PlantVillage difoto di laboratorium dengan latar polos. Foto lapangan (latar tanah, daun lain, cahaya matahari)
  hampir pasti menurunkan akurasi. Kumpulkan 30–50 foto lapangan per kelas dari kebun lokal dan laporkan
  akurasinya terpisah dari akurasi PlantVillage.
- Saran pengobatan di aplikasi bersifat umum, bukan rekomendasi agronom.
- Ide: fine-tuning beberapa lapis terakhir, deteksi lokasi bercak (YOLO), mode offline di HP (ONNX/TFLite).
