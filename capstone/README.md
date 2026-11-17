# Capstone AI: 7 proyek contoh dari data sampai web (Pertemuan 12–13)

Tujuh contoh ini memperlihatkan cara mengolah data, mengevaluasi model **terhadap baseline**, dan mendemokan hasilnya di browser. Untuk tugas kuliah, ikuti [panduan proyek kelompok](../PROYEK_KELOMPOK.md), termasuk deployment dan bukti latensi p50/p95, rasio error, serta estimasi biaya sesuai jalur proyek.
Semua angka di bawah berasal dari run di mesin uji (Linux, 2 vCPU, RAM 7 GB, **tanpa GPU**).
Ulangi evaluasi di laptopmu dan laporkan perubahan hasil, termasuk bila baseline unggul.

| No | Proyek | Jenis AI | Data (lisensi) | Hasil di data uji (vs baseline) | Web |
|----|--------|----------|----------------|----------------------------------|-----|
| 01 | [Smart Greenhouse](01-iot-greenhouse/README.md) | IoT + deteksi anomali | sensor ESP32 virtual via MQTT | waktu deteksi kipas mati 13,5 s vs aturan ambang 57 s (median 20 simulasi), 0 alarm palsu/jam | http://localhost:8070 |
| 02 | [Early warning siswa](02-tabular-early-warning/README.md) | ML data tabular | [UCI Student Performance](https://archive.ics.uci.edu/dataset/320/student%2Bperformance) (CC BY 4.0) | F1 kelas berisiko 0,750 vs aturan G1<10 0,735 | http://127.0.0.1:7872 |
| 03 | [Tulisan tangan](03-dl-cnn-tulisan-tangan/README.md) | Deep learning (CNN) | [MNIST](https://keras.io/api/datasets/mnist/#license) (CC BY-SA 3.0) | akurasi 98,33 % vs Logistic Regression 91,71 % | http://127.0.0.1:7873 |
| 04 | [Penyakit daun tomat](04-cv-penyakit-daun/README.md) | Computer vision | [PlantVillage](https://arxiv.org/abs/1511.08060) (CC BY-SA 3.0) | macro-F1 0,885 vs histogram warna 0,602 | http://127.0.0.1:7874 |
| 05 | [Sentimen ulasan](05-nlp-sentimen-ulasan/README.md) | NLP | [IndoNLU SmSA](https://github.com/IndoNLP/indonlu) | macro-F1 0,851 vs tebak mayoritas 0,246 | http://127.0.0.1:7875 |
| 06 | [Prakiraan suhu Bandung](06-timeseries-suhu-bandung/README.md) | Time series | [Open-Meteo](https://open-meteo.com/en/licence) (CC BY 4.0) | MAE jam 1–6: 0,55 °C vs "kemarin jam sama" 0,77 °C | http://127.0.0.1:7876 |
| 07 | [Notulen suara](07-sound-notulen/README.md) | Sound (speech-to-text) | [LibriVox Indonesia](https://huggingface.co/datasets/indonesian-nlp/librivox-indonesia) (CC0) | WER whisper-small 30,2 % vs whisper-tiny 55,0 % | http://127.0.0.1:7877 |

Detail tiap proyek (langkah, hasil, batasan, ide pengembangan) ada di README masing-masing folder.
File metrik dan grafik yang disertakan dapat digunakan untuk mencocokkan hasil percobaanmu.

## Langkah 0: siapkan sekali untuk proyek 02–07

**Windows (PowerShell)**

```powershell
cd capstone
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\Activate.ps1
```

**macOS / Linux (Terminal)**

```bash
cd capstone
python3.12 -m venv .venv          # atau python3 bila 3.12 adalah versi default
source .venv/bin/activate
python -m pip install --upgrade pip
# Linux tanpa GPU: pasang torch CPU dulu supaya tidak mengunduh versi CUDA 2+ GB
python -m pip install torch==2.15.0 torchvision==0.29.2 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
```

Setelah venv aktif, perintah `python` sama di semua OS. Proyek 01 (IoT) memakai Docker, lihat
`01-iot-greenhouse/README.md`.

> **⚠ SESUAIKAN DI DEVICE-MU**
> - **Python**: diuji 3.12. Gunakan Python 3.12 untuk versi paket di sini. `py -3.12` hanya ada di Windows.
> - **PowerShell menolak `Activate.ps1`**: jalankan Python dari venv secara langsung, misalnya `..\.venv\Scripts\python.exe train.py` dari folder proyek.
> - **Proxy kampus/kantor**: set dulu `$env:HTTPS_PROXY="http://proxy:port"` (PowerShell) atau
>   `export HTTPS_PROXY=http://proxy:port` (macOS/Linux) sebelum `pip install` dan `download_data.py`.
> - **Port bentrok**: app Gradio proyek 02–07 membaca variabel `PORT`.
>   PowerShell `$env:PORT=7882; python app.py` · macOS/Linux `PORT=7882 python app.py`.
> - **Ruang disk**: siapkan beberapa GB untuk paket Python, data, dan model yang dipilih. Model Hugging Face disimpan di
>   `~/.cache/huggingface` (Windows: `C:\Users\<nama>\.cache\huggingface`); pindahkan dengan variabel `HF_HOME`.
> - **GPU**: tidak wajib. Semua proyek dirancang selesai di CPU dalam hitungan menit.

## Pola 0 → 100 di setiap proyek

1. **Masalah & pertanyaan**: siapa penggunanya, keputusan apa yang dibantu, metrik apa yang dipakai.
2. **Data**: `python download_data.py` untuk proyek 02, 04–07. Proyek 01 memakai data simulasi dan proyek 03 mengunduh MNIST saat pelatihan.
3. **Latih + evaluasi**: `python train.py` untuk proyek 02–06, `python evaluasi_offline.py` untuk proyek 01, atau `python evaluate.py` untuk proyek 07. Bandingkan metrik model dan baseline.
4. **Web UI**: `python app.py`, buka alamat di tabel atas, uji dengan contoh yang disediakan dan data milikmu sendiri.
5. **Laporan & demo**: salin angka dari `metrics.json`, tulis batasan (data apa yang belum diuji), lalu
   gladi demo dengan langkah di slide Pertemuan 13.

## Tips menjadikannya capstone

- Ganti datanya dengan data nyata dari mitra (sekolah, UMKM, kebun, kampus) dan ulangi evaluasi vs baseline.
- Tambahkan satu hal yang tidak ada di contoh: monitoring (Pertemuan 10), RAG/LLM (Pertemuan 11),
  deploy cloud (Pertemuan 6–8), atau perangkat fisik (ESP32).
- Catat kegagalan model (contoh salah tebak) dan jelaskan sebabnya. Analisis kesalahan sering lebih bernilai
  daripada menaikkan akurasi 1 %.
