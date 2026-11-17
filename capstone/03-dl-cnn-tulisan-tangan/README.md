# 03 · Pengenal angka tulisan tangan (deep learning, CNN)

**Masalah.** Formulir kertas (nilai, nomor antrean, meteran) masih diketik ulang manual.
**Pertanyaan.** Seberapa jauh CNN kecil yang dilatih di CPU mengalahkan model linear pada angka tulisan tangan?

Data sumber: [MNIST, Yann LeCun dan Corinna Cortes](https://keras.io/api/datasets/mnist/#license) ([CC BY-SA 3.0](https://creativecommons.org/licenses/by-sa/3.0/)). Gambar contoh untuk UI dibuat saat pelatihan.

## Langkah (venv capstone aktif, perintah sama di Windows/macOS/Linux)

```bash
cd 03-dl-cnn-tulisan-tangan
python train.py      # unduh MNIST (parquet, 18 MB) + latih CNN 3 epoch + baseline; simpan model.pt, metrics.json, kurva.png
python app.py        # buka http://127.0.0.1:7873
```

## Hasil di mesin uji (10.000 gambar uji MNIST, CPU 2 vCPU)

| Model | Akurasi uji |
|---|---|
| Logistic Regression (piksel mentah, 20.000 data latih) | 91,71 % |
| **CNN** 2 konvolusi + 2 dense, 105.866 parameter, epoch 1 / 2 / 3 | 96,83 % / 98,29 % / **98,33 %** |

Tiap epoch sekitar 10 detik di CPU. Kurva loss dan akurasi tersimpan di `kurva.png`, contoh yang salah di `metrics.json`.

## Tampilan web

- **Gambar sendiri**: klik ikon kuas (Brush) dulu, gambar satu angka 0–9 dengan mouse/jari, klik Tebak.
  Gambar dipotong, diposisikan di tengah, dan dikecilkan ke 28×28 seperti MNIST sebelum masuk CNN.
- **Contoh data uji**: angka asli dari MNIST beserta tebakan model.
- **Kurva training**: grafik dari `kurva.png`.

> **⚠ SESUAIKAN DI DEVICE-MU**
> - Port: `$env:PORT=7883; python app.py` (PowerShell) atau `PORT=7883 python app.py` (macOS/Linux).
> - Laptop dengan GPU NVIDIA: kode ini sengaja tetap di CPU (tensor tidak dipindah ke GPU) agar sama di semua laptop.
>   Boleh tambahkan `.to("cuda")` pada model dan batch bila torch versi CUDA terpasang.
> - Linux: pasang torch CPU dari index PyTorch (lihat README induk) agar tidak mengunduh versi CUDA.

## Batasan dan ide capstone

- MNIST = angka Amerika yang sudah bersih dan terpusat. Tulisan tangan formulir Indonesia (angka 1 dan 7 berbeda gaya)
  perlu data sendiri: kumpulkan ±50 contoh per angka dari teman sekelas, latih ulang, bandingkan akurasinya.
- Ide: baca kolom nilai dari foto lembar jawaban (deteksi kotak + CNN per digit), validasi NIM, ekspor ke spreadsheet.
