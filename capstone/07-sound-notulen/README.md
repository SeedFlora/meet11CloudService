# 07 · Notulen otomatis dari suara (sound, speech-to-text)

**Masalah.** Notulen rapat/kuliah ditulis manual dan sering terlambat.
**Pertanyaan.** Seberapa akurat Whisper ukuran kecil untuk bahasa Indonesia di CPU laptop,
dan berapa harga akurasi itu dalam waktu proses?

Data sumber: [LibriVox Indonesia](https://huggingface.co/datasets/indonesian-nlp/librivox-indonesia) ([CC0](https://creativecommons.org/publicdomain/zero/1.0/)).

## Langkah (venv capstone aktif, perintah sama di Windows/macOS/Linux)

```bash
cd 07-sound-notulen
python download_data.py                                  # LibriVox Indonesia bagian uji (CC0), 31 MB
python evaluate.py --model openai/whisper-tiny  --n 20   # WER + waktu proses, simpan metrics_whisper-tiny.json
python evaluate.py --model openai/whisper-base  --n 20
python evaluate.py --model openai/whisper-small --n 20
python app.py                                            # buka http://127.0.0.1:7877
```

Tidak ada pelatihan: kita mengevaluasi model pralatih secara jujur sebelum memakainya.

## Hasil di mesin uji (20 klip bahasa Indonesia ≥ 6 kata, total 96 detik audio, CPU 2 vCPU)

| Model (parameter) | WER | Waktu proses | Real-time factor |
|---|---|---|---|
| whisper-tiny (39 juta) | 55,0 % | 34 s | 0,36 |
| whisper-base (74 juta), default aplikasi | 40,1 % | 66 s | 0,69 |
| **whisper-small (244 juta)** | **30,2 %** | 215 s | 2,23 |

- WER = persen kata yang salah (substitusi + hilang + sisipan) dibanding transkrip referensi. Makin kecil makin baik.
- Waktu proses bervariasi antar-run (run ulang whisper-tiny: 38 s, RTF 0,40), WER tetap 55,0 %. Laporkan spesifikasi mesin bersama angka waktu.
- Real-time factor > 1 berarti lebih lambat dari durasi audionya: whisper-small butuh ±2 menit untuk 1 menit rekaman di CPU ini.
- Referensi LibriVox memakai ejaan lama (tj, dj, oe). `asr.py` menormalkan ejaan dan aksen sebelum WER dihitung,
  supaya model tidak dihukum karena ejaan modern.

## Tampilan web

Rekam dari mikrofon atau unggah file, atau pilih sampel data uji (WER langsung dihitung karena ada referensi).
Pilih model tiny/base/small untuk merasakan sendiri trade-off akurasi vs kecepatan. Info di bawah transkrip memisahkan
waktu **muat model** (lama hanya pada panggilan pertama) dan waktu **proses**: contoh klip 3,6 s, whisper-small muat 5,4 s +
proses 10,8 s (WER 0 %), whisper-tiny muat 1,2 s + proses 2,1 s (WER 38 %).

> **⚠ SESUAIKAN DI DEVICE-MU**
> - Port: `$env:PORT=7887; python app.py` (PowerShell) atau `PORT=7887 python app.py` (macOS/Linux).
> - **Mikrofon**: browser hanya mengizinkan mikrofon di `http://127.0.0.1`/`localhost` atau HTTPS. Bila dibuka dari
>   perangkat lain lewat IP laptop, pakai unggah file. Windows: Settings → Privacy → Microphone → izinkan browser.
>   macOS: System Settings → Privacy & Security → Microphone.
> - **Format audio**: WAV, FLAC, OGG, MP3 dibaca langsung (tanpa ffmpeg). M4A/AAC dari HP: ubah dulu ke WAV/MP3.
> - **Unduhan pertama model**: tiny ±150 MB, base ±290 MB, small ±970 MB ke cache Hugging Face.
> - **Laptop lebih cepat/GPU**: real-time factor akan jauh lebih kecil; laporkan spesifikasi mesin bersama angkanya.

## Batasan dan ide capstone

- Rekaman buku audio jelas dan satu pembicara. Rapat asli (banyak orang, bising, campur bahasa daerah) akan lebih buruk:
  rekam 10 menit rapat sendiri, buat transkrip referensi, ukur WER-nya.
- Ide: pisahkan pembicara (diarization), ringkas transkrip menjadi notulen poin-poin dengan LLM (Pertemuan 11),
  kata kunci tindak lanjut, fine-tuning Whisper dengan data rapat kampus.
