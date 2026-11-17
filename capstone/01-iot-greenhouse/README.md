# 01 · Smart Greenhouse: IoT + deteksi anomali + prakiraan

**Masalah.** Kipas greenhouse mati atau pompa bocor biasanya baru ketahuan saat suhu/tanah sudah melewati ambang.
**Pertanyaan.** Bisakah model yang belajar dari pola normal memberi alarm lebih awal daripada aturan ambang,
tanpa banyak alarm palsu?

```
ESP32 (virtual atau asli) --MQTT--> Mosquitto --> FastAPI (fitur + model anomali + prakiraan) --> dashboard web
   suhu, kelembapan, tanah          broker          detektor.py, main.py                         http://localhost:8070
```

## Jalankan (sama di Windows, macOS, Linux)

Prasyarat: Docker Desktop (Windows/macOS) atau Docker Engine + Compose plugin (Linux).

```bash
cd capstone/01-iot-greenhouse
docker compose up -d --build
docker compose ps                 # 3 service: broker, device, app (app "healthy")
```

Buka **http://localhost:8070**. Model langsung siap karena dilatih dari `app/riwayat/normal_10menit.csv`
(600 data normal = 1 siklus siang-malam simulasi). Skor anomali mulai muncul setelah 60 detik data pertama.

Coba tombol: **Kipas mati** → tunggu alarm ML (oranye) lalu alarm aturan (merah) → **Normal** → tunggu tenang →
**Pompa bocor** → **Normal** → **Sensor rusak** → **Normal**. Tabel "Waktu deteksi" mencatat detik ML vs aturan.
Satu kali uji live di mesin uji: kipas mati ML 15 s vs aturan 47 s, pompa bocor 4 s vs 45 s, sensor rusak 3 s vs 1 s.

Matikan: `docker compose down`.

## Evaluasi tanpa Docker (model diuji sebelum dipakai)

Dengan venv capstone aktif (lihat README induk):

```bash
python evaluasi_offline.py
```

Skrip ini memakai **kode yang sama** dengan server (`app/detektor.py`, `device/sim.py`) pada 20 simulasi
(9,7 jam kondisi normal + tiap gangguan 20 kali, fase siang/malam acak). Hasil di mesin uji:

| Metode | Alarm palsu/jam | Kipas mati: median (p90) | Pompa bocor | Sensor rusak |
|---|---|---|---|---|
| **ML: EllipticEnvelope** (dipakai) | 0,00 | **13,5 s** (20 s), 0 terlewat | **6 s**, 0 terlewat | 6 s, 0 terlewat |
| Isolation Forest (dicoba) | 0,00 | 23,5 s (79 s), 2 terlewat | 41 s, **14 terlewat** | 8 s, 8 terlewat |
| Aturan ambang (baseline) | 0,00 | 57 s (90 s) | 45 s | **1 s** |

Pelajaran: Isolation Forest populer di tutorial, tetapi skornya **jenuh** di luar rentang data latih
(suhu 85 °C tidak tampak lebih aneh daripada titik normal paling ekstrem), sehingga banyak gangguan terlewat.
Jarak Mahalanobis (EllipticEnvelope) terus membesar makin jauh data dari pola normal. Untuk sensor macet di 85 °C,
aturan ambang sederhana justru lebih cepat; ML unggul pada gangguan yang **bertahap**.

## Uji validasi telemetri

Dengan venv capstone aktif:

```bash
python uji_ketahanan.py
```

Tes ini memeriksa validasi data sensor dan pengisian waktu server melalui fungsi aplikasi.

## Fitur dan model (app/detektor.py)

- Fitur "stasioner": suhu − median 60 s, perubahan suhu 10 s, kelembapan − median 60 s, perubahan tanah 10 s.
  Siklus siang-malam yang wajar tidak memicu alarm, perubahan mendadak menonjol.
- Model: EllipticEnvelope, ambang = persentil 99,9 jarak pada data latih (skor > 1 berarti anomali).
- Alarm ML menyala setelah 3 titik anomali berturut-turut (debounce) dan padam setelah 10 titik normal (histeresis), jadi tidak berkedip.
- Prakiraan: regresi linear 30 detik terakhir → suhu 60 detik lagi, ETA melewati 35 °C bila naik > 3 °C/menit.
  Prakiraan **ditahan** selama alarm ML aktif: saat uji, sensor macet 85 °C membuat regresi memprediksi 201 °C.
- Payload divalidasi sebelum masuk riwayat: nilai teks, NaN, kolom hilang, atau bukan JSON ditolak dan dihitung di
  `/health` (`pesan_ditolak`) dan `/metrics`.
- ESP32 yang jamnya belum sinkron (ts kosong/0) diberi waktu server.
- `/metrics` berformat Prometheus, bisa ditambahkan ke stack monitoring Pertemuan 10.

## Pakai ESP32 asli (folder esp32/)

1. Buka `esp32/greenhouse_esp32.ino` di Arduino IDE, pasang library PubSubClient + DHT sensor library.
2. Isi Wi-Fi, **IP laptop** yang menjalankan Docker, dan kalibrasi sensor tanah (lihat blok SESUAIKAN di file).
3. Matikan simulator dengan `docker compose stop device`. Atur `MQTT_BIND_IP` di file `.env` Compose ke IP LAN laptop, lalu jalankan `docker compose up -d broker app`. Gunakan jaringan privat dan atur firewall agar hanya perangkat uji dapat mencapai port broker.
4. Biarkan ESP32 asli berjalan normal ±10 menit, unduh `http://localhost:8070/api/export.csv`,
   simpan sebagai `app/riwayat/normal_10menit.csv`, lalu `docker compose up -d --build app`.
   Model dilatih ulang dari pola normal greenhouse-mu sendiri.

Sketch ESP32 belum dikompilasi dan belum diuji di perangkat fisik oleh penyusun; yang diuji end-to-end adalah ESP32 virtual.

> **⚠ SESUAIKAN DI DEVICE-MU**
> - **Port bentrok** (sudah ada Mosquitto/aplikasi lain): atur `MQTT_HOST_PORT=1884` atau `GREENHOUSE_PORT=8071` di file `.env` Compose; gunakan port baru itu pada klien atau browser.
> - **ESP32 asli**: `MQTT_HOST` = IP laptop di jaringan yang sama (`ipconfig` di Windows,
>   `ipconfig getifaddr en0` di macOS, `hostname -I` di Linux), bukan `localhost`. Izinkan port 1883 di firewall
>   (Windows Defender Firewall akan bertanya saat pertama kali). Wi-Fi kampus sering memblokir antar-perangkat:
>   pakai hotspot HP.
> - **Mac Apple Silicon / Windows ARM**: image `python:3.12-slim` dan `eclipse-mosquitto:2.0` tersedia untuk ARM, tidak perlu diubah.
> - **Tanpa Docker**: jalankan broker Mosquitto lokal, lalu `pip install -r app/requirements.txt`,
>   `cd app && uvicorn main:app --port 8070` dan di terminal lain `cd device && python sim.py`.
> - **Riwayat normal tidak ada**: server melatih model dari `WARMUP` detik data live (default 600).
> - **Dua server sekaligus** (mis. versi lama dan baru): `client_id` MQTT di `app/main.py` harus berbeda,
>   kalau sama broker memutus salah satunya bergantian.

## Batasan dan ide capstone

- Data sensor di sini simulasi; dinamika greenhouse asli (cuaca, penyiraman manual) perlu direkam dan dievaluasi ulang.
- Belum ada autentikasi MQTT/TLS. Untuk deployment nyata: user/password Mosquitto, TLS, dan jaringan terpisah.
- Ide: kendali aktuator (kipas/pompa) otomatis dari alarm, notifikasi Telegram, simpan data ke TimescaleDB,
  dashboard Grafana dari `/metrics`, beberapa greenhouse (DEVICE_ID berbeda).
