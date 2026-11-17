"""Fitur + model anomali. Dipakai server (main.py) dan evaluasi_offline.py supaya yang diuji sama dengan yang dijalankan.

Fitur dibuat "stasioner": selisih terhadap median 60 detik dan perubahan 10 detik, jadi naik-turun suhu
siang-malam yang wajar tidak dianggap aneh, tetapi perubahan mendadak (kipas mati, pompa bocor) menonjol.

Model: EllipticEnvelope = jarak Mahalanobis ke pola normal. Jaraknya terus membesar makin jauh data dari pola
normal. (Isolation Forest sempat dicoba: skornya "jenuh" di luar rentang data latih, sehingga suhu 85 °C
tidak tampak lebih aneh daripada titik normal paling ekstrem. Angkanya ada di evaluasi_offline.py.)
"""
import csv

import numpy as np
from sklearn.covariance import EllipticEnvelope

KOLOM = ("suhu", "kelembapan", "tanah")
NAMA_FITUR = ["suhu - median 60 s", "perubahan suhu 10 s", "kelembapan - median 60 s", "perubahan tanah 10 s"]
JENDELA = 60  # data pertama baru punya fitur lengkap setelah 60 detik


def fitur(seq, i):
    r = seq[i]
    lalu60 = seq[max(0, i - JENDELA):i + 1]
    p10 = seq[max(0, i - 10)]
    return [r["suhu"] - float(np.median([x["suhu"] for x in lalu60])),
            r["suhu"] - p10["suhu"],
            r["kelembapan"] - float(np.median([x["kelembapan"] for x in lalu60])),
            r["tanah"] - p10["tanah"]]


def fitur_semua(seq):
    return np.array([fitur(seq, i) for i in range(len(seq))])


class Detektor:
    """skor = jarak Mahalanobis² / ambang. Ambang = persentil 99,9 jarak pada data latih normal, jadi skor > 1 = anomali."""

    metode = "EllipticEnvelope (jarak Mahalanobis)"

    def __init__(self, persentil=99.9):
        self.persentil = persentil

    def latih(self, seq):
        X = fitur_semua(seq)[JENDELA:]
        self.model = EllipticEnvelope(random_state=42).fit(X)
        self.ambang = float(np.percentile(self.model.mahalanobis(X), self.persentil))
        self.n_latih = len(X)
        return self

    def skor_banyak(self, X):
        return self.model.mahalanobis(np.asarray(X)) / self.ambang

    def skor(self, x):
        return float(self.skor_banyak([x])[0])


class Alarm:
    """Nyala setelah `nyala` titik anomali berturut-turut (debounce), padam setelah `padam` titik normal berturut-turut.

    Tanpa histeresis, alarm berkedip on/off saat skor naik-turun di sekitar ambang.
    """

    def __init__(self, nyala=3, padam=10):
        self.nyala, self.padam = nyala, padam
        self.n_anomali = self.n_normal = 0
        self.aktif = False

    def update(self, anomali):
        if anomali:
            self.n_anomali, self.n_normal = self.n_anomali + 1, 0
        else:
            self.n_anomali, self.n_normal = 0, self.n_normal + 1
        if not self.aktif and self.n_anomali >= self.nyala:
            self.aktif = True
        elif self.aktif and self.n_normal >= self.padam:
            self.aktif = False
        return self.aktif


def baca_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        return [{k: float(r[k]) for k in KOLOM} for r in csv.DictReader(f)]


def tulis_csv(path, seq):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=KOLOM, extrasaction="ignore")
        w.writeheader()
        w.writerows(seq)
