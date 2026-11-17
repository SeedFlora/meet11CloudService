"""Uji model anomali TANPA Docker/broker: berapa alarm palsu per jam dan berapa detik sampai gangguan terdeteksi.

Jalankan dari folder ini (venv capstone aktif):
    python evaluasi_offline.py              # pakai app/riwayat/normal_10menit.csv (dibuat otomatis bila belum ada)
    python evaluasi_offline.py --csv rekaman_greenhouse.csv   # uji riwayat normal dari perangkatmu sendiri

Pembanding:
- ML  : detektor.py (EllipticEnvelope, ambang persentil 99,9 data latih, alarm nyala 3 / padam 10) = yang dipakai server
- IF  : Isolation Forest dengan ambang dan aturan alarm yang sama (alasan tidak dipakai)
- Aturan: suhu > 35 °C atau tanah > 85 %
"""
import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
from sklearn.ensemble import IsolationForest

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE / "app"), str(HERE / "device")]
from detektor import JENDELA, Alarm, Detektor, baca_csv, fitur_semua, tulis_csv  # noqa: E402
from sim import Greenhouse  # noqa: E402

GANGGUAN = ["kipas_mati", "pompa_bocor", "sensor_rusak"]
LAMA_GANGGUAN, LAMA_NORMAL = 180, 1800


def rekam(gh, rencana):
    seq = []
    for durasi, sk in rencana:
        seq += [gh.langkah(sk) for _ in range(durasi)]
    return seq


def alarm_seri(flag):
    al = Alarm(nyala=3, padam=10)
    return np.array([al.update(bool(f)) for f in flag])


def naik(alarm):
    """jumlah kali alarm berubah dari mati ke menyala"""
    return int(alarm[0]) + int(np.sum(alarm[1:] & ~alarm[:-1]))


def ringkas(d, n):
    return {"median_s": float(np.median(d)) if d else None, "p90_s": round(float(np.quantile(d, 0.9)), 1) if d else None,
            "terlewat": n - len(d)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=str(HERE / "app" / "riwayat" / "normal_10menit.csv"))
    ap.add_argument("--uji", type=int, default=20, help="jumlah simulasi uji (tiap simulasi 30 menit normal + 3 gangguan)")
    a = ap.parse_args()

    path = Path(a.csv)
    if not path.exists():  # rekam 1 siklus siang-malam (600 s) kondisi normal dari simulator
        path.parent.mkdir(parents=True, exist_ok=True)
        tulis_csv(path, rekam(Greenhouse(seed=999, t0=random.Random(7).randrange(600)), [(600, "normal")]))
        print("riwayat normal dibuat:", path.relative_to(HERE) if path.is_relative_to(HERE) else path)
    latih = baca_csv(path)
    det = Detektor().latih(latih)
    X_latih = fitur_semua(latih)[JENDELA:]
    iforest = IsolationForest(n_estimators=200, random_state=42).fit(X_latih)
    ambang_if = float(np.percentile(-iforest.score_samples(X_latih), 99.9))
    print(f"latih: {len(latih)} data normal -> {det.n_latih} vektor fitur, ambang Mahalanobis² {det.ambang:.2f}")

    hasil = {k: {g: [] for g in GANGGUAN} for k in ("ml", "if", "aturan")}
    palsu = {"ml": 0, "if": 0, "aturan": 0}
    for s in range(a.uji):
        r = random.Random(1000 + s)
        rencana, mulai = [(LAMA_NORMAL, "normal")], {}
        for g in GANGGUAN:
            rencana.append((200 + r.randrange(300), "normal"))      # pulih penuh + fase hari acak
            mulai[g] = sum(d for d, _ in rencana)
            rencana.append((LAMA_GANGGUAN, g))
        seq = rekam(Greenhouse(seed=s, t0=r.randrange(600)), rencana + [(200, "normal")])
        X = fitur_semua(seq)
        alarm = {"ml": alarm_seri(det.skor_banyak(X) > 1),
                 "if": alarm_seri(-iforest.score_samples(X) > ambang_if),
                 "aturan": np.array([x["suhu"] > 35 or x["tanah"] > 85 for x in seq])}
        for k, al in alarm.items():
            palsu[k] += naik(al[JENDELA:LAMA_NORMAL])
            for g, t0 in mulai.items():
                w = al[t0:t0 + LAMA_GANGGUAN]
                if w.any():
                    hasil[k][g].append(int(np.argmax(w)) + 1)
    jam = a.uji * (LAMA_NORMAL - JENDELA) / 3600
    laporan = {"data_latih": str(path.name), "n_latih": det.n_latih, "uji": a.uji, "jam_normal_diuji": round(jam, 2),
               "alarm": "nyala 3 titik anomali, padam 10 titik normal", "hasil": {}}
    nama = {"ml": "ML (EllipticEnvelope)", "if": "Isolation Forest", "aturan": "Aturan ambang"}
    print(f"\n{a.uji} simulasi uji, {jam:.1f} jam kondisi normal, tiap gangguan diuji {a.uji}x")
    print(f"{'metode':24s} {'alarm palsu/jam':>15s} | " + " | ".join(f"{g} median (p90, terlewat)" for g in GANGGUAN))
    for k in ("ml", "if", "aturan"):
        rk = {g: ringkas(hasil[k][g], a.uji) for g in GANGGUAN}
        laporan["hasil"][nama[k]] = {"alarm_palsu_per_jam": round(palsu[k] / jam, 2), **rk}
        print(f"{nama[k]:24s} {palsu[k] / jam:15.2f} | " + " | ".join(
            f"{rk[g]['median_s']} s ({rk[g]['p90_s']}, {rk[g]['terlewat']})" for g in GANGGUAN))
    (HERE / "evaluasi_offline.json").write_text(json.dumps(laporan, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\ntersimpan: evaluasi_offline.json")


if __name__ == "__main__":
    main()
