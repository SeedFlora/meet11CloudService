"""Server IoT + ML: terima data sensor via MQTT, deteksi anomali, prakiraan suhu, dashboard web.

Alur ML:
1. Latih model dari riwayat data NORMAL (riwayat/normal_10menit.csv = 1 siklus siang-malam, 600 data).
   Tidak ada file? Server mengumpulkan WARMUP detik data live dulu, lalu melatih model dari situ.
2. Fitur dan model ada di detektor.py (sama persis dengan yang diuji evaluasi_offline.py).
3. Alarm ML menyala setelah 3 titik anomali berturut-turut (debounce) dan padam setelah 10 titik normal (histeresis).
4. Pembanding (baseline): aturan ambang suhu > BATAS_SUHU atau tanah > 85 %.
"""
import csv
import io
import json
import math
import os
import threading
import time
from collections import deque
from pathlib import Path
from typing import Literal

import numpy as np
import paho.mqtt.client as mqtt
from fastapi import FastAPI
from fastapi.responses import FileResponse, PlainTextResponse, StreamingResponse
from pydantic import BaseModel

from detektor import JENDELA, Alarm, Detektor, baca_csv, fitur

HERE = Path(__file__).resolve().parent
HOST = os.getenv("MQTT_HOST", "localhost")
PORT = int(os.getenv("MQTT_PORT", "1883"))
DEVICE_ID = os.getenv("DEVICE_ID", "gh-01")
WARMUP = int(os.getenv("WARMUP", "600"))
BATAS_SUHU = float(os.getenv("BATAS_SUHU", "35"))
RIWAYAT = Path(os.getenv("RIWAYAT_CSV", str(HERE / "riwayat" / "normal_10menit.csv")))

lock = threading.Lock()
rows = deque(maxlen=900)          # 15 menit terakhir (1 data/detik)
alerts = deque(maxlen=30)
episodes = deque(maxlen=100)                     # tiap perubahan skenario: kapan ML dan aturan pertama kali mendeteksi
state = {"model": None, "sumber": None, "alarm": Alarm(nyala=3, padam=10), "alarm_ml": False, "alarm_aturan": False,
         "pesan_masuk": 0, "pesan_ditolak": 0}

if RIWAYAT.exists():
    state["model"] = Detektor().latih(baca_csv(RIWAYAT))
    state["sumber"] = f"riwayat normal {RIWAYAT.name}"
    alerts.appendleft({"ts": time.time(), "jenis": "INFO",
                       "pesan": f"Model dilatih dari {state['model'].n_latih} data normal ({RIWAYAT.name})"})


def angka_valid(value):
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and -1e12 <= value <= 1e12 and math.isfinite(value))


def valid(r):
    """Tolak payload rusak SEBELUM masuk riwayat: satu nilai teks/NaN bisa merusak median, model, dan dashboard."""
    return isinstance(r, dict) and all(
        angka_valid(r.get(k))
        for k in ("suhu", "kelembapan", "tanah"))


def proses(r):
    if not valid(r):
        with lock:
            state["pesan_ditolak"] += 1
        return
    if not angka_valid(r.get("ts")) or r["ts"] < 1e9:   # ESP32 yang jamnya belum sinkron NTP
        r["ts"] = time.time()
    r.setdefault("device", "esp32")
    with lock:
        state["pesan_masuk"] += 1
        rows.append(r)
        seq = list(rows)
        i = len(seq) - 1
        r["fitur"] = fitur(seq, i)
        if state["model"] is None and len(seq) >= WARMUP:
            state["model"] = Detektor().latih(seq)
            state["sumber"] = f"warm-up live {len(seq)} detik"
            alerts.appendleft({"ts": r["ts"], "jenis": "INFO", "pesan": f"Model dilatih dari {state['model'].n_latih} data live"})
        if state["model"] is not None and i >= JENDELA:
            skor = state["model"].skor(r["fitur"])
            r["skor"], r["anomali"] = round(skor, 2), skor > 1
        else:
            r["skor"], r["anomali"] = None, False
        ml = state["alarm"].update(r["anomali"])
        aturan = r["suhu"] > BATAS_SUHU or r["tanah"] > 85
        r["alarm_ml"], r["alarm_aturan"] = ml, aturan
        if ml and not state["alarm_ml"]:
            alerts.appendleft({"ts": r["ts"], "jenis": "ML", "pesan": f"Pola tidak normal (skor {r['skor']}): suhu {r['suhu']} °C, tanah {r['tanah']} %"})
        if aturan and not state["alarm_aturan"]:
            alerts.appendleft({"ts": r["ts"], "jenis": "ATURAN", "pesan": f"Melewati ambang: suhu {r['suhu']} °C, tanah {r['tanah']} %"})
        state["alarm_ml"], state["alarm_aturan"] = ml, aturan
        sk = r.get("skenario", "normal")
        if not episodes or episodes[-1]["skenario"] != sk:
            episodes.append({"skenario": sk, "mulai": r["ts"], "ml": None, "aturan": None})
        ep = episodes[-1]
        if sk != "normal":
            if ml and ep["ml"] is None:
                ep["ml"] = round(r["ts"] - ep["mulai"], 1)
            if aturan and ep["aturan"] is None:
                ep["aturan"] = round(r["ts"] - ep["mulai"], 1)


def prakiraan(seq):
    """Regresi linear 30 detik terakhir -> suhu 60 detik lagi dan perkiraan kapan melewati batas."""
    if len(seq) < 30:
        return None
    y = np.array([x["suhu"] for x in seq[-30:]])
    slope, intercept = np.polyfit(np.arange(30), y, 1)
    nanti = float(intercept + slope * (29 + 60))
    eta = None
    # Siklus harian normal naik paling cepat ±1,9 °C/menit; peringatan dini hanya bila laju jauh di atas itu.
    if slope * 60 > 3 and y[-1] < BATAS_SUHU:
        eta = round((BATAS_SUHU - y[-1]) / slope)
    return {"suhu_60s": round(nanti, 1), "laju_per_menit": round(slope * 60, 2), "eta_batas_s": eta}


def on_connect(client, userdata, flags, reason_code, properties=None):
    client.subscribe("greenhouse/+/telemetry")


def on_message(client, userdata, msg):
    try:
        proses(json.loads(msg.payload))
    except ValueError:              # bukan JSON
        with lock:
            state["pesan_ditolak"] += 1


client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id="greenhouse-server")
client.on_connect, client.on_message = on_connect, on_message


def hubungkan():
    while True:
        try:
            client.connect(HOST, PORT, keepalive=30)
            client.loop_forever(retry_first_connection=True)
        except OSError:
            time.sleep(2)


threading.Thread(target=hubungkan, daemon=True).start()
app = FastAPI(title="Smart Greenhouse IoT + ML")


class Skenario(BaseModel):
    fault: Literal["normal", "kipas_mati", "sensor_rusak", "pompa_bocor"]


@app.get("/")
def index():
    return FileResponse(HERE / "static" / "index.html")


@app.get("/health")
def health():
    return {"status": "ok", "mqtt": client.is_connected(), "data": len(rows), "model_siap": state["model"] is not None,
            "pesan_ditolak": state["pesan_ditolak"]}


@app.get("/api/state")
def api_state():
    with lock:
        seq = list(rows)
        last = seq[-1] if seq else None
        m = state["model"]
        p = prakiraan(seq)
        if p:
            p["andal"] = not state["alarm_ml"]  # saat data anomali (mis. sensor macet 85 °C) prakiraan linear tidak bisa dipercaya
        return {"terakhir": {k: last[k] for k in ("ts", "device", "suhu", "kelembapan", "tanah", "skor", "anomali", "alarm_ml", "alarm_aturan")} if last else None,
                "seri": [{k: x[k] for k in ("ts", "suhu", "kelembapan", "tanah", "anomali", "alarm_ml")} for x in seq[-300:]],
                "model_siap": m is not None, "warmup": [min(len(seq), WARMUP), WARMUP],
                "model": {"metode": m.metode, "sumber": state["sumber"], "n_latih": m.n_latih} if m else None,
                "prakiraan": p, "batas_suhu": BATAS_SUHU, "alerts": list(alerts)[:8],
                "deteksi": [e for e in episodes if e["skenario"] != "normal"][-4:], "mqtt": client.is_connected()}


@app.post("/api/skenario")
def api_skenario(s: Skenario):
    client.publish(f"greenhouse/{DEVICE_ID}/cmd", json.dumps({"fault": s.fault}))
    return {"dikirim": s.fault, "device": DEVICE_ID}


@app.get("/api/export.csv")
def export_csv():
    """Unduh data 15 menit terakhir. Rekam saat kondisi normal untuk dijadikan riwayat/normal_10menit.csv perangkatmu."""
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=["ts", "device", "suhu", "kelembapan", "tanah"], extrasaction="ignore")
    w.writeheader()
    with lock:
        w.writerows(list(rows))
    buf.seek(0)
    return StreamingResponse(buf, media_type="text/csv",
                             headers={"Content-Disposition": "attachment; filename=rekaman_greenhouse.csv"})


@app.get("/metrics", response_class=PlainTextResponse)
def metrics():
    """Format Prometheus (bisa dipantau stack Pertemuan 10)."""
    with lock:
        last = rows[-1] if rows else {}
        masuk, ditolak = state["pesan_masuk"], state["pesan_ditolak"]
    return (f"greenhouse_suhu_celsius {last.get('suhu', 0)}\n"
            f"greenhouse_tanah_persen {last.get('tanah', 0)}\n"
            f"greenhouse_anomali_skor {last.get('skor') or 0}\n"
            f"greenhouse_alarm_ml {int(bool(last.get('alarm_ml')))}\n"
            f"greenhouse_pesan_mqtt_total {masuk}\n"
            f"greenhouse_pesan_ditolak_total {ditolak}\n")
