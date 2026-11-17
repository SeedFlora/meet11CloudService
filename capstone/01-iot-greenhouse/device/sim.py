"""ESP32 virtual: sensor suhu, kelembapan udara, dan kelembapan tanah greenhouse, kirim JSON via MQTT tiap detik.

Payload sama dengan sketch ESP32 di folder esp32/, jadi server tidak perlu tahu sensornya asli atau simulasi.
Perintah gangguan (untuk latihan) diterima di topik greenhouse/<id>/cmd: normal, kipas_mati, sensor_rusak, pompa_bocor.
Kelas Greenhouse juga dipakai evaluasi_offline.py untuk menguji model tanpa broker.
"""
import json
import math
import os
import random
import time

HOST = os.getenv("MQTT_HOST", "localhost")
PORT = int(os.getenv("MQTT_PORT", "1883"))
DEVICE = os.getenv("DEVICE_ID", "gh-01")
INTERVAL = float(os.getenv("INTERVAL_S", "1"))
TELEMETRY, CMD = f"greenhouse/{DEVICE}/telemetry", f"greenhouse/{DEVICE}/cmd"
SKENARIO = ("normal", "kipas_mati", "sensor_rusak", "pompa_bocor")


class Greenhouse:
    """Fisika sederhana per detik. 1 "hari" dipersingkat menjadi 600 detik agar siklusnya cepat terlihat."""

    def __init__(self, seed=None, t0=0):
        self.rng = random.Random(seed)
        self.t, self.panas, self.tanah, self.pompa = t0, 0.0, 50.0, False

    def langkah(self, sk="normal"):
        fase = 2 * math.pi * self.t / 600
        self.panas = min(self.panas + 0.12, 15) if sk == "kipas_mati" else self.panas * 0.9
        if sk == "pompa_bocor":
            self.tanah = min(self.tanah + 0.8, 98)
        elif self.tanah > 52:              # setelah bocor: air meresap/terbuang kembali ke normal
            self.tanah -= 0.5
        else:                              # pompa otomatis menjaga tanah 48-50 %
            self.pompa = self.tanah < 48 or (self.pompa and self.tanah < 50)
            self.tanah += -0.03 + (0.15 if self.pompa else 0)
        suhu = 27 + 3 * math.sin(fase) + self.panas + self.rng.gauss(0, 0.15)
        lembap = 72 - 8 * math.sin(fase) - self.panas * 1.2 + self.rng.gauss(0, 0.6)
        if sk == "sensor_rusak" and self.rng.random() < 0.7:
            suhu = 85.0                    # nilai macet khas sensor DHT rusak/terputus
        self.t += 1
        return {"suhu": round(suhu, 2), "kelembapan": round(lembap, 2),
                "tanah": round(self.tanah + self.rng.gauss(0, 0.3), 2)}


def main() -> None:
    import paho.mqtt.client as mqtt

    gh, skenario = Greenhouse(), {"aktif": "normal"}

    def on_connect(client, userdata, flags, reason_code, properties=None):
        client.subscribe(CMD)
        print(f"terhubung ke {HOST}:{PORT}, kirim ke {TELEMETRY}", flush=True)

    def on_message(client, userdata, msg):
        try:
            fault = json.loads(msg.payload).get("fault", "normal")
        except ValueError:
            return
        if fault in SKENARIO:
            skenario["aktif"] = fault
            print("perintah skenario:", fault, flush=True)

    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=f"sim-{DEVICE}")
    client.on_connect, client.on_message = on_connect, on_message
    while True:  # broker bisa belum siap saat container baru naik
        try:
            client.connect(HOST, PORT, keepalive=30)
            break
        except OSError:
            print("broker belum siap, coba lagi 2 detik ...", flush=True)
            time.sleep(2)
    client.loop_start()
    while True:
        sk = skenario["aktif"]
        payload = {"device": DEVICE, "ts": time.time(), **gh.langkah(sk),
                   "skenario": sk}  # hanya untuk simulasi/evaluasi; perangkat asli tidak mengirim field ini
        client.publish(TELEMETRY, json.dumps(payload), qos=0)
        time.sleep(INTERVAL)


if __name__ == "__main__":
    main()
