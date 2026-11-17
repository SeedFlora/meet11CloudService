"""Langkah 1 - unduh data: suhu & kelembapan per jam Kota Bandung dari Open-Meteo Archive (gratis, tanpa API key).

Sama di semua OS:  python download_data.py   (macOS/Linux: python3)
Ubah lokasi lewat LAT/LON di bawah (mis. kota kampusmu). Data: Open-Meteo.com, lisensi CC BY 4.0.
"""
import json
import time
from datetime import date, timedelta
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import urlopen

import pandas as pd

LAT, LON, KOTA = -6.9175, 107.6191, "Bandung"
MULAI = "2023-01-01"
SELESAI = (date.today() - timedelta(days=7)).isoformat()  # arsip tertinggal beberapa hari
DATA = Path(__file__).resolve().parent / "data"


def main() -> None:
    DATA.mkdir(exist_ok=True)
    url = ("https://archive-api.open-meteo.com/v1/archive"
           f"?latitude={LAT}&longitude={LON}&start_date={MULAI}&end_date={SELESAI}"
           "&hourly=temperature_2m,relative_humidity_2m,precipitation&timezone=Asia%2FJakarta")
    for coba in range(5):
        try:
            raw = json.load(urlopen(url, timeout=120))["hourly"]
            break
        except HTTPError as exc:  # 429 = terlalu banyak request dari IP yang sama; tunggu lalu ulangi
            if exc.code != 429 or coba == 4:
                raise
            print(f"HTTP 429, coba lagi dalam {15 * (coba + 1)} detik ...")
            time.sleep(15 * (coba + 1))
    df = pd.DataFrame(raw).rename(columns={"time": "waktu", "temperature_2m": "suhu",
                                           "relative_humidity_2m": "kelembapan", "precipitation": "hujan"})
    df = df.dropna()
    df.to_csv(DATA / "bandung_hourly.csv", index=False)
    print(f"tersimpan: data/bandung_hourly.csv ({len(df)} jam, {df['waktu'].iloc[0]} s.d. {df['waktu'].iloc[-1]}, {KOTA})")
    print("Sumber: Weather data by Open-Meteo.com (CC BY 4.0)")


if __name__ == "__main__":
    main()
