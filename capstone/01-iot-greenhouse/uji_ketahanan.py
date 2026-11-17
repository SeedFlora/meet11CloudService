"""Uji validasi sensor tanpa broker MQTT."""

from pathlib import Path
import sys
import time
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent / "app"))
with patch("threading.Thread.start"):
    from main import health, proses, valid


class ValidasiSensor(unittest.TestCase):
    def test_sensor_membutuhkan_nilai_angka_hingga(self):
        dasar = {"suhu": 28.1, "kelembapan": 70.2, "tanah": 50.3}
        self.assertTrue(valid(dasar))
        for nilai in ("28.1", float("nan"), True, None):
            with self.subTest(nilai=nilai):
                self.assertFalse(valid(dict(dasar, suhu=nilai)))

    def test_waktu_server_diisi_untuk_esp32_yang_belum_sinkron(self):
        data = {"ts": 0, "suhu": 28.1, "kelembapan": 70.2, "tanah": 50.3}
        proses(data)
        self.assertAlmostEqual(data["ts"], time.time(), delta=3)
        self.assertGreater(health()["data"], 0)


if __name__ == "__main__":
    unittest.main()
