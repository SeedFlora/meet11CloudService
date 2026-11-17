"""Langkah 1 - unduh data: UCI Student Performance (CC BY 4.0), nilai siswa + kebiasaan belajar.

Sama di semua OS:  python download_data.py   (macOS/Linux: python3)
"""
from io import BytesIO
from pathlib import Path
from urllib.request import urlopen
from zipfile import ZipFile

URL = "https://archive.ics.uci.edu/static/public/320/student+performance.zip"
DATA = Path(__file__).resolve().parent / "data"


def main() -> None:
    DATA.mkdir(exist_ok=True)
    outer = ZipFile(BytesIO(urlopen(URL, timeout=60).read()))
    inner = ZipFile(BytesIO(outer.read("student.zip")))  # zip UCI berisi zip lagi
    for name in ("student-mat.csv", "student-por.csv"):
        (DATA / name).write_bytes(inner.read(name))
        print("tersimpan:", f"data/{name}")
    print("Sumber: https://archive.ics.uci.edu/dataset/320/student+performance (Cortez & Silva, 2008)")


if __name__ == "__main__":
    main()
