"""Langkah 1 - unduh data: LibriVox Indonesia (rekaman buku domain publik, CC0) bagian uji, 31 MB.

Sama di semua OS:  python download_data.py   (macOS/Linux: python3)
"""
import tarfile
from pathlib import Path
from urllib.request import urlopen

BASE = "https://huggingface.co/datasets/indonesian-nlp/librivox-indonesia/resolve/main/data/"
DATA = Path(__file__).resolve().parent / "data"


def main() -> None:
    DATA.mkdir(exist_ok=True)
    for name in ("metadata_test.csv.gz", "audio_test.tgz"):
        if not (DATA / name).exists():
            (DATA / name).write_bytes(urlopen(BASE + name, timeout=300).read())
        print("tersimpan:", f"data/{name}")
    with tarfile.open(DATA / "audio_test.tgz") as tar:
        tar.extractall(DATA, filter="data")  # filter="data": aman dari path berbahaya di arsip
    print("diekstrak ke: data/librivox-indonesia")
    print("Sumber: https://huggingface.co/datasets/indonesian-nlp/librivox-indonesia (Public Domain/CC0)")


if __name__ == "__main__":
    main()
