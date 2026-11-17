"""Langkah 1 - unduh data: IndoNLU SmSA (ulasan berbahasa Indonesia, label positive/neutral/negative).

Sama di semua OS:  python download_data.py   (macOS/Linux: python3)
"""
from pathlib import Path
from urllib.request import urlopen

BASE = "https://raw.githubusercontent.com/IndoNLP/indonlu/master/dataset/smsa_doc-sentiment-prosa/"
DATA = Path(__file__).resolve().parent / "data"


def main() -> None:
    DATA.mkdir(exist_ok=True)
    for name in ("train_preprocess.tsv", "valid_preprocess.tsv"):
        (DATA / name).write_bytes(urlopen(BASE + name, timeout=60).read())
        print("tersimpan:", f"data/{name}")
    print("Sumber: IndoNLU (Wilie dkk., 2020) https://github.com/IndoNLP/indonlu")


if __name__ == "__main__":
    main()
