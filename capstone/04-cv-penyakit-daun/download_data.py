"""Langkah 1 - unduh data: foto daun tomat dari PlantVillage (mirror Hugging Face; lisensi sumber CC BY-SA 3.0).

Sama di semua OS:  python download_data.py   (macOS/Linux: python3)
Mengunduh 1 file parquet 170 MB sekali, lalu menyimpan maksimal 150 foto per kelas tomat ke data/daun/<kelas>/.
"""
import io
import json
import random
from collections import defaultdict
from pathlib import Path
from urllib.request import urlopen

import pyarrow.parquet as pq
from PIL import Image

URL = "https://huggingface.co/datasets/BrandonFors/Plant-Diseases-PlantVillage-Dataset/resolve/main/data/test-00000-of-00001.parquet"
HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
PER_KELAS = 150


def main() -> None:
    DATA.mkdir(exist_ok=True)
    raw = DATA / "plantvillage_test.parquet"
    if not raw.exists():
        print("mengunduh 170 MB ... (sekali saja)")
        raw.write_bytes(urlopen(URL, timeout=600).read())
    tabel = pq.read_table(raw)
    meta = pq.read_metadata(raw).metadata or {}  # nama kelas ada di metadata "huggingface" pada parquet
    info = json.loads(meta.get(b"huggingface", b"{}") or b"{}")
    kelas = info["info"]["features"]["label"]["names"]
    per_kelas = defaultdict(list)
    for i, lab in enumerate(tabel.column("label").to_pylist()):
        if kelas[lab].startswith("Tomato"):
            per_kelas[kelas[lab]].append(i)
    random.seed(42)
    gambar = tabel.column("image").to_pylist()
    for k, idx in sorted(per_kelas.items()):
        pilih = random.sample(idx, min(PER_KELAS, len(idx)))
        folder = DATA / "daun" / k.replace("Tomato___", "")
        folder.mkdir(parents=True, exist_ok=True)
        for n, i in enumerate(pilih):
            Image.open(io.BytesIO(gambar[i]["bytes"])).convert("RGB").save(folder / f"{n:03d}.jpg", quality=92)
        print(f"{k:55} tersedia {len(idx):4d}, disimpan {len(pilih)}")
    print("Sumber: PlantVillage (Hughes & Salathe, 2015) via huggingface.co/datasets/BrandonFors/Plant-Diseases-PlantVillage-Dataset")


if __name__ == "__main__":
    main()
