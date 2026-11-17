"""Langkah 1-3 - unduh MNIST, latih CNN kecil di CPU, evaluasi vs baseline Logistic Regression.

Sama di semua OS:  python train.py   (macOS/Linux: python3)
Waktu di mesin uji 2 vCPU: sekitar 1 menit (baseline 6 s + 3 epoch masing-masing 10-13 s).
Hasil: model.pt, metrics.json, kurva.png, contoh/*.png di folder ini.
"""
import io
import json
import os
import time
from pathlib import Path
from urllib.request import urlopen

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.linear_model import LogisticRegression
from torch import nn

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
URL = "https://huggingface.co/datasets/ylecun/mnist/resolve/main/mnist/{}-00000-of-00001.parquet"
EPOCH = 3
torch.manual_seed(42)
torch.set_num_threads(os.cpu_count() or 2)


def load(split):
    DATA.mkdir(exist_ok=True)
    path = DATA / f"{split}.parquet"
    if not path.exists():
        path.write_bytes(urlopen(URL.format(split), timeout=120).read())
    df = pd.read_parquet(path)
    x = np.stack([np.array(Image.open(io.BytesIO(item["bytes"]))) for item in df["image"]]).astype(np.float32) / 255.0
    return x, df["label"].to_numpy().copy()  # salinan agar bisa ditulis (aman untuk torch)


class CNN(nn.Module):
    """2 blok konvolusi + 2 lapis dense; sekitar 0,1 juta parameter."""

    def __init__(self):
        super().__init__()
        self.c1 = nn.Conv2d(1, 16, 3, padding=1)
        self.c2 = nn.Conv2d(16, 32, 3, padding=1)
        self.fc1 = nn.Linear(32 * 7 * 7, 64)
        self.fc2 = nn.Linear(64, 10)
        self.drop = nn.Dropout(0.25)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.c1(x)), 2)
        x = F.max_pool2d(F.relu(self.c2(x)), 2)
        return self.fc2(self.drop(F.relu(self.fc1(x.flatten(1)))))


def main() -> None:
    x_tr, y_tr = load("train")
    x_te, y_te = load("test")
    print(f"MNIST: latih {len(x_tr)} gambar, uji {len(x_te)} gambar (28x28 piksel)")

    t0 = time.perf_counter()
    base = LogisticRegression(max_iter=300).fit(x_tr[:20000].reshape(20000, -1), y_tr[:20000])
    acc_base = float((base.predict(x_te.reshape(len(x_te), -1)) == y_te).mean())
    print(f"Baseline Logistic Regression (piksel mentah, 20k data): akurasi uji {acc_base:.4f} ({time.perf_counter() - t0:.0f} s)")

    model = CNN()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    xt, yt = torch.from_numpy(x_tr).unsqueeze(1), torch.from_numpy(y_tr)
    xe, ye = torch.from_numpy(x_te).unsqueeze(1), torch.from_numpy(y_te)
    riwayat = []
    for ep in range(1, EPOCH + 1):
        t0 = time.perf_counter()
        model.train()
        idx = torch.randperm(len(xt))
        total = 0.0
        for i in range(0, len(xt), 128):
            b = idx[i:i + 128]
            opt.zero_grad()
            loss = F.cross_entropy(model(xt[b]), yt[b])
            loss.backward()
            opt.step()
            total += loss.item() * len(b)
        model.eval()
        with torch.no_grad():
            pred = torch.cat([model(xe[i:i + 1000]).argmax(1) for i in range(0, len(xe), 1000)])
        acc = float((pred == ye).float().mean())
        riwayat.append({"epoch": ep, "loss": round(total / len(xt), 4), "akurasi_uji": round(acc, 4),
                        "detik": round(time.perf_counter() - t0, 1)})
        print(f"epoch {ep}: loss {riwayat[-1]['loss']}, akurasi uji {acc:.4f}, {riwayat[-1]['detik']} s")

    torch.save(model.state_dict(), HERE / "model.pt")
    salah = (pred != ye).nonzero().flatten()[:8].tolist()
    contoh = HERE / "contoh"
    contoh.mkdir(exist_ok=True)
    for k in range(10):  # satu gambar uji per angka untuk contoh di web UI
        i = int(np.where(y_te == k)[0][0])
        Image.fromarray((255 - x_te[i] * 255).astype(np.uint8)).resize((140, 140), Image.NEAREST).save(contoh / f"angka_{k}.png")
    (HERE / "metrics.json").write_text(json.dumps({"baseline_logreg": round(acc_base, 4), "cnn": riwayat,
                                                  "parameter": sum(p.numel() for p in model.parameters()),
                                                  "salah_contoh": [[int(y_te[i]), int(pred[i])] for i in salah]}, indent=2),
                                       encoding="utf-8")
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.2))
    ax[0].plot([r["epoch"] for r in riwayat], [r["loss"] for r in riwayat], "o-", color="#2563eb")
    ax[0].set_title("Loss latih per epoch")
    ax[1].plot([r["epoch"] for r in riwayat], [r["akurasi_uji"] for r in riwayat], "o-", color="#059669", label="CNN")
    ax[1].axhline(acc_base, color="#f59e0b", ls="--", label="baseline LogReg")
    ax[1].set_title("Akurasi data uji")
    ax[1].legend()
    for a in ax:
        a.set_xlabel("epoch")
        a.set_xticks(range(1, EPOCH + 1))
    fig.tight_layout()
    fig.savefig(HERE / "kurva.png", dpi=130)
    print(f"parameter CNN: {sum(p.numel() for p in model.parameters()):,}; tersimpan: model.pt, metrics.json, kurva.png, contoh/")


if __name__ == "__main__":
    main()
