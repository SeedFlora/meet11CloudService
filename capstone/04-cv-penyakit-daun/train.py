"""Langkah 2-3 - latih & evaluasi deteksi penyakit daun tomat (transfer learning di CPU).

Ide: CNN EfficientNet-B0 yang sudah dilatih di ImageNet dipakai sebagai "mata" (fitur 1.280 angka per foto),
lalu Logistic Regression belajar membedakan 10 kelas daun. Baseline: histogram warna + Logistic Regression.

Sama di semua OS:  python train.py   (macOS/Linux: python3)
Hasil: classifier.joblib, metrics.json, confusion.png, contoh/*.jpg di folder ini.
"""
import json
import shutil
import time
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import timm
import torch
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, f1_score
from sklearn.model_selection import train_test_split

HERE = Path(__file__).resolve().parent
BACKBONE = "efficientnet_b0.ra_in1k"   # 5,3 juta parameter, unduh sekali ±21 MB dari Hugging Face


def histogram_warna(img):
    hsv = np.asarray(img.convert("HSV").resize((128, 128)))
    h, _ = np.histogramdd(hsv.reshape(-1, 3), bins=(8, 8, 8), range=((0, 256),) * 3)
    return (h / h.sum()).ravel()


def main() -> None:
    files = sorted((HERE / "data" / "daun").glob("*/*.jpg"))
    labels = [f.parent.name for f in files]
    kelas = sorted(set(labels))
    y = np.array([kelas.index(l) for l in labels])
    idx_tr, idx_te = train_test_split(np.arange(len(files)), test_size=0.3, stratify=y, random_state=42)
    print(f"{len(files)} foto, {len(kelas)} kelas; latih {len(idx_tr)}, uji {len(idx_te)}")

    hist = np.stack([histogram_warna(Image.open(f)) for f in files])
    base = LogisticRegression(max_iter=3000).fit(hist[idx_tr], y[idx_tr])
    pb = base.predict(hist[idx_te])

    model = timm.create_model(BACKBONE, pretrained=True, num_classes=0).eval()
    cfg = timm.data.resolve_data_config({}, model=model)
    tf = timm.data.create_transform(**cfg)
    t0 = time.perf_counter()
    with torch.no_grad():
        fitur = np.concatenate([model(torch.stack([tf(Image.open(f).convert("RGB")) for f in files[i:i + 32]])).numpy()
                                for i in range(0, len(files), 32)])
    ips = len(files) / (time.perf_counter() - t0)
    clf = LogisticRegression(max_iter=3000, C=2.0).fit(fitur[idx_tr], y[idx_tr])
    pm = clf.predict(fitur[idx_te])

    hasil = {"histogram_warna_logreg": {"akurasi": round(accuracy_score(y[idx_te], pb), 3), "macro_f1": round(f1_score(y[idx_te], pb, average="macro"), 3)},
             "efficientnet_b0_logreg": {"akurasi": round(accuracy_score(y[idx_te], pm), 3), "macro_f1": round(f1_score(y[idx_te], pm, average="macro"), 3)}}
    print(f"Baseline histogram warna + LogReg : akurasi {hasil['histogram_warna_logreg']['akurasi']}, macro-F1 {hasil['histogram_warna_logreg']['macro_f1']}")
    print(f"EfficientNet-B0 (beku) + LogReg   : akurasi {hasil['efficientnet_b0_logreg']['akurasi']}, macro-F1 {hasil['efficientnet_b0_logreg']['macro_f1']}")
    print(f"Ekstraksi fitur CNN di CPU: {ips:.1f} foto/detik")

    joblib.dump({"clf": clf, "kelas": kelas, "backbone": BACKBONE}, HERE / "classifier.joblib")
    contoh = HERE / "contoh"
    contoh.mkdir(exist_ok=True)
    for k in kelas:  # satu foto uji per kelas untuk contoh di web UI
        i = next(i for i in idx_te if labels[i] == k)
        shutil.copy(files[i], contoh / f"{k}.jpg")
    (HERE / "metrics.json").write_text(json.dumps({"hasil_uji": hasil, "kelas": kelas, "n_latih": len(idx_tr), "n_uji": len(idx_te),
                                                  "foto_per_detik_cpu": round(ips, 1), "backbone": BACKBONE}, indent=2), encoding="utf-8")
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    pendek = [k.replace("Spider_mites Two-spotted_spider_mite", "Spider_mites").replace("Tomato_", "")[:18] for k in kelas]
    ConfusionMatrixDisplay.from_predictions(y[idx_te], pm, display_labels=pendek, ax=ax, colorbar=False, xticks_rotation=60)
    ax.set_title("Confusion matrix data uji (EfficientNet-B0 + LogReg)")
    fig.tight_layout()
    fig.savefig(HERE / "confusion.png", dpi=120)
    print("tersimpan: classifier.joblib, metrics.json, confusion.png, contoh/")


if __name__ == "__main__":
    main()
