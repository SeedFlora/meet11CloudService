"""Langkah 4 - web UI (Gradio): gambar angka 0-9 dengan mouse/jari, CNN menebak.

Sama di semua OS:  python app.py   lalu buka http://127.0.0.1:7873
Port lain: PowerShell $env:PORT=7883 · cmd set PORT=7883 · macOS/Linux PORT=7883 python app.py
"""
import json
import os
import sys
from pathlib import Path

import gradio as gr
import numpy as np
import torch
from PIL import Image

from train import CNN

HERE = Path(__file__).resolve().parent
model = CNN()
if not (HERE / "model.pt").exists():  # langkah sebelumnya belum dijalankan
    sys.exit("model.pt belum ada. Jalankan dulu di folder ini:  python train.py")
model.load_state_dict(torch.load(HERE / "model.pt", map_location="cpu"))
model.eval()
metrics = json.loads((HERE / "metrics.json").read_text(encoding="utf-8"))


def ke_mnist(gambar):
    """Ubah gambar bebas menjadi format MNIST: angka putih di latar hitam, 20x20 di tengah kanvas 28x28."""
    if gambar is None:
        return None
    if isinstance(gambar, dict):  # Sketchpad mengirim {"composite": ...}
        gambar = gambar.get("composite")
    if gambar is None:
        return None
    arr = np.asarray(gambar).astype(np.float32)
    if arr.ndim < 2:
        return None
    if arr.ndim == 3 and arr.shape[2] == 4 and arr[..., 3].min() < 250:
        tinta = arr[..., 3] * (1 - arr[..., :3].mean(axis=2) / 255.0)  # coretan gelap di kanvas transparan
    else:
        abu = arr[..., :3].mean(axis=2) if arr.ndim == 3 else arr
        tinta = 255.0 - abu if abu.mean() > 127 else abu  # latar terang -> balik warnanya
    if tinta.max() < 30:
        return None
    ys, xs = np.where(tinta > 30)
    kotak = Image.fromarray(tinta[ys.min():ys.max() + 1, xs.min():xs.max() + 1].astype(np.uint8))
    kotak.thumbnail((20, 20), Image.LANCZOS)
    kanvas = Image.new("L", (28, 28))
    kanvas.paste(kotak, ((28 - kotak.width) // 2, (28 - kotak.height) // 2))
    return np.asarray(kanvas, dtype=np.float32) / 255.0


def tebak(gambar):
    x = ke_mnist(gambar)
    if x is None:
        return {}, None
    with torch.no_grad():
        p = torch.softmax(model(torch.from_numpy(x)[None, None]), dim=1)[0].numpy()
    return {str(k): float(p[k]) for k in range(10)}, Image.fromarray((x * 255).astype(np.uint8)).resize((140, 140), Image.NEAREST)


akhir = metrics["cnn"][-1]
with gr.Blocks(title="Pengenal Angka Tulisan Tangan") as demo:
    gr.Markdown(f"# Pengenal Angka Tulisan Tangan\nDeep learning · CNN {metrics['parameter']:,} parameter dilatih sendiri di CPU · "
                f"akurasi uji MNIST {akhir['akurasi_uji']:.2%} (baseline Logistic Regression {metrics['baseline_logreg']:.2%})")
    with gr.Tab("Gambar sendiri"):
        with gr.Row():
            kanvas = gr.Sketchpad(label="Gambar satu angka (0-9)", type="numpy", height=380, width=380,
                                  brush=gr.Brush(default_size=18, colors=["#000000"], color_mode="fixed"))
            with gr.Column():
                hasil = gr.Label(num_top_classes=3, label="Tebakan CNN")
                lihat = gr.Image(label="Yang dilihat model (28x28)", height=160, width=160)
        gr.Button("Tebak", variant="primary").click(tebak, kanvas, [hasil, lihat])
    with gr.Tab("Contoh data uji"):
        with gr.Row():
            unggah = gr.Image(type="numpy", label="Gambar angka", height=280)
            with gr.Column():
                hasil2 = gr.Label(num_top_classes=3, label="Tebakan CNN")
                lihat2 = gr.Image(label="Yang dilihat model (28x28)", height=160, width=160)
        gr.Examples([[str(HERE / "contoh" / f"angka_{k}.png")] for k in (3, 7, 9)], inputs=unggah)
        unggah.change(tebak, unggah, [hasil2, lihat2])
    with gr.Tab("Kurva training"):
        gr.Markdown("\n".join(f"- epoch {r['epoch']}: loss {r['loss']}, akurasi uji {r['akurasi_uji']:.2%}, {r['detik']} s"
                              for r in metrics["cnn"]))
        gr.Image(str(HERE / "kurva.png"), label="Loss dan akurasi per epoch")

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=int(os.getenv("PORT", "7873")))
