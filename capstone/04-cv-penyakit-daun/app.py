"""Langkah 4 - web UI (Gradio): unggah/foto daun tomat, model menebak penyakit + saran umum.

Sama di semua OS:  python app.py   lalu buka http://127.0.0.1:7874
Port lain: PowerShell $env:PORT=7884 · cmd set PORT=7884 · macOS/Linux PORT=7884 python app.py
"""
import json
import os
import sys
from pathlib import Path

import gradio as gr
import joblib
import timm
import torch

HERE = Path(__file__).resolve().parent
if not (HERE / "classifier.joblib").exists():  # langkah sebelumnya belum dijalankan
    sys.exit("classifier.joblib belum ada. Jalankan dulu di folder ini:  python download_data.py  lalu  python train.py")
bundle = joblib.load(HERE / "classifier.joblib")
metrics = json.loads((HERE / "metrics.json").read_text(encoding="utf-8"))
model = timm.create_model(bundle["backbone"], pretrained=True, num_classes=0).eval()
tf = timm.data.create_transform(**timm.data.resolve_data_config({}, model=model))
INFO = {  # nama Indonesia + saran umum (bukan pengganti penyuluh pertanian)
    "Bacterial_spot": ("Bercak bakteri", "buang daun sakit, hindari menyiram dari atas daun"),
    "Early_blight": ("Hawar daun awal", "pangkas daun bawah yang terinfeksi, beri jarak tanam"),
    "Late_blight": ("Hawar daun akhir", "pisahkan tanaman sakit, kurangi kelembapan berlebih"),
    "Leaf_Mold": ("Kapang daun", "perbaiki sirkulasi udara dan ventilasi greenhouse"),
    "Septoria_leaf_spot": ("Bercak daun Septoria", "buang daun bawah, lakukan rotasi tanaman"),
    "Spider_mites Two-spotted_spider_mite": ("Tungau laba-laba", "periksa bagian bawah daun, semprot air/sabun insektisida"),
    "Target_Spot": ("Bercak target", "jaga kebersihan kebun dan sisa tanaman"),
    "Tomato_Yellow_Leaf_Curl_Virus": ("Virus kuning keriting (TYLCV)", "kendalikan kutu kebul sebagai pembawa virus"),
    "Tomato_mosaic_virus": ("Virus mosaik tomat", "cabut tanaman sakit, cuci tangan dan alat"),
    "healthy": ("Sehat", "tidak ada tindakan; tetap pantau rutin"),
}


def diagnosa(img):
    if img is None:
        return {}, ""
    with torch.no_grad():
        fitur = model(tf(img.convert("RGB")).unsqueeze(0)).numpy()
    proba = bundle["clf"].predict_proba(fitur)[0]
    nama = {INFO[k][0]: float(p) for k, p in zip(bundle["kelas"], proba)}
    top = bundle["kelas"][int(proba.argmax())]
    yakin = proba.max()
    catatan = "" if yakin >= 0.6 else "\n\n**Model ragu (< 60%)**: foto ulang dengan cahaya cukup, satu daun memenuhi bingkai."
    return nama, f"### {INFO[top][0]}\nSaran umum: {INFO[top][1]}.{catatan}"


h = metrics["hasil_uji"]
with gr.Blocks(title="Deteksi Penyakit Daun Tomat") as demo:
    gr.Markdown(f"# Deteksi Penyakit Daun Tomat\nComputer vision · transfer learning EfficientNet-B0 + Logistic Regression · "
                f"PlantVillage 10 kelas · akurasi uji {h['efficientnet_b0_logreg']['akurasi']:.1%} "
                f"(baseline histogram warna {h['histogram_warna_logreg']['akurasi']:.1%})")
    with gr.Tab("Diagnosa foto"):
        with gr.Row():
            foto = gr.Image(type="pil", label="Foto daun (unggah atau kamera)", height=340, sources=["upload", "webcam"])
            with gr.Column():
                hasil = gr.Label(num_top_classes=3, label="Kemungkinan")
                saran = gr.Markdown()
        gr.Examples([[str(HERE / "contoh" / f"{k}.jpg")] for k in ("Early_blight", "Tomato_Yellow_Leaf_Curl_Virus", "healthy")],
                    inputs=foto, label="Contoh foto data uji")
        foto.change(diagnosa, foto, [hasil, saran])
    with gr.Tab("Evaluasi model"):
        gr.Markdown(f"| Model | Akurasi | macro-F1 |\n|---|---|---|\n"
                    f"| Baseline histogram warna + LogReg | {h['histogram_warna_logreg']['akurasi']} | {h['histogram_warna_logreg']['macro_f1']} |\n"
                    f"| **EfficientNet-B0 beku + LogReg** | **{h['efficientnet_b0_logreg']['akurasi']}** | **{h['efficientnet_b0_logreg']['macro_f1']}** |\n\n"
                    f"Data uji {metrics['n_uji']} foto, latih {metrics['n_latih']} foto. Ekstraksi fitur CPU {metrics['foto_per_detik_cpu']} foto/detik. "
                    "Catatan: foto PlantVillage diambil di latar polos; foto kebun asli lebih sulit.")
        gr.Image(str(HERE / "confusion.png"), label="Confusion matrix", height=520)

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=int(os.getenv("PORT", "7874")))
