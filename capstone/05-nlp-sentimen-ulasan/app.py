"""Langkah 4 - web UI (Gradio): analisis 1 ulasan + dashboard banyak ulasan.

Sama di semua OS:  python app.py   lalu buka http://127.0.0.1:7875
Port lain: PowerShell $env:PORT=7885 · cmd set PORT=7885 · macOS/Linux PORT=7885 python app.py
"""
import json
import os
import sys
from pathlib import Path

import gradio as gr
import joblib
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
if not (HERE / "model.joblib").exists():  # langkah sebelumnya belum dijalankan
    sys.exit("model.joblib belum ada. Jalankan dulu di folder ini:  python download_data.py  lalu  python train.py")
model = joblib.load(HERE / "model.joblib")
metrics = json.loads((HERE / "metrics.json").read_text(encoding="utf-8"))
NAMA = {"positive": "Positif", "neutral": "Netral", "negative": "Negatif"}
CONTOH = """pelayanan cepat dan ramah, makanannya enak sekali
aplikasi sering error saat bayar, kecewa berat
pesanan datang tepat waktu tapi kemasannya penyok
harga mahal tidak sebanding dengan rasa
tempatnya bersih, parkir luas, pasti kembali lagi
customer service lambat membalas chat
biasa saja, tidak ada yang istimewa
kurirnya sopan dan barang sesuai foto"""


def kata_penting(teks, kelas, k=5):
    vec, lr = model.named_steps["tfidf"], model.named_steps["lr"]
    x = vec.transform([teks])
    idx = list(lr.classes_).index(kelas)
    kontrib = x.multiply(lr.coef_[idx]).toarray()[0]
    top = np.argsort(kontrib)[::-1][:k]
    nama = vec.get_feature_names_out()
    return [nama[i] for i in top if kontrib[i] > 0]


def satu(teks):
    proba = model.predict_proba([teks])[0]
    hasil = {NAMA[c]: float(p) for c, p in zip(model.classes_, proba)}
    kelas = model.classes_[int(np.argmax(proba))]
    kata = kata_penting(teks, kelas)
    return hasil, f"Kata yang paling mendorong **{NAMA[kelas]}**: " + (", ".join(f"`{w}`" for w in kata) or "-")


def banyak(teks):
    baris = [b.strip() for b in teks.splitlines() if b.strip()]
    label = [NAMA[c] for c in model.predict(baris)]
    tabel = pd.DataFrame({"ulasan": baris, "sentimen": label})
    hitung = tabel["sentimen"].value_counts().reindex(["Positif", "Netral", "Negatif"], fill_value=0).reset_index()
    hitung.columns = ["sentimen", "jumlah"]
    persen = (hitung.set_index("sentimen")["jumlah"] / len(tabel) * 100).round(0).astype(int)
    ringkas = (f"**{len(tabel)} ulasan** · Positif {persen['Positif']}% · Netral {persen['Netral']}% · "
               f"Negatif {persen['Negatif']}%")
    # sumbu y mulai dari 0 agar kelas dengan 1 ulasan tetap terlihat batangnya
    return gr.BarPlot(value=hitung, y_lim=[0, int(hitung['jumlah'].max()) + 1]), tabel, ringkas


def evaluasi():
    f1 = metrics["macro_f1"]
    baris = ["| Model | macro-F1 (validasi) |", "|---|---|",
             f"| Baseline mayoritas (semua positif) | {f1['baseline_mayoritas']} |",
             f"| **TF-IDF + Logistic Regression** | **{f1['tfidf_logreg']}** |"]
    if "embedding_logreg" in f1:
        baris.append(f"| Embedding multibahasa + LogReg | {f1['embedding_logreg']} ({f1['embedding_detik']} s di CPU) |")
    per = metrics["f1_per_kelas"]
    baris.append(f"\nF1 per kelas: negatif {per['negative']}, netral {per['neutral']}, positif {per['positive']}. "
                 f"Data IndoNLU SmSA: {metrics['n_latih']} latih, {metrics['n_validasi']} validasi.")
    return "\n".join(baris)


with gr.Blocks(title="Dashboard Sentimen Ulasan") as demo:
    gr.Markdown("# Dashboard Sentimen Ulasan\nNLP bahasa Indonesia · TF-IDF + Logistic Regression · data IndoNLU SmSA")
    with gr.Tab("Dashboard banyak ulasan"):
        masukan = gr.Textbox(CONTOH, lines=8, label="Satu ulasan per baris (tempel dari CSV/Google Form)")
        tombol = gr.Button("Analisis semua", variant="primary")
        ringkas = gr.Markdown()
        with gr.Row():
            grafik = gr.BarPlot(x="sentimen", y="jumlah", title="Sebaran sentimen", height=280)
            tabel = gr.Dataframe(label="Hasil per ulasan", wrap=True)
        tombol.click(banyak, masukan, [grafik, tabel, ringkas])
    with gr.Tab("Analisis 1 ulasan"):
        teks = gr.Textbox("aplikasi sering error saat bayar, kecewa berat", label="Ulasan")
        cek = gr.Button("Analisis", variant="primary")
        prob = gr.Label(label="Probabilitas")
        alasan = gr.Markdown()
        cek.click(satu, teks, [prob, alasan])
    with gr.Tab("Evaluasi model"):
        gr.Markdown(evaluasi())

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=int(os.getenv("PORT", "7875")))
