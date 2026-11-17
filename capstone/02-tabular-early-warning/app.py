"""Langkah 4 - web UI (Gradio): isi data siswa, model memberi peringatan dini.

Sama di semua OS:  python app.py   lalu buka http://127.0.0.1:7872
Port lain: PowerShell $env:PORT=7882 · cmd set PORT=7882 · macOS/Linux PORT=7882 python app.py
"""
import json
import os
import sys
from pathlib import Path

import gradio as gr
import joblib
import pandas as pd

HERE = Path(__file__).resolve().parent
if not (HERE / "model.joblib").exists():  # langkah sebelumnya belum dijalankan
    sys.exit("model.joblib belum ada. Jalankan dulu di folder ini:  python download_data.py  lalu  python train.py")
bundle = joblib.load(HERE / "model.joblib")
metrics = json.loads((HERE / "metrics.json").read_text(encoding="utf-8"))
YA_TIDAK = ["yes", "no"]


def predict(G1, failures, absences, studytime, goout, higher, schoolsup, famsup, age, Dalc):
    row = dict(bundle["defaults"])  # fitur lain diisi nilai tipikal data latih
    row.update(G1=G1, failures=failures, absences=absences, studytime=studytime, goout=goout,
               higher=higher, schoolsup=schoolsup, famsup=famsup, age=age, Dalc=Dalc)
    p = float(bundle["model"].predict_proba(pd.DataFrame([row])[bundle["columns"]])[0, 1])
    status = "BERISIKO" if p >= 0.5 else "AMAN"
    aturan = "berisiko" if G1 < 10 else "aman"
    pesan = (f"### {status}\nProbabilitas berisiko tidak lulus: **{p:.0%}**. "
             f"Pembanding aturan sederhana *G1 < 10*: **{aturan}**.\n\n"
             "Gunakan sebagai pemicu konseling, bukan keputusan akhir.")
    return {"Berisiko tidak lulus": p, "Diperkirakan lulus": 1 - p}, pesan


def tabel_evaluasi():
    rows = ["| Model | Akurasi | Recall berisiko | Presisi | F1 |", "|---|---|---|---|---|"]
    for name, s in metrics["hasil_uji"].items():
        rows.append(f"| {name} | {s['akurasi']} | {s['recall_berisiko']} | {s['presisi_berisiko']} | {s['f1_berisiko']} |")
    rows.append(f"\nData uji: {metrics['n_uji']} dari {metrics['n_siswa']} siswa. "
                f"CV 5-fold F1 (latih): {metrics['cv_f1_mean']} ± {metrics['cv_f1_std']}.")
    return "\n".join(rows)


with gr.Blocks(title="Early Warning Kelulusan") as demo:
    gr.Markdown("# Early Warning Kelulusan Siswa\nTabular ML (Random Forest) · data UCI Student Performance · "
                "prediksi setelah nilai periode 1 (G1) keluar")
    with gr.Tab("Prediksi"):
        with gr.Row():
            with gr.Column():
                G1 = gr.Slider(0, 20, value=9, step=1, label="Nilai periode 1 (G1, skala 0-20)")
                failures = gr.Slider(0, 3, value=1, step=1, label="Jumlah gagal mata pelajaran sebelumnya")
                absences = gr.Slider(0, 75, value=8, step=1, label="Jumlah absen")
                studytime = gr.Slider(1, 4, value=1, step=1, label="Waktu belajar per minggu (1: <2 jam ... 4: >10 jam)")
                goout = gr.Slider(1, 5, value=4, step=1, label="Sering keluar dengan teman (1-5)")
            with gr.Column():
                higher = gr.Radio(YA_TIDAK, value="yes", label="Ingin kuliah?")
                schoolsup = gr.Radio(YA_TIDAK, value="no", label="Dapat bimbingan tambahan sekolah?")
                famsup = gr.Radio(YA_TIDAK, value="yes", label="Dapat dukungan belajar keluarga?")
                age = gr.Slider(15, 22, value=17, step=1, label="Usia")
                Dalc = gr.Slider(1, 5, value=1, step=1, label="Konsumsi alkohol hari kerja (1-5)")
        tombol = gr.Button("Cek risiko", variant="primary")
        with gr.Row():
            prob = gr.Label(label="Probabilitas", num_top_classes=2)
            pesan = gr.Markdown()
        inputs = [G1, failures, absences, studytime, goout, higher, schoolsup, famsup, age, Dalc]
        tombol.click(predict, inputs, [prob, pesan])
        gr.Examples([[7, 2, 14, 1, 5, "no", "no", "no", 18, 3], [15, 0, 2, 3, 2, "yes", "no", "yes", 16, 1]],
                    inputs=inputs, label="Contoh siswa")
    with gr.Tab("Evaluasi model"):
        gr.Markdown(tabel_evaluasi())
        gr.Image(str(HERE / "evaluasi.png"), label="F1 vs baseline dan fitur terpenting", show_label=True)

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=int(os.getenv("PORT", "7872")))
