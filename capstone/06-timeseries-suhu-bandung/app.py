"""Langkah 4 - web UI (Gradio): pilih tanggal di periode uji, lihat prakiraan 24 jam vs suhu aktual.

Sama di semua OS:  python app.py   lalu buka http://127.0.0.1:7876
Port lain: PowerShell $env:PORT=7886 · cmd set PORT=7886 · macOS/Linux PORT=7886 python app.py
"""
import json
import os
import sys
from pathlib import Path

import gradio as gr
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

HERE = Path(__file__).resolve().parent
if not (HERE / "prediksi_uji.csv").exists():  # langkah sebelumnya belum dijalankan
    sys.exit("prediksi_uji.csv belum ada. Jalankan dulu di folder ini:  python download_data.py  lalu  python train.py")
pred = pd.read_csv(HERE / "prediksi_uji.csv", parse_dates=["t0", "waktu"])
metrics = json.loads((HERE / "metrics.json").read_text(encoding="utf-8"))
pred["tanggal"] = pred["waktu"].dt.strftime("%Y-%m-%d")
TANGGAL = sorted(pred.loc[pred["h"] == 1, "tanggal"].unique())


def tampil(tanggal):
    d = pred[(pred["tanggal"] == tanggal) & (pred["t0"] == pred.loc[pred["tanggal"] == tanggal, "t0"].min())]
    fig, ax = plt.subplots(figsize=(9, 3.8))
    ax.plot(d["waktu"].dt.hour, d["aktual"], color="#111827", lw=2.6, label="aktual (Open-Meteo)")
    ax.plot(d["waktu"].dt.hour, d["model"], color="#2563eb", lw=2.2, marker="o", ms=3.5, label="prakiraan model")
    ax.plot(d["waktu"].dt.hour, d["musiman"], color="#f59e0b", lw=1.8, ls="--", label="baseline: kemarin jam sama")
    ax.set_xlabel("jam (WIB)")
    ax.set_ylabel("suhu (°C)")
    ax.set_xticks(range(0, 24, 2))
    ax.grid(alpha=0.25)
    ax.legend(loc="upper left", fontsize=9)
    ax.set_title(f"Bandung, {tanggal} (prakiraan dibuat pukul 23.00 hari sebelumnya)")
    fig.tight_layout()
    mae_m = (d["model"] - d["aktual"]).abs().mean()
    mae_b = (d["musiman"] - d["aktual"]).abs().mean()
    puncak = d.loc[d["model"].idxmax()]
    teks = (f"**MAE hari ini:** model {mae_m:.2f} °C · baseline {mae_b:.2f} °C  \n"
            f"**Puncak prakiraan:** {puncak['model']:.1f} °C pukul {puncak['waktu']:%H.%M} "
            f"(aktual tertinggi {d['aktual'].max():.1f} °C)")
    return fig, teks


def ringkasan():
    m24, m6 = metrics["mae_jam_1_24"], metrics["mae_jam_1_6"]
    return ("| Metode | MAE jam 1-6 | MAE jam 1-24 |\n|---|---|---|\n"
            f"| Persistensi (suhu jam terakhir) | {m6['persistensi']} | {m24['persistensi']} |\n"
            f"| Baseline musiman (kemarin jam sama) | {m6['musiman']} | {m24['musiman']} |\n"
            f"| **Model residual gradient boosting** | **{m6['model']}** | **{m24['model']}** |\n\n"
            f"Uji: {metrics['hari_uji']} hari terakhir ({metrics['periode'][1]}), latih {metrics['hari_latih']} hari. "
            "Data cuaca: [Open-Meteo.com](https://open-meteo.com/) ([CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)). Data dikelompokkan per jam untuk evaluasi.")


with gr.Blocks(title="Prakiraan Suhu Bandung") as demo:
    gr.Markdown("# Prakiraan Suhu Bandung 24 Jam\nTime series · data per jam Open-Meteo 2023-2026 · "
                "model belajar koreksi terhadap baseline musiman")
    with gr.Row():
        tanggal = gr.Dropdown(TANGGAL, value=TANGGAL[-1], label="Tanggal (periode uji, belum pernah dilihat model)")
    plot = gr.Plot(label="Prakiraan vs aktual")
    info = gr.Markdown()
    gr.Markdown(ringkasan())
    gr.Image(str(HERE / "evaluasi.png"), label="Error per horizon", height=300)
    tanggal.change(tampil, tanggal, [plot, info])
    demo.load(tampil, tanggal, [plot, info])

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=int(os.getenv("PORT", "7876")))
