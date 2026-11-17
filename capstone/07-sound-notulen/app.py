"""Langkah 4 - web UI (Gradio): rekam/unggah suara bahasa Indonesia -> transkrip (notulen) + WER bila ada referensi.

Sama di semua OS:  python app.py   lalu buka http://127.0.0.1:7877
Port lain: PowerShell $env:PORT=7887 · cmd set PORT=7887 · macOS/Linux PORT=7887 python app.py
"""
import csv
import gzip
import json
import os
import sys
import time
from pathlib import Path

import gradio as gr
import jiwer

from asr import baca_audio, normal, pipa, transkrip

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
MODEL = {"whisper-tiny (39 juta)": "openai/whisper-tiny", "whisper-base (74 juta)": "openai/whisper-base",
         "whisper-small (244 juta)": "openai/whisper-small"}
if not (DATA / "metadata_test.csv.gz").exists():  # langkah 1 belum dijalankan
    sys.exit("data belum ada. Jalankan dulu di folder ini:  python download_data.py")
with gzip.open(DATA / "metadata_test.csv.gz", "rt", encoding="utf-8") as f:
    SAMPEL = {Path(r["path"]).stem: r for r in csv.DictReader(f) if r["language"] == "ind" and len(r["sentence"].split()) >= 6}
PILIHAN = list(SAMPEL)[::40][:8]


def ringkasan():
    baris = ["| Model | WER (20 klip) | Waktu proses / durasi audio |", "|---|---|---|"]
    for nama, mid in MODEL.items():
        f = HERE / f"metrics_{mid.split('/')[-1]}.json"
        if f.exists():
            m = json.loads(f.read_text(encoding="utf-8"))
            baris.append(f"| {nama} | {m['wer']:.1%} | {m['real_time_factor']}x |")
    return "\n".join(baris) + "\n\nWER = persentase kata salah (makin kecil makin baik). Referensi ejaan lama dinormalkan ke EYD dulu."


def transkripsikan(audio_path, sampel, model_label):
    ref = None
    if audio_path is None and sampel:
        audio_path = str(DATA / "librivox-indonesia" / SAMPEL[sampel]["path"])
        ref = SAMPEL[sampel]["sentence"]
    if audio_path is None:
        return "Rekam/unggah suara atau pilih sampel.", ""
    audio = baca_audio(audio_path)
    t0 = time.perf_counter()
    pipa(MODEL[model_label])  # memuat model: lama hanya pada panggilan pertama, sesudahnya dari memori
    muat = time.perf_counter() - t0
    t1 = time.perf_counter()
    teks = transkrip(MODEL[model_label], audio)
    detik = time.perf_counter() - t1
    info = (f"Durasi audio {len(audio) / 16000:.1f} s · muat model {muat:.1f} s · proses {detik:.1f} s di CPU"
            f" · {model_label}")
    if ref:
        wer = jiwer.wer(normal(ref, ejaan_lama=True), normal(teks))
        info += f"\n\n**Referensi (ejaan lama):** {ref}\n\n**WER klip ini:** {wer:.0%}"
    return teks, info


with gr.Blocks(title="Notulen Otomatis") as demo:
    gr.Markdown("# Notulen Otomatis (Speech-to-Text)\nSound · Whisper berjalan di CPU · sampel rekaman LibriVox Indonesia (domain publik)")
    with gr.Row():
        with gr.Column():
            rekaman = gr.Audio(sources=["microphone", "upload"], type="filepath", label="Rekam atau unggah (wav/mp3)")
            sampel = gr.Dropdown(PILIHAN, value=PILIHAN[0], label="...atau pilih sampel data uji")
            model = gr.Radio(list(MODEL), value="whisper-base (74 juta)", label="Model (makin besar makin akurat, makin lambat)")
            tombol = gr.Button("Transkripsikan", variant="primary")
        with gr.Column():
            hasil = gr.Textbox(label="Transkrip", lines=4)
            info = gr.Markdown()
    gr.Markdown(ringkasan())
    tombol.click(transkripsikan, [rekaman, sampel, model], [hasil, info])

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=int(os.getenv("PORT", "7877")))
