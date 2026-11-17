"""Langkah 2-3 - evaluasi speech-to-text Whisper pada rekaman bahasa Indonesia (tanpa melatih ulang).

Sama di semua OS:  python evaluate.py --model openai/whisper-small --n 20
Hasil: metrics_<model>.json (WER, waktu proses, real-time factor) di folder ini.
"""
import argparse
import csv
import gzip
import json
import time
from pathlib import Path

import jiwer

from asr import baca_audio, normal, transkrip

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"


def sampel(n):
    with gzip.open(DATA / "metadata_test.csv.gz", "rt", encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r["language"] == "ind"]
    rows = [r for r in rows if len(r["sentence"].split()) >= 6]  # kalimat cukup panjang
    step = max(1, len(rows) // n)
    return rows[::step][:n]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="openai/whisper-small")
    ap.add_argument("--n", type=int, default=20)
    args = ap.parse_args()
    rows = sampel(args.n)
    ref, hyp, durasi, proses = [], [], 0.0, 0.0
    transkrip(args.model, baca_audio(DATA / "librivox-indonesia" / rows[0]["path"]))  # pemanasan: muat model
    for r in rows:
        audio = baca_audio(DATA / "librivox-indonesia" / r["path"])
        t0 = time.perf_counter()
        teks = transkrip(args.model, audio)
        proses += time.perf_counter() - t0
        durasi += len(audio) / 16000
        ref.append(normal(r["sentence"], ejaan_lama=True))
        hyp.append(normal(teks))
    wer = jiwer.wer(ref, hyp)
    hasil = {"model": args.model, "n_klip": len(rows), "durasi_audio_s": round(durasi, 1), "waktu_proses_s": round(proses, 1),
             "real_time_factor": round(proses / durasi, 2), "wer": round(wer, 3),
             "contoh": [{"referensi": a, "transkrip": b} for a, b in list(zip(ref, hyp))[:5]]}
    print(f"{args.model}: {len(rows)} klip, audio {durasi:.0f} s, proses {proses:.0f} s "
          f"(RTF {hasil['real_time_factor']}), WER {wer:.1%}")
    name = args.model.split("/")[-1]
    (HERE / f"metrics_{name}.json").write_text(json.dumps(hasil, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
