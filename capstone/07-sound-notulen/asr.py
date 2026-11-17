"""Fungsi bersama: baca audio tanpa ffmpeg, transkripsi Whisper, normalisasi ejaan lama, hitung WER."""
import re
import unicodedata
from functools import lru_cache
from math import gcd

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly


def baca_audio(path, sr_target=16000):
    """mp3/wav/flac/ogg -> mono float32 16 kHz (soundfile, tanpa ffmpeg)."""
    data, sr = sf.read(path, dtype="float32", always_2d=True)
    mono = data.mean(axis=1)
    if sr != sr_target:
        g = gcd(sr, sr_target)
        mono = resample_poly(mono, sr_target // g, sr // g).astype(np.float32)
    return mono


@lru_cache(maxsize=3)
def pipa(model_id):
    from transformers import pipeline
    from transformers.utils import logging as hf_logging

    hf_logging.set_verbosity_error()  # sembunyikan peringatan internal transformers (tidak memengaruhi hasil transkrip)
    return pipeline("automatic-speech-recognition", model=model_id, device="cpu")


def transkrip(model_id, audio16k):
    out = pipa(model_id)({"raw": audio16k, "sampling_rate": 16000},
                         generate_kwargs={"language": "indonesian", "task": "transcribe",
                                          "no_repeat_ngram_size": 4})  # cegah "olah olah olah ..." berulang
    return out["text"].strip()


EJAAN_LAMA = [("tj", "c"), ("dj", "J"), ("j", "y"), ("J", "j"), ("oe", "u"), ("nj", "ny"), ("sj", "sy"), ("ch", "kh")]


def normal(teks, ejaan_lama=False):
    t = unicodedata.normalize("NFKD", teks.lower())
    t = "".join(c for c in t if not unicodedata.combining(c))  # "boléh" -> "boleh"
    if ejaan_lama:  # rekaman LibriVox memakai ejaan lama: "itoe" -> "itu", "djoega" -> "juga"
        for a, b in EJAAN_LAMA:
            t = t.replace(a, b)
    t = re.sub(r"[^a-z0-9 ]+", " ", t)
    return re.sub(r"\s+", " ", t).strip()
