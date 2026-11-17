"""Langkah 2-3 - latih & evaluasi klasifikasi sentimen ulasan (IndoNLU SmSA).

Sama di semua OS:  python train.py              (TF-IDF + Logistic Regression, < 1 menit)
                   python train.py --embedding  (tambahan: embedding multibahasa + LogReg, beberapa menit di CPU)
Hasil: model.joblib, metrics.json di folder ini.
"""
import argparse
import json
from pathlib import Path
from time import perf_counter

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.pipeline import Pipeline

HERE = Path(__file__).resolve().parent
LABELS = ["negative", "neutral", "positive"]


def load(name):
    return pd.read_csv(HERE / "data" / name, sep="\t", header=None, names=["teks", "label"], quoting=3)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--embedding", action="store_true", help="bandingkan dengan embedding multibahasa (lebih lambat)")
    args = ap.parse_args()
    train, valid = load("train_preprocess.tsv"), load("valid_preprocess.tsv")
    hasil = {}

    mayoritas = train["label"].mode()[0]
    hasil["baseline_mayoritas"] = round(f1_score(valid["label"], [mayoritas] * len(valid), average="macro"), 3)

    t0 = perf_counter()
    tfidf = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)),
        ("lr", LogisticRegression(max_iter=2000, C=4.0, class_weight="balanced")),
    ])
    tfidf.fit(train["teks"], train["label"])
    pred = tfidf.predict(valid["teks"])
    hasil["tfidf_logreg"] = round(f1_score(valid["label"], pred, average="macro"), 3)
    detik_tfidf = perf_counter() - t0
    per_kelas = classification_report(valid["label"], pred, labels=LABELS, output_dict=True, zero_division=0)

    if args.embedding:
        from sentence_transformers import SentenceTransformer
        t0 = perf_counter()
        enc = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
        xtr = enc.encode(train["teks"].tolist(), batch_size=64, show_progress_bar=True)
        xva = enc.encode(valid["teks"].tolist(), batch_size=64)
        lr = LogisticRegression(max_iter=3000, class_weight="balanced").fit(xtr, train["label"])
        hasil["embedding_logreg"] = round(f1_score(valid["label"], lr.predict(xva), average="macro"), 3)
        hasil["embedding_detik"] = round(perf_counter() - t0, 1)

    print(f"Data latih {len(train)} ulasan, validasi {len(valid)}; label {train['label'].value_counts().to_dict()}")
    print(f"macro-F1 baseline mayoritas ({mayoritas}): {hasil['baseline_mayoritas']}")
    print(f"macro-F1 TF-IDF + LogReg: {hasil['tfidf_logreg']}  (latih {detik_tfidf:.1f} s)")
    for lab in LABELS:
        print(f"  F1 {lab:9}: {per_kelas[lab]['f1-score']:.3f}")
    if "embedding_logreg" in hasil:
        print(f"macro-F1 embedding multibahasa + LogReg: {hasil['embedding_logreg']}  ({hasil['embedding_detik']} s)")

    joblib.dump(tfidf, HERE / "model.joblib")
    (HERE / "metrics.json").write_text(json.dumps({
        "macro_f1": hasil, "f1_per_kelas": {k: round(per_kelas[k]["f1-score"], 3) for k in LABELS},
        "confusion_matrix": confusion_matrix(valid["label"], pred, labels=LABELS).tolist(), "labels": LABELS,
        "n_latih": len(train), "n_validasi": len(valid)}, indent=2), encoding="utf-8")
    print("tersimpan: model.joblib, metrics.json")


if __name__ == "__main__":
    main()
