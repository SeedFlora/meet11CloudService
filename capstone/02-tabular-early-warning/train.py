"""Langkah 2-3 - latih & evaluasi: siswa berisiko tidak lulus (G3 < 10) setelah nilai periode 1 (G1) keluar.

Sama di semua OS:  python train.py   (macOS/Linux: python3)
Hasil: model.joblib, metrics.json, evaluasi.png di folder ini.
"""
import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

HERE = Path(__file__).resolve().parent
SEED = 42


def scores(y_true, y_pred):
    return {
        "akurasi": round(accuracy_score(y_true, y_pred), 3),
        "recall_berisiko": round(recall_score(y_true, y_pred, zero_division=0), 3),
        "presisi_berisiko": round(precision_score(y_true, y_pred, zero_division=0), 3),
        "f1_berisiko": round(f1_score(y_true, y_pred, zero_division=0), 3),
    }


def main() -> None:
    df = pd.read_csv(HERE / "data" / "student-mat.csv", sep=";")
    y = (df["G3"] < 10).astype(int)  # 1 = berisiko tidak lulus
    X = df.drop(columns=["G2", "G3"])  # G2/G3 belum diketahui saat peringatan dini -> dibuang agar tidak bocor
    cat = X.select_dtypes(exclude="number").columns.tolist()  # kolom teks (yes/no, kategori)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, stratify=y, random_state=SEED)

    model = Pipeline([
        ("prep", ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore"), cat)], remainder="passthrough")),
        ("rf", RandomForestClassifier(n_estimators=400, min_samples_leaf=2, class_weight="balanced", random_state=SEED)),
    ])
    cv = cross_val_score(model, X_tr, y_tr, cv=StratifiedKFold(5, shuffle=True, random_state=SEED), scoring="f1")
    model.fit(X_tr, y_tr)

    hasil = {
        "baseline_mayoritas (semua lulus)": scores(y_te, [0] * len(y_te)),
        "baseline_aturan (G1 < 10)": scores(y_te, (X_te["G1"] < 10).astype(int)),
        "random_forest": scores(y_te, model.predict(X_te)),
    }
    imp = permutation_importance(model, X_te, y_te, scoring="f1", n_repeats=20, random_state=SEED)
    top = sorted(zip(X.columns, imp.importances_mean), key=lambda kv: kv[1], reverse=True)[:8]

    print(f"Data: {len(df)} siswa, {X.shape[1]} fitur, berisiko {y.mean():.1%}; uji = {len(y_te)} siswa (25%)")
    print(f"{'Model':34} {'akurasi':>8} {'recall':>7} {'presisi':>8} {'F1':>6}")
    for name, s in hasil.items():
        print(f"{name:34} {s['akurasi']:>8} {s['recall_berisiko']:>7} {s['presisi_berisiko']:>8} {s['f1_berisiko']:>6}")
    print(f"CV 5-fold F1 random forest (data latih): {cv.mean():.3f} ± {cv.std():.3f}")
    print("Fitur terpenting (permutation importance F1):", ", ".join(f"{k} {v:.3f}" for k, v in top[:5]))

    defaults = {c: (X[c].mode()[0] if c in cat else float(X[c].median())) for c in X.columns}
    joblib.dump({"model": model, "defaults": defaults, "columns": X.columns.tolist()}, HERE / "model.joblib")
    cm = confusion_matrix(y_te, model.predict(X_te)).tolist()
    (HERE / "metrics.json").write_text(json.dumps({"hasil_uji": hasil, "cv_f1_mean": round(cv.mean(), 3), "cv_f1_std": round(cv.std(), 3),
                                                  "confusion_matrix": cm, "fitur_penting": [[k, round(v, 4)] for k, v in top],
                                                  "n_siswa": len(df), "n_uji": len(y_te)}, indent=2), encoding="utf-8")

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    names = list(hasil)
    ax[0].bar(["Mayoritas", "Aturan G1<10", "Random Forest"], [hasil[n]["f1_berisiko"] for n in names], color=["#94a3b8", "#f59e0b", "#2563eb"])
    ax[0].set_title("F1 kelas 'berisiko' pada data uji")
    ax[0].set_ylim(0, 1)
    ax[1].barh([k for k, _ in top][::-1], [v for _, v in top][::-1], color="#2563eb")
    ax[1].set_title("Fitur terpenting (permutation importance)")
    fig.tight_layout()
    fig.savefig(HERE / "evaluasi.png", dpi=130)
    print("tersimpan: model.joblib, metrics.json, evaluasi.png")


if __name__ == "__main__":
    main()
