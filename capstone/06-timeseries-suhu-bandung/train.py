"""Langkah 2-3 - latih & evaluasi prakiraan suhu 24 jam ke depan: dibuat pukul 23.00 untuk jam 00.00-23.00 esok hari.

Ide model: baseline musiman (suhu kemarin di jam yang sama) sudah kuat untuk Bandung,
jadi model belajar KOREKSI (residual) terhadap baseline itu dari kondisi 24 jam terakhir.

Sama di semua OS:  python train.py   (macOS/Linux: python3)
Hasil: model.joblib, metrics.json, prediksi_uji.csv, evaluasi.png di folder ini.
"""
import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

HERE = Path(__file__).resolve().parent
HARI_UJI = 120  # hari terakhir = data uji (tidak pernah dilihat model saat latih)
FITUR = ["h", "jam", "doy_sin", "doy_cos", "suhu_t", "d24", "d1", "rata24", "rata24_kemarin", "lembap_t",
         "dlembap24", "hujan24", "maks24", "min24", "kemarin", "dua_hari", "seminggu", "rata3hari"]


def susun(df):
    """Satu baris per (waktu prakiraan t0 = 23.00, horizon h = 1..24); fitur hanya memakai data <= t0."""
    s, rh, hujan = df["suhu"].to_numpy(), df["kelembapan"].to_numpy(), df["hujan"].to_numpy()
    rows = []
    for t0 in pd.date_range(df.index[0].normalize() + pd.Timedelta(days=8), df.index[-1], freq="D") - pd.Timedelta(hours=1):
        i = df.index.get_loc(t0)
        if i + 24 >= len(s):
            break
        base = {"t0": t0, "suhu_t": s[i], "d24": s[i] - s[i - 24], "d1": s[i] - s[i - 1], "rata24": s[i - 23:i + 1].mean(),
                "rata24_kemarin": s[i - 47:i - 23].mean(), "lembap_t": rh[i], "dlembap24": rh[i] - rh[i - 24],
                "hujan24": hujan[i - 23:i + 1].sum(), "maks24": s[i - 23:i + 1].max(), "min24": s[i - 23:i + 1].min()}
        for h in range(1, 25):
            j, waktu = i + h, df.index[i + h]
            doy = waktu.day_of_year / 365.25 * 2 * np.pi
            rows.append(base | {"h": h, "waktu": waktu, "jam": waktu.hour, "doy_sin": np.sin(doy), "doy_cos": np.cos(doy),
                                "kemarin": s[j - 24], "dua_hari": s[j - 48], "seminggu": s[j - 168],
                                "rata3hari": np.mean([s[j - 24], s[j - 48], s[j - 72]]), "aktual": s[j]})
    return pd.DataFrame(rows).dropna()


def main() -> None:
    df = pd.read_csv(HERE / "data" / "bandung_hourly.csv", parse_dates=["waktu"]).set_index("waktu").asfreq("h")
    df = df.interpolate(limit=3)
    data = susun(df)
    batas = data["t0"].max() - pd.Timedelta(days=HARI_UJI - 1)
    latih, uji = data[data["t0"] < batas], data[data["t0"] >= batas].copy()

    model = HistGradientBoostingRegressor(max_iter=500, learning_rate=0.05, max_leaf_nodes=15, l2_regularization=1.0,
                                          random_state=42).fit(latih[FITUR], latih["aktual"] - latih["kemarin"])
    uji["model"] = uji["kemarin"] + model.predict(uji[FITUR])
    uji["musiman"], uji["persistensi"] = uji["kemarin"], uji["suhu_t"]

    def mae(col, q=None):
        part = uji if q is None else uji[uji["h"] <= q]
        return round(float((part[col] - part["aktual"]).abs().mean()), 3)

    mae24 = {k: mae(k) for k in ("persistensi", "musiman", "model")}
    mae6 = {k: mae(k, 6) for k in ("persistensi", "musiman", "model")}
    print(f"Data {len(df)} jam ({df.index[0]:%Y-%m-%d} s.d. {df.index[-1]:%Y-%m-%d}); "
          f"latih {latih['t0'].nunique()} hari, uji {uji['t0'].nunique()} hari terakhir")
    print(f"{'MAE (°C) pada data uji':38} {'jam 1-6':>8} {'jam 1-24':>9}")
    for k, label in (("persistensi", "persistensi (suhu jam terakhir)"), ("musiman", "musiman (kemarin, jam yang sama)"),
                     ("model", "model residual gradient boosting")):
        print(f"  {label:36} {mae6[k]:>8} {mae24[k]:>9}")
    print(f"  model vs musiman: {1 - mae6['model'] / mae6['musiman']:.0%} lebih baik (jam 1-6), "
          f"{1 - mae24['model'] / mae24['musiman']:.0%} lebih baik (jam 1-24)")

    joblib.dump(model, HERE / "model.joblib")
    uji[["t0", "waktu", "h", "aktual", "model", "musiman", "persistensi"]].to_csv(HERE / "prediksi_uji.csv", index=False)
    per_h = {k: uji.assign(e=(uji[k] - uji["aktual"]).abs()).groupby("h")["e"].mean().round(3).tolist()
             for k in ("persistensi", "musiman", "model")}
    (HERE / "metrics.json").write_text(json.dumps({"mae_jam_1_24": mae24, "mae_jam_1_6": mae6, "mae_per_horizon": per_h,
                                                  "hari_latih": int(latih["t0"].nunique()), "hari_uji": int(uji["t0"].nunique()),
                                                  "periode": [f"{df.index[0]:%Y-%m-%d}", f"{df.index[-1]:%Y-%m-%d}"]}, indent=2),
                                       encoding="utf-8")
    fig, ax = plt.subplots(figsize=(8, 3.6))
    for k, c in (("musiman", "#f59e0b"), ("model", "#2563eb")):
        ax.plot(range(1, 25), per_h[k], label=k, color=c, lw=2.2)
    ax.set_xlabel("jam ke depan")
    ax.set_ylabel("MAE (°C)")
    ax.set_title(f"Error per horizon, {uji['t0'].nunique()} hari uji (persistensi 3+ °C tidak digambar)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(HERE / "evaluasi.png", dpi=130)
    print("tersimpan: model.joblib, metrics.json, prediksi_uji.csv, evaluasi.png")


if __name__ == "__main__":
    main()
