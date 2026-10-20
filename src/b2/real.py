"""
Rama B: evaluación predictiva retrospectiva con datos FRED (sin acciones ni IPW).

Y_t (índice, no pérdida de bienestar) = sum_{h=1..12} infl_m[t+h] + 1.0 * mean_{h=1..12} unrate[t+h]
  = columna loss_cum_lam100 de 03_feature_engineering.py (se recalcula y se verifica).

Origen de predicción: final del mes t. Variante lagL (L=1 principal, L=2 sensibilidad):
  - 12 predictores: infl_m, infl_yoy, unrate, fedfunds, indpro_g_yoy en t-L y t-L-11,
    más std móvil 12m (ddof=1) de infl_m y unrate terminando en t-L.
  - Sin tratamiento, sin dummies de acción, sin variables del mes t, sin NBER.
  - Etiqueta del origen r se conoce al cierre de r+12; entra al ajuste solo si r+12 <= t_eval - L.
  - Folds internos: TimeSeriesSplit(5, gap=11+L) sobre orígenes mensuales consecutivos,
    con verificación por fechas max(label_end_train) <= primer_origen_validación - L meses.
Referencias: media de Y del entrenamiento permitido (fija en cada bloque) y persistencia
  P_t = sum infl_m[t-L-11 .. t-L] + mean unrate[t-L-11 .. t-L].
"""
from __future__ import annotations

import json
import subprocess
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

from .common import ROOT, candidates, make_model, mae, rmse, select

MODELS = ("ridge", "rf")
PRED_COLS = ["pred_media", "pred_persistencia", "pred_ridge", "pred_rf"]


def build_derived(out_dir, cfg):
    """Reconstruye base_macro_fred_unida.csv y base_macro_features.csv desde los 4 CSV FRED
    con los scripts originales (merge + 03_feature_engineering.py)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    raw = ROOT / cfg["inputs"]["fred_dir"]
    py = sys.executable
    subprocess.run([py, "-I", str(ROOT / "src/merge_bases_de_datos.py"), str(raw), str(out_dir)], check=True)
    subprocess.run([py, "-I", str(ROOT / "src/03_feature_engineering.py"),
                    str(out_dir / "base_macro_fred_unida.csv"), str(out_dir / "base_macro_features.csv")], check=True)
    return out_dir / "base_macro_features.csv"


def months(d, k):
    return (pd.Timestamp(d) + pd.DateOffset(months=k))


def load_base(path, cfg):
    df = pd.read_csv(path, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
    raw_cols = ["unrate", "cpi", "fedfunds", "indpro"]
    if df[raw_cols].isna().any().any():
        bad = df.loc[df[raw_cols].isna().any(axis=1), "date"].dt.date.tolist()
        raise ValueError(f"Faltantes inesperados en las series FRED: {bad[:10]}")
    if not (df["date"].diff().dropna().dt.days.between(28, 31)).all():
        raise ValueError("La serie no es mensual consecutiva")
    H = cfg["real"]["horizon"]
    y = sum(df["infl_m"].shift(-h) for h in range(1, H + 1)) + cfg["real"]["lambda"] * (
        sum(df["unrate"].shift(-h) for h in range(1, H + 1)) / H)
    diff = np.nanmax(np.abs(y - df[cfg["real"]["target_col"]]))
    if diff > 1e-9:
        raise AssertionError(f"Y recalculado difiere de {cfg['real']['target_col']}: {diff}")
    df["Y"] = df[cfg["real"]["target_col"]]
    return df


def build_features(df, L, cfg):
    rc = cfg["real"]; W = rc["vol_window"]
    X = pd.DataFrame({"date": df["date"], "Y": df["Y"]})
    feats = []
    for s in rc["series"]:
        for k in (L, L + 11):
            X[f"{s}_t{-k}"] = df[s].shift(k); feats.append(f"{s}_t{-k}")
    for s in rc["vol_series"]:
        X[f"{s}_std12_t{-L}"] = df[s].rolling(W, min_periods=W).std(ddof=1).shift(L)
        feats.append(f"{s}_std12_t{-L}")
    X["persistencia"] = (df["infl_m"].rolling(12, min_periods=12).sum().shift(L)
                         + rc["lambda"] * df["unrate"].rolling(12, min_periods=12).mean().shift(L))
    return X, feats


def prepare(df, L, cfg):
    rc = cfg["real"]
    X, feats = build_features(df, L, cfg)
    cand = X[X["date"] >= rc["first_origin"]]
    need = feats + ["Y", "persistencia"]
    ok = cand[need].notna().all(axis=1)
    dropped = cand.loc[~ok, "date"]
    O = cand[ok].reset_index(drop=True)
    test = O[(O.date >= rc["test_start"]) & (O.date <= rc["test_end"])].reset_index(drop=True)
    limit = months(rc["test_start"], -L - rc["horizon"])        # r + 12 <= t0 - L
    dev = O[O.date <= limit].reset_index(drop=True)
    exp_months = pd.date_range(dev.date.min(), dev.date.max(), freq="MS")
    if len(exp_months) != len(dev):
        raise ValueError("Orígenes de desarrollo no consecutivos; TimeSeriesSplit(gap) no aplicaría")
    info = {"variante": f"lag{L}", "L": L, "n_origenes_completos": len(O),
            "filas_descartadas_desde_1992": [str(d.date()) for d in dropped],
            "dev_desde": str(dev.date.min().date()), "dev_hasta": str(dev.date.max().date()), "n_dev": len(dev),
            "test_desde": str(test.date.min().date()), "test_hasta": str(test.date.max().date()), "n_test": len(test),
            "limite_origen_train": str(limit.date())}
    return dev, test, feats, info


def _fit_predict(model, params, cfg, A, B, feats):
    m = make_model(cfg, model, params).fit(A[feats].to_numpy(), A["Y"].to_numpy())
    return m.predict(B[feats].to_numpy())


def fit_and_predict(dev, test, feats, L, cfg):
    """Selección en folds de desarrollo + ajuste final fijo. No lee test['Y']."""
    rc = cfg["real"]; tag = f"lag{L}"
    gap = rc["horizon"] + L - 1
    folds = list(TimeSeriesSplit(n_splits=rc["cv_n_splits"], gap=gap).split(np.arange(len(dev))))
    cuts, preds, chosen = [], [], {}
    for k, (tr, va) in enumerate(folds, 1):
        lab_end = months(dev.date.iloc[tr].max(), rc["horizon"])
        if lab_end > months(dev.date.iloc[va].min(), -L):
            raise AssertionError(f"{tag} fold {k}: etiquetas de train terminan {lab_end.date()} > primer origen val - {L}m")
    for model in MODELS:
        scores = []
        for params in candidates(cfg, model):
            yh, yt = [], []
            for k, (tr, va) in enumerate(folds, 1):
                pr = _fit_predict(model, params, cfg, dev.iloc[tr], dev.iloc[va], feats)
                yh.append(pr); yt.append(dev["Y"].iloc[va].to_numpy())
            s = rmse(np.concatenate(yt), np.concatenate(yh))
            scores.append((params, s))
            cuts.append({"rama": "real", "id": tag, "fold": "agregado", "modelo": model,
                         "hiperparametros": json.dumps(params), "rmse_seleccion": s,
                         "n_eval": int(sum(len(v) for _, v in folds))})
        chosen[model], _ = select(cfg, scores)
    # predicciones de folds con los hiperparámetros elegidos (diagnóstico, no evaluación independiente)
    for k, (tr, va) in enumerate(folds, 1):
        A, B = dev.iloc[tr], dev.iloc[va]
        P = pd.DataFrame({"variante": tag, "bloque": f"fold{k}", "date": B.date.values, "Y": B.Y.values,
                          "pred_media": A.Y.mean(), "pred_persistencia": B.persistencia.values})
        for model in MODELS:
            P[f"pred_{model}"] = _fit_predict(model, chosen[model], cfg, A, B, feats)
        P["info_hasta"] = [months(d, -L).date() for d in B.date]
        P["train_desde"], P["train_hasta"] = A.date.min().date(), A.date.max().date()
        P["etiquetas_train_hasta"] = months(A.date.max(), rc["horizon"]).date()
        preds.append(P)
        cuts.append({"rama": "real", "id": tag, "fold": k, "modelo": "ridge|rf|media|persistencia",
                     "hiperparametros": json.dumps(chosen),
                     "train_desde": str(A.date.min().date()), "train_hasta": str(A.date.max().date()),
                     "eval_desde": str(B.date.min().date()), "eval_hasta": str(B.date.max().date()),
                     "ultima_etiqueta_train": str(months(A.date.max(), rc["horizon"]).date()),
                     "n_train": len(A), "n_eval": len(B)})
    lab_end = months(dev.date.max(), rc["horizon"])
    if lab_end > months(test.date.min(), -L):
        raise AssertionError(f"{tag} ajuste final: etiquetas hasta {lab_end.date()} alcanzan el test")
    P = pd.DataFrame({"variante": tag, "bloque": "test", "date": test.date.values,
                      "pred_media": dev.Y.mean(), "pred_persistencia": test.persistencia.values})
    for model in MODELS:
        P[f"pred_{model}"] = _fit_predict(model, chosen[model], cfg, dev, test, feats)
    P["info_hasta"] = [months(d, -L).date() for d in test.date]
    P["train_desde"], P["train_hasta"] = dev.date.min().date(), dev.date.max().date()
    P["etiquetas_train_hasta"] = lab_end.date()
    preds.append(P)
    cuts.append({"rama": "real", "id": tag, "fold": "final_test", "modelo": "ridge|rf|media|persistencia",
                 "hiperparametros": json.dumps(chosen),
                 "train_desde": str(dev.date.min().date()), "train_hasta": str(dev.date.max().date()),
                 "eval_desde": str(test.date.min().date()), "eval_hasta": str(test.date.max().date()),
                 "ultima_etiqueta_train": str(lab_end.date()), "n_train": len(dev), "n_eval": len(test)})
    return chosen, cuts, preds


def attach_test_y(preds, test):
    P = preds[-1]
    P.insert(3, "Y", test.Y.values)
    return preds


def metrics(pred_df):
    rows = []
    blocks = [("test", pred_df[pred_df.bloque == "test"]),
              ("folds_desarrollo_agregado", pred_df[pred_df.bloque.str.startswith("fold")])]
    for b in sorted(pred_df.bloque.unique()):
        if b.startswith("fold"):
            blocks.append((b, pred_df[pred_df.bloque == b]))
    for var in pred_df.variante.unique():
        for name, B in blocks:
            B = B[B.variante == var]
            if B.empty:
                continue
            rp = rmse(B.Y, B.pred_persistencia)
            for c in PRED_COLS:
                r = rmse(B.Y, B[c])
                rows.append({"variante": var, "modelo": c.replace("pred_", ""), "bloque": name, "n": len(B),
                             "fecha_desde": str(pd.Timestamp(B.date.min()).date()),
                             "fecha_hasta": str(pd.Timestamp(B.date.max()).date()),
                             "rmse": r, "mae": mae(B.Y, B[c]),
                             "mejora_rmse_vs_persistencia": 1.0 - r / rp,
                             "uso": "evaluacion_historica_reutilizada" if name == "test"
                                    else "diagnostico_seleccion (no evaluacion independiente)"})
    return pd.DataFrame(rows)


def run_variant(df, L, cfg):
    dev, test, feats, info = prepare(df, L, cfg)
    chosen, cuts, preds = fit_and_predict(dev, test.drop(columns=["Y"]).assign(Y=np.nan), feats, L, cfg)
    preds = attach_test_y(preds, test)
    info["features"] = feats
    info["hiperparametros"] = chosen
    return info, cuts, pd.concat(preds, ignore_index=True)
