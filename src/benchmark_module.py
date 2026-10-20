#!/usr/bin/env python3
"""
benchmark_module.py
===================

Modulo unico de benchmarking predictivo orientado a decision para estimar
E[L | X, T] con dos benchmarks (Ridge con interacciones y Random Forest),
construir h(X) = argmin_T mu_hat(X, T) y evaluar en datos sinteticos y reales.

Respeta la especificacion cerrada del proyecto:
    - sintetico: X = (inflation, unemployment, fedfunds, activity), target=loss
    - real (FRED): 9 features de nivel/regimen + 16 lags = 25 base, target=loss_cum_lam100
    - tratamiento real derivado de dfedfunds con threshold 0.125
    - particion temporal 70/15/15, buffer 0 (sint) / 12 (real)
    - Ridge alpha en {0.01, 0.1, 1, 10, 100} con StandardScaler en train
    - RF n_estimators=300 fijo, max_depth en {None,5,10,20}, min_samples_leaf en {1,5,20}
    - tuning con TimeSeriesSplit (5 folds), seleccion por RMSE factual
    - propensity multinomial logit (C=1.0) sobre las 25 features sin T
    - IPW value para evaluacion real, regret oraculo para sintetico
    - empate si |diff_ipw| < 0.5% baseline_observed -> preferir Ridge

Uso:
    python benchmark_module.py --mode synthetic --input base_sintetica.csv         --output ./outputs_synth
    python benchmark_module.py --mode real      --input base_macro_features.csv    --output ./outputs_real
"""
from __future__ import annotations

import argparse
import json
import sys
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler


# =============================================================================
# Constantes (no editar salvo cambio explicito de spec)
# =============================================================================
T_LEVELS = ("T0", "T1", "T2")
RANDOM_STATE = 42

# Particion temporal
TRAIN_FRAC = 0.70
VAL_FRAC = 0.15
SYNTH_BUFFER = 0
REAL_BUFFER = 12

# Sintetico
SYNTH_BASE_FEATURES = ["inflation", "unemployment", "fedfunds", "activity"]
SYNTH_TREATMENT_COL = "treatment"
SYNTH_TARGET = "loss"
SYNTH_EVAL_COLS = {"T0": "loss_T0", "T1": "loss_T1", "T2": "loss_T2",
                   "true_optimal": "true_optimal_treatment"}
SYNTH_DATE_COL = "date"

# Real (FRED)
REAL_DATE_COL = "date"
REAL_TARGET = "loss_cum_lam100"
REAL_TREATMENT_THRESHOLD = 0.125

# Mapeo logico (spec) -> nombre real esperado en base_macro_features.csv
REAL_COLUMN_MAP = {
    "inflation_m":         "infl_m",
    "inflation_yoy":       "infl_yoy",
    "unemployment":        "unrate",
    "fedfunds":            "fedfunds",
    "ip_growth":           "indpro_g_yoy",
    "dfedfunds":           "fedfunds_diff",
    "vol_inflation_12":    "infl_m_std12",
    "vol_unemployment_12": "unrate_std12",   # si no existe, se calcula
    "recession":           "nber_recession",
}
REAL_LEVEL_FEATURES = [
    "inflation_m", "inflation_yoy", "unemployment", "fedfunds",
    "ip_growth", "dfedfunds", "vol_inflation_12", "vol_unemployment_12",
    "recession",
]
REAL_LAG_BASE = ["inflation_m", "unemployment", "fedfunds", "ip_growth"]
REAL_LAG_PERIODS = [1, 3, 6, 12]

# Hiperparametros
RIDGE_ALPHA_GRID = [0.01, 0.1, 1.0, 10.0, 100.0]
RF_PARAM_GRID = {
    "n_estimators": [300],
    "max_depth":    [None, 5, 10, 20],
    "min_samples_leaf": [1, 5, 20],
}
CV_N_SPLITS = 5
PROPENSITY_C = 1.0
PROPENSITY_MAX_ITER = 2000

# Comparacion
TIE_TOLERANCE_FRAC = 0.005


# =============================================================================
# 1. Carga y validacion
# =============================================================================
def _validate_columns(df, required, context):
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"[{context}] columnas faltantes: {missing}")


def _normalize_treatment_strings(df, col):
    """0/1/2 -> 'T0'/'T1'/'T2'; valida estricto."""
    s = df[col]
    df = df.copy()
    if pd.api.types.is_numeric_dtype(s):
        df[col] = s.astype(int).map({0: "T0", 1: "T1", 2: "T2"})
        if df[col].isna().any():
            raise ValueError(f"Valores fuera de {{0,1,2}} en '{col}'.")
    else:
        df[col] = s.astype(str).str.strip()
    invalid = ~df[col].isin(T_LEVELS)
    if invalid.any():
        raise ValueError(f"Valores invalidos en '{col}': "
                         f"{df.loc[invalid, col].unique().tolist()}")
    return df


def load_synthetic(path):
    """Carga base sintetica y normaliza tratamientos."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"No existe: {p}")
    df = pd.read_csv(p)
    if SYNTH_DATE_COL in df.columns:
        df[SYNTH_DATE_COL] = pd.to_datetime(df[SYNTH_DATE_COL])
        df = df.sort_values(SYNTH_DATE_COL).reset_index(drop=True)

    required = (SYNTH_BASE_FEATURES
                + [SYNTH_TREATMENT_COL, SYNTH_TARGET]
                + list(SYNTH_EVAL_COLS.values()))
    _validate_columns(df, required, "base sintetica")

    df = _normalize_treatment_strings(df, SYNTH_TREATMENT_COL)
    df = _normalize_treatment_strings(df, SYNTH_EVAL_COLS["true_optimal"])
    return df


def load_real(path):
    """Carga base real (FRED), aplica REAL_COLUMN_MAP, valida y construye
    features faltantes (vol_unemployment_12, lags, treatment T0/T1/T2)."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"No existe: {p}")
    df_raw = pd.read_csv(p)
    if REAL_DATE_COL in df_raw.columns:
        df_raw[REAL_DATE_COL] = pd.to_datetime(df_raw[REAL_DATE_COL])
        df_raw = df_raw.sort_values(REAL_DATE_COL).reset_index(drop=True)

    # Renombrado fisico->logico
    rename_map = {v: k for k, v in REAL_COLUMN_MAP.items() if v in df_raw.columns}
    df = df_raw.rename(columns=rename_map).copy()

    # Columnas logicas que pueden auto-construirse si faltan
    auto_buildable = {"vol_unemployment_12", "dfedfunds", "vol_inflation_12", "recession"}
    hard_missing = [c for c in REAL_LEVEL_FEATURES
                    if c not in df.columns and c not in auto_buildable]
    if hard_missing:
        raise KeyError(f"[base real] columnas logicas faltantes: {hard_missing}. "
                       f"Editar REAL_COLUMN_MAP o agregar al CSV.")

    if "dfedfunds" not in df.columns:
        if "fedfunds" not in df.columns:
            raise KeyError("Falta 'fedfunds' para construir dfedfunds.")
        df["dfedfunds"] = df["fedfunds"].diff()

    if "vol_inflation_12" not in df.columns:
        if "inflation_m" not in df.columns:
            raise KeyError("Falta 'inflation_m' para construir vol_inflation_12.")
        df["vol_inflation_12"] = (
            df["inflation_m"].rolling(window=12, min_periods=12).std()
        )

    if "recession" not in df.columns:
        # NBER episodios oficiales (segun handoff del proyecto)
        if REAL_DATE_COL not in df.columns:
            raise KeyError("Falta 'date' para construir recession.")
        nber = [
            ("1990-07-01", "1991-03-31"),
            ("2001-03-01", "2001-11-30"),
            ("2007-12-01", "2009-06-30"),
            ("2020-02-01", "2020-04-30"),
        ]
        d = df[REAL_DATE_COL]
        flag = pd.Series(0, index=df.index, dtype=int)
        for a, b in nber:
            flag |= ((d >= pd.Timestamp(a)) & (d <= pd.Timestamp(b))).astype(int)
        df["recession"] = flag

    if "vol_unemployment_12" not in df.columns:
        if "unemployment" not in df.columns:
            raise KeyError("Falta 'unemployment' para construir vol_unemployment_12.")
        df["vol_unemployment_12"] = (
            df["unemployment"].rolling(window=12, min_periods=12).std()
        )

    # Lags: usar columnas del archivo si existen con nombre fisico, sino computar
    for base in REAL_LAG_BASE:
        if base not in df.columns:
            raise KeyError(f"Falta columna base para lags: {base}")
        for k in REAL_LAG_PERIODS:
            lag_col = f"{base}_lag{k}"
            phys = f"{REAL_COLUMN_MAP[base]}_lag{k}"
            if lag_col in df.columns:
                continue
            if phys in df_raw.columns:
                df[lag_col] = df_raw[phys].values
            else:
                df[lag_col] = df[base].shift(k)

    # Target
    if REAL_TARGET not in df.columns:
        raise KeyError(f"Target {REAL_TARGET} ausente en la base real.")

    # Tratamiento desde dfedfunds
    df = _build_real_treatment(df)

    # Drop NaN (lags al inicio, target forward al final)
    feats = real_base_features()
    needed = feats + [REAL_TARGET, "treatment", "T1", "T2", REAL_DATE_COL]
    df = df.dropna(subset=needed).reset_index(drop=True)
    return df


def _build_real_treatment(df):
    th = REAL_TREATMENT_THRESHOLD
    d = df["dfedfunds"]
    treatment = pd.Series(np.where(d > th, "T1",
                          np.where(d < -th, "T2", "T0")),
                          index=df.index, name="treatment")
    df = df.copy()
    df["treatment"] = treatment
    df["T1"] = (treatment == "T1").astype(int)
    df["T2"] = (treatment == "T2").astype(int)
    return df


# =============================================================================
# 2. Listas de features
# =============================================================================
def synth_base_features():
    return list(SYNTH_BASE_FEATURES)


def real_base_features():
    feats = list(REAL_LEVEL_FEATURES)
    for base in REAL_LAG_BASE:
        for k in REAL_LAG_PERIODS:
            feats.append(f"{base}_lag{k}")
    return feats


def attach_treatment_dummies(df, treatment_col):
    if {"T1", "T2"}.issubset(df.columns):
        return df
    df = df.copy()
    df["T1"] = (df[treatment_col] == "T1").astype(int)
    df["T2"] = (df[treatment_col] == "T2").astype(int)
    return df


# =============================================================================
# 3. Matriz de diseno (factual o contrafactual)
# =============================================================================
def build_design_matrix(df, base_features, model_type, treatment_value=None):
    """
        rf:    [X, T1, T2]              (k+2)
        ridge: [X, T1, T2, X*T1, X*T2]  (3k+2)
    Si treatment_value es None usa dummies del df; sino fuerza T0/T1/T2.
    """
    X_base = df[base_features].to_numpy(dtype=float)
    if treatment_value is None:
        if not {"T1", "T2"}.issubset(df.columns):
            raise KeyError("Faltan dummies T1/T2 para tratamiento observado.")
        T1 = df["T1"].to_numpy(dtype=float)
        T2 = df["T2"].to_numpy(dtype=float)
    elif treatment_value == "T0":
        T1, T2 = np.zeros(len(df)), np.zeros(len(df))
    elif treatment_value == "T1":
        T1, T2 = np.ones(len(df)),  np.zeros(len(df))
    elif treatment_value == "T2":
        T1, T2 = np.zeros(len(df)), np.ones(len(df))
    else:
        raise ValueError(f"treatment_value invalido: {treatment_value}")
    T1c, T2c = T1.reshape(-1, 1), T2.reshape(-1, 1)
    if model_type == "rf":
        return np.hstack([X_base, T1c, T2c])
    if model_type == "ridge":
        return np.hstack([X_base, T1c, T2c, X_base * T1c, X_base * T2c])
    raise ValueError(f"model_type invalido: {model_type}")


# =============================================================================
# 4. Split temporal
# =============================================================================
def temporal_split(n, train_frac=TRAIN_FRAC, val_frac=VAL_FRAC, buffer=0):
    n_train = int(np.floor(n * train_frac))
    n_val = int(np.floor(n * val_frac))
    if n - n_train - n_val <= 0:
        raise ValueError("Sin filas para test; revisar fracciones.")
    train_idx = np.arange(0, n_train)
    val_start = n_train + buffer
    val_end = val_start + n_val
    test_start = val_end + buffer
    if test_start >= n:
        raise ValueError("Buffer excesivo; no quedan filas para test.")
    return train_idx, np.arange(val_start, val_end), np.arange(test_start, n)


# =============================================================================
# 5. Tuning + entrenamiento
# =============================================================================
def _rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def _ts_splits(n_rows, n_splits=CV_N_SPLITS):
    return list(TimeSeriesSplit(n_splits=n_splits).split(np.arange(n_rows)))


def tune_ridge(df_train, base_features, target):
    splits = _ts_splits(len(df_train))
    cv = {}
    for alpha in RIDGE_ALPHA_GRID:
        fold_rmse = []
        for tr_idx, va_idx in splits:
            df_tr, df_va = df_train.iloc[tr_idx], df_train.iloc[va_idx]
            X_tr = build_design_matrix(df_tr, base_features, "ridge")
            X_va = build_design_matrix(df_va, base_features, "ridge")
            sc = StandardScaler().fit(X_tr)
            m = Ridge(alpha=alpha, random_state=RANDOM_STATE)
            m.fit(sc.transform(X_tr), df_tr[target].to_numpy())
            fold_rmse.append(_rmse(df_va[target].to_numpy(),
                                   m.predict(sc.transform(X_va))))
        cv[alpha] = float(np.mean(fold_rmse))
    return {"best_alpha": min(cv, key=cv.get), "cv_rmse_per_alpha": cv}


def tune_rf(df_train, base_features, target):
    splits = _ts_splits(len(df_train))
    keys = list(RF_PARAM_GRID.keys())
    combos = [dict(zip(keys, v)) for v in product(*[RF_PARAM_GRID[k] for k in keys])]
    results = []
    for params in combos:
        fold_rmse = []
        for tr_idx, va_idx in splits:
            df_tr, df_va = df_train.iloc[tr_idx], df_train.iloc[va_idx]
            X_tr = build_design_matrix(df_tr, base_features, "rf")
            X_va = build_design_matrix(df_va, base_features, "rf")
            m = RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1, **params)
            m.fit(X_tr, df_tr[target].to_numpy())
            fold_rmse.append(_rmse(df_va[target].to_numpy(), m.predict(X_va)))
        results.append({"params": params, "cv_rmse": float(np.mean(fold_rmse))})
    results.sort(key=lambda d: d["cv_rmse"])
    return {"best_params": results[0]["params"], "cv_results": results}


def fit_ridge_final(df_fit, base_features, target, alpha):
    X = build_design_matrix(df_fit, base_features, "ridge")
    sc = StandardScaler().fit(X)
    m = Ridge(alpha=alpha, random_state=RANDOM_STATE)
    m.fit(sc.transform(X), df_fit[target].to_numpy())
    return m, sc


def fit_rf_final(df_fit, base_features, target, params):
    X = build_design_matrix(df_fit, base_features, "rf")
    m = RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1, **params)
    m.fit(X, df_fit[target].to_numpy())
    return m


# =============================================================================
# 6. Prediccion factual + contrafactual + h(X)
# =============================================================================
def predict_factual(model, df, base_features, model_type, scaler=None):
    X = build_design_matrix(df, base_features, model_type)
    if model_type == "ridge":
        if scaler is None:
            raise ValueError("Ridge requiere scaler.")
        X = scaler.transform(X)
    return model.predict(X)


def predict_counterfactual(model, df, base_features, model_type, scaler=None,
                           t_levels=T_LEVELS):
    cols = []
    for t in t_levels:
        X = build_design_matrix(df, base_features, model_type, treatment_value=t)
        if model_type == "ridge":
            if scaler is None:
                raise ValueError("Ridge requiere scaler.")
            X = scaler.transform(X)
        cols.append(model.predict(X))
    return np.column_stack(cols)


def decision_rule(mu_matrix, t_levels=T_LEVELS):
    if mu_matrix.shape[1] != len(t_levels):
        raise ValueError("Columnas de mu_matrix != len(t_levels).")
    return np.array(t_levels)[np.argmin(mu_matrix, axis=1)]


# =============================================================================
# 7. Metricas
# =============================================================================
def predictive_metrics(y_true, y_pred):
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae":  float(mean_absolute_error(y_true, y_pred)),
        "r2":   float(r2_score(y_true, y_pred)),
    }


def synthetic_decision_metrics(decisions, true_optimal, loss_T0, loss_T1, loss_T2):
    decisions = np.asarray(decisions); true_optimal = np.asarray(true_optimal)
    L = np.column_stack([loss_T0, loss_T1, loss_T2])
    idx = {t: i for i, t in enumerate(T_LEVELS)}
    rows = np.arange(len(decisions))
    loss_h = L[rows, np.array([idx[t] for t in decisions])]
    loss_o = L[rows, np.array([idx[t] for t in true_optimal])]
    regret = loss_h - loss_o
    confusion = pd.crosstab(
        pd.Series(true_optimal, name="true_optimal"),
        pd.Series(decisions,   name="h"),
        dropna=False,
    ).reindex(index=list(T_LEVELS), columns=list(T_LEVELS), fill_value=0)
    distribution = (pd.Series(decisions).value_counts(normalize=True)
                    .reindex(list(T_LEVELS), fill_value=0.0).to_dict())
    return {
        "policy_accuracy": float(np.mean(decisions == true_optimal)),
        "regret_mean":     float(np.mean(regret)),
        "regret_p90":      float(np.percentile(regret, 90)),
        "h_distribution":  distribution,
        "confusion":       confusion,
    }


def fit_propensity(X, T_strings):
    if X.shape[0] != len(T_strings):
        raise ValueError("X y T deben tener misma longitud.")
    m = LogisticRegression(
        solver="lbfgs",
        C=PROPENSITY_C, max_iter=PROPENSITY_MAX_ITER, random_state=RANDOM_STATE,
    )
    m.fit(X, T_strings)
    return m


def propensity_matrix(model, X, t_levels=T_LEVELS):
    proba = model.predict_proba(X)
    classes = list(model.classes_)
    out = np.zeros((X.shape[0], len(t_levels)))
    for j, t in enumerate(t_levels):
        if t not in classes:
            raise ValueError(f"Clase {t} ausente en propensity.")
        out[:, j] = proba[:, classes.index(t)]
    return np.clip(out, 1e-3, 1.0)


def _ipw_value(L_obs, T_obs, policy, p_matrix, t_levels=T_LEVELS):
    L_obs = np.asarray(L_obs, dtype=float); T_obs = np.asarray(T_obs)
    policy = np.asarray(policy)
    idx = {t: i for i, t in enumerate(t_levels)}
    e_obs = p_matrix[np.arange(len(T_obs)), np.array([idx[t] for t in T_obs])]
    return float(np.mean((T_obs == policy).astype(float) * L_obs / e_obs))


def real_decision_metrics(decisions, T_obs, L_obs, p_matrix, recession=None,
                          t_levels=T_LEVELS):
    decisions = np.asarray(decisions); T_obs = np.asarray(T_obs)
    L_obs = np.asarray(L_obs, dtype=float)
    coincident = (T_obs == decisions)
    factual_coincident = (float(np.mean(L_obs[coincident]))
                          if coincident.any() else float("nan"))
    ipw = _ipw_value(L_obs, T_obs, decisions, p_matrix, t_levels)
    baseline_obs = float(np.mean(L_obs))
    baseline_T = {t: _ipw_value(L_obs, T_obs,
                                np.full_like(T_obs, t, dtype=object),
                                p_matrix, t_levels) for t in t_levels}
    distribution = (pd.Series(decisions).value_counts(normalize=True)
                    .reindex(list(t_levels), fill_value=0.0).to_dict())
    distribution_by_regime = None
    if recession is not None:
        ct = pd.crosstab(
            pd.Series(np.asarray(recession), name="recession"),
            pd.Series(decisions, name="h"),
            normalize="index", dropna=False,
        ).reindex(columns=list(t_levels), fill_value=0.0)
        distribution_by_regime = ct
    return {
        "factual_loss_coincident":     factual_coincident,
        "n_coincident":                int(coincident.sum()),
        "ipw_value":                   ipw,
        "ipw_value_baseline_observed": baseline_obs,
        "ipw_value_baseline_T0":       baseline_T["T0"],
        "ipw_value_baseline_T1":       baseline_T["T1"],
        "ipw_value_baseline_T2":       baseline_T["T2"],
        "h_distribution":              distribution,
        "h_distribution_by_regime":    distribution_by_regime,
    }


def compare_synthetic(metrics_ridge, metrics_rf):
    winner = ("ridge" if metrics_ridge["regret_mean"] < metrics_rf["regret_mean"]
              else "rf")
    return {
        "diff_rmse_ridge_minus_rf":        metrics_ridge["rmse"] - metrics_rf["rmse"],
        "diff_regret_mean_ridge_minus_rf": metrics_ridge["regret_mean"] - metrics_rf["regret_mean"],
        "winner_by_regret":                winner,
    }


def compare_real(metrics_ridge, metrics_rf, baseline_observed,
                 tol_frac=TIE_TOLERANCE_FRAC):
    diff_ipw = metrics_ridge["ipw_value"] - metrics_rf["ipw_value"]
    tol = abs(baseline_observed) * tol_frac
    if abs(diff_ipw) < tol:
        winner = "tie_prefer_ridge"
    elif metrics_ridge["ipw_value"] < metrics_rf["ipw_value"]:
        winner = "ridge"
    else:
        winner = "rf"
    return {
        "diff_rmse_ridge_minus_rf":      metrics_ridge["rmse"] - metrics_rf["rmse"],
        "diff_ipw_value_ridge_minus_rf": diff_ipw,
        "tolerance_value":               float(tol),
        "winner_by_ipw":                 winner,
    }


# =============================================================================
# 8. Helpers de output
# =============================================================================
def _scalarize(v):
    if isinstance(v, dict):
        return json.dumps({k: round(float(x), 4) for k, x in v.items()})
    return v


def _save_metrics(out_dir, metrics_ridge, metrics_rf, drop_keys=()):
    rows = []
    for name, m in [("ridge", metrics_ridge), ("rf", metrics_rf)]:
        row = {"model": name}
        for k, v in m.items():
            if k in drop_keys:
                continue
            row[k] = _scalarize(v)
        rows.append(row)
    pd.DataFrame(rows).to_csv(out_dir / "metrics.csv", index=False)


def _save_predictions(out_dir, dates, mu_matrix, name):
    df = pd.DataFrame(mu_matrix, columns=[f"mu_{t}" for t in T_LEVELS])
    df.insert(0, "date", pd.Series(dates).reset_index(drop=True))
    df.to_csv(out_dir / f"predictions_{name}.csv", index=False)


# =============================================================================
# 9. Pipelines
# =============================================================================
def run_synthetic(input_path, output_dir):
    print("[1/7] Cargando base sintetica...")
    df = load_synthetic(input_path)
    df = attach_treatment_dummies(df, SYNTH_TREATMENT_COL)
    base = synth_base_features()
    print(f"      filas={len(df)}, base_features={len(base)}")

    print("[2/7] Split temporal 70/15/15 buffer=0...")
    tr, va, te = temporal_split(len(df), buffer=SYNTH_BUFFER)
    df_tr, df_va, df_te = (df.iloc[tr].reset_index(drop=True),
                           df.iloc[va].reset_index(drop=True),
                           df.iloc[te].reset_index(drop=True))
    print(f"      train={len(df_tr)} val={len(df_va)} test={len(df_te)}")

    print("[3/7] Tuning Ridge (TS-CV)...")
    rt = tune_ridge(df_tr, base, SYNTH_TARGET)
    print(f"      best_alpha={rt['best_alpha']}")

    print("[4/7] Tuning RF (TS-CV)...")
    ft = tune_rf(df_tr, base, SYNTH_TARGET)
    print(f"      best_params={ft['best_params']}")

    print("[5/7] Retrain final en train+val...")
    df_fit = pd.concat([df_tr, df_va], axis=0, ignore_index=True)
    ridge, scaler = fit_ridge_final(df_fit, base, SYNTH_TARGET, rt["best_alpha"])
    rf = fit_rf_final(df_fit, base, SYNTH_TARGET, ft["best_params"])

    print("[6/7] Predicciones en test + h(X)...")
    y_te = df_te[SYNTH_TARGET].to_numpy()
    pred_r = predict_factual(ridge, df_te, base, "ridge", scaler=scaler)
    pred_f = predict_factual(rf,    df_te, base, "rf")
    mu_r = predict_counterfactual(ridge, df_te, base, "ridge", scaler=scaler)
    mu_f = predict_counterfactual(rf,    df_te, base, "rf")
    h_r, h_f = decision_rule(mu_r), decision_rule(mu_f)

    pm_r = predictive_metrics(y_te, pred_r)
    pm_f = predictive_metrics(y_te, pred_f)
    dm_r = synthetic_decision_metrics(
        h_r, df_te[SYNTH_EVAL_COLS["true_optimal"]].to_numpy(),
        df_te[SYNTH_EVAL_COLS["T0"]].to_numpy(),
        df_te[SYNTH_EVAL_COLS["T1"]].to_numpy(),
        df_te[SYNTH_EVAL_COLS["T2"]].to_numpy(),
    )
    dm_f = synthetic_decision_metrics(
        h_f, df_te[SYNTH_EVAL_COLS["true_optimal"]].to_numpy(),
        df_te[SYNTH_EVAL_COLS["T0"]].to_numpy(),
        df_te[SYNTH_EVAL_COLS["T1"]].to_numpy(),
        df_te[SYNTH_EVAL_COLS["T2"]].to_numpy(),
    )
    metrics_r = {**pm_r, **{k: v for k, v in dm_r.items() if k != "confusion"}}
    metrics_f = {**pm_f, **{k: v for k, v in dm_f.items() if k != "confusion"}}
    comparison = compare_synthetic(metrics_r, metrics_f)

    print("[7/7] Guardando outputs...")
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    _save_metrics(out, metrics_r, metrics_f)
    dates_te = df_te[SYNTH_DATE_COL].values if SYNTH_DATE_COL in df_te.columns else np.arange(len(df_te))
    pd.DataFrame({"date": dates_te,
                  "T_obs": df_te[SYNTH_TREATMENT_COL].values,
                  "h_ridge": h_r, "h_rf": h_f}).to_csv(out / "decisions.csv", index=False)
    _save_predictions(out, dates_te, mu_r, "ridge")
    _save_predictions(out, dates_te, mu_f, "rf")
    dm_r["confusion"].to_csv(out / "confusion_ridge.csv")
    dm_f["confusion"].to_csv(out / "confusion_rf.csv")
    with open(out / "hyperparameters.json", "w") as f:
        json.dump({"ridge": rt, "rf": ft}, f, indent=2, default=str)
    with open(out / "comparison.json", "w") as f:
        json.dump(comparison, f, indent=2, default=str)
    print(f"      outputs en: {out}")
    print("\n=== Sintetico ===")
    print(pd.read_csv(out / "metrics.csv").to_string(index=False))
    print("Comparacion:", comparison)


def run_real(input_path, output_dir):
    print("[1/8] Cargando base real (FRED)...")
    df = load_real(input_path)
    base = real_base_features()
    if len(base) != 25:
        raise RuntimeError(f"Esperaba 25 base features; obtuve {len(base)}.")
    print(f"      filas validas={len(df)}, base_features={len(base)}")

    print("[2/8] Split temporal 70/15/15 buffer=12...")
    tr, va, te = temporal_split(len(df), buffer=REAL_BUFFER)
    df_tr, df_va, df_te = (df.iloc[tr].reset_index(drop=True),
                           df.iloc[va].reset_index(drop=True),
                           df.iloc[te].reset_index(drop=True))
    print(f"      train={len(df_tr)} val={len(df_va)} test={len(df_te)}")

    print("[3/8] Tuning Ridge (TS-CV)...")
    rt = tune_ridge(df_tr, base, REAL_TARGET)
    print(f"      best_alpha={rt['best_alpha']}")

    print("[4/8] Tuning RF (TS-CV)...")
    ft = tune_rf(df_tr, base, REAL_TARGET)
    print(f"      best_params={ft['best_params']}")

    print("[5/8] Retrain final en train+val...")
    df_fit = pd.concat([df_tr, df_va], axis=0, ignore_index=True)
    ridge, scaler = fit_ridge_final(df_fit, base, REAL_TARGET, rt["best_alpha"])
    rf = fit_rf_final(df_fit, base, REAL_TARGET, ft["best_params"])

    print("[6/8] Propensity multinomial (25 features sin T)...")
    Xp_fit = df_fit[base].to_numpy(dtype=float)
    Tp_fit = df_fit["treatment"].to_numpy()
    prop = fit_propensity(Xp_fit, Tp_fit)
    Xp_te = df_te[base].to_numpy(dtype=float)
    p_te = propensity_matrix(prop, Xp_te)

    print("[7/8] Predicciones en test + h(X)...")
    y_te = df_te[REAL_TARGET].to_numpy()
    pred_r = predict_factual(ridge, df_te, base, "ridge", scaler=scaler)
    pred_f = predict_factual(rf,    df_te, base, "rf")
    mu_r = predict_counterfactual(ridge, df_te, base, "ridge", scaler=scaler)
    mu_f = predict_counterfactual(rf,    df_te, base, "rf")
    h_r, h_f = decision_rule(mu_r), decision_rule(mu_f)

    pm_r = predictive_metrics(y_te, pred_r)
    pm_f = predictive_metrics(y_te, pred_f)
    rec_te = df_te["recession"].to_numpy() if "recession" in df_te.columns else None
    dm_r = real_decision_metrics(h_r, df_te["treatment"].to_numpy(), y_te,
                                 p_te, recession=rec_te)
    dm_f = real_decision_metrics(h_f, df_te["treatment"].to_numpy(), y_te,
                                 p_te, recession=rec_te)
    metrics_r = {**pm_r, **{k: v for k, v in dm_r.items()
                            if k != "h_distribution_by_regime"}}
    metrics_f = {**pm_f, **{k: v for k, v in dm_f.items()
                            if k != "h_distribution_by_regime"}}
    comparison = compare_real(metrics_r, metrics_f, dm_r["ipw_value_baseline_observed"])

    print("[8/8] Guardando outputs...")
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    _save_metrics(out, metrics_r, metrics_f)
    pd.DataFrame({"date": df_te[REAL_DATE_COL].values,
                  "T_obs": df_te["treatment"].values,
                  "h_ridge": h_r, "h_rf": h_f}).to_csv(out / "decisions.csv", index=False)
    _save_predictions(out, df_te[REAL_DATE_COL].values, mu_r, "ridge")
    _save_predictions(out, df_te[REAL_DATE_COL].values, mu_f, "rf")
    if dm_r["h_distribution_by_regime"] is not None:
        dm_r["h_distribution_by_regime"].to_csv(out / "h_by_regime_ridge.csv")
        dm_f["h_distribution_by_regime"].to_csv(out / "h_by_regime_rf.csv")
    pd.DataFrame(p_te, columns=[f"p_{t}" for t in T_LEVELS]).assign(
        date=df_te[REAL_DATE_COL].values,
        T_obs=df_te["treatment"].values,
    ).to_csv(out / "propensity_test.csv", index=False)
    with open(out / "hyperparameters.json", "w") as f:
        json.dump({"ridge": rt, "rf": ft}, f, indent=2, default=str)
    with open(out / "comparison.json", "w") as f:
        json.dump(comparison, f, indent=2, default=str)
    print(f"      outputs en: {out}")
    print("\n=== Real ===")
    print(pd.read_csv(out / "metrics.csv").to_string(index=False))
    print("Comparacion:", comparison)


# =============================================================================
# 10. CLI
# =============================================================================
def _parse_args(argv=None):
    p = argparse.ArgumentParser(description="Modulo de benchmarking unico.")
    p.add_argument("--mode",   required=True, choices=["synthetic", "real"])
    p.add_argument("--input",  required=True, help="Ruta al CSV de entrada.")
    p.add_argument("--output", required=True, help="Carpeta de salida.")
    return p.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv)
    if args.mode == "synthetic":
        run_synthetic(args.input, args.output)
    else:
        run_real(args.input, args.output)


if __name__ == "__main__":
    main()
