"""
Rama A: laboratorio de decisiones ex ante a un paso (S0_B2_exante).

Separación de información:
  - fit_and_decide() recibe SOLO la tabla factual (X, T observado, pérdida de la acción
    asignada). Las etiquetas del test no se leen ahí: las decisiones dependen de X_test.
  - evaluate() es el único lugar donde entra la tabla de verdad (q y pérdidas potenciales).

Cronología: la etiqueta `loss` de la fila t se realiza en t+1 (usa X_{t+1}). Tres momentos
distintos (convención aceptada en la revisión del bloque 3):
  - selección de hiperparámetros: usa las filas de desarrollo 0..479; la etiqueta de la fila
    479 se conoce al observar X480 (primer estado de test). Control: max(label_time usado en
    selección) <= primer estado de test (desigualdad no estricta).
  - ajuste final: filas 0..478 (purga de 1); la última etiqueta se conoce en 479 < 480.
    Dentro de cada fold, gap=1 deja la última etiqueta de ajuste en primer_evaluado-1 (estricto).
  - decisión en X480: se toma antes de generar el choque que determina su resultado.

Diseño de los modelos (igual que el benchmark original, benchmark_module.build_design_matrix):
  rf:    [X, T1, T2]
  ridge: [X, T1, T2, X*T1, X*T2]  con StandardScaler aprendido en cada ajuste
"""
from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

from .common import ROOT, T_LEVELS, candidates, make_model, rmse, select, tie_choice, tied_label

sys.path.insert(0, str(ROOT / "src"))
import benchmark_module as bm  # noqa: E402  (reutiliza la matriz de diseño original)

MODELS = ("ridge", "rf")


def _design(df, feats, model, t=None):
    return bm.build_design_matrix(df, feats, model, treatment_value=t)


def _with_dummies(df):
    df = df.copy()
    df["T1"] = (df["treatment"] == "T1").astype(int)
    df["T2"] = (df["treatment"] == "T2").astype(int)
    return df


def split_indices(n, sc):
    n_dev = int(np.floor(n * sc["dev_frac"]))
    dev = np.arange(n_dev)
    test = np.arange(n_dev, n)
    fit_final = np.arange(n_dev - sc["gap"])          # retira la(s) observación(es) previa(s) al test
    return dev, fit_final, test


def label_time(idx):
    """Momento en que se conoce la etiqueta de la fila idx (pérdida de t usa X_{t+1})."""
    return np.asarray(idx) + 1


def fit_and_decide(F, cfg):
    sc = cfg["synthetic"]; feats = sc["base_features"]
    F = _with_dummies(F)
    n = len(F)
    dev, fit_final, test = split_indices(n, sc)
    D = F.iloc[dev].reset_index(drop=True)
    folds = list(TimeSeriesSplit(n_splits=sc["cv_n_splits"], gap=sc["gap"]).split(np.arange(len(D))))
    cuts, chosen, fold_scores = [], {}, {}
    for k, (tr, va) in enumerate(folds, 1):
        if label_time(tr).max() > va.min() - 1:
            raise AssertionError(f"fold {k}: etiqueta de ajuste alcanza el primer momento evaluado")
    for model in MODELS:
        scores = []
        for params in candidates(cfg, model):
            yhat, ytrue = [], []
            for k, (tr, va) in enumerate(folds, 1):
                m = make_model(cfg, model, params).fit(_design(D.iloc[tr], feats, model), D["loss"].iloc[tr])
                pr = m.predict(_design(D.iloc[va], feats, model))
                yhat.append(pr); ytrue.append(D["loss"].iloc[va].to_numpy())
                cuts.append({"rama": "sintetico", "id": None, "fold": k, "modelo": model,
                             "hiperparametros": json.dumps(params),
                             "train_desde": int(tr.min()), "train_hasta": int(tr.max()),
                             "eval_desde": int(va.min()), "eval_hasta": int(va.max()),
                             "ultima_etiqueta_train": int(label_time(tr).max()),
                             "n_train": len(tr), "n_eval": len(va), "rmse_fold": rmse(ytrue[-1], pr)})
            s = rmse(np.concatenate(ytrue), np.concatenate(yhat))
            scores.append((params, s))
            cuts.append({"rama": "sintetico", "id": None, "fold": "agregado", "modelo": model,
                         "hiperparametros": json.dumps(params), "rmse_seleccion": s,
                         "n_eval": int(sum(len(v) for _, v in folds))})
        chosen[model], _ = select(cfg, scores)
        fold_scores[model] = scores
    # control de disponibilidad de la selección, derivado de los índices efectivamente usados
    used = np.unique(np.concatenate([np.r_[tr, va] for tr, va in folds]))
    sel_max = int(label_time(dev[used]).max())
    if sel_max > test.min():
        raise AssertionError(f"selección: etiqueta conocida en {sel_max} > primer estado de test {test.min()}")
    cuts.append({"rama": "sintetico", "id": None, "fold": "seleccion_global", "modelo": "ridge|rf",
                 "train_desde": int(dev[used].min()), "train_hasta": int(dev[used].max()),
                 "eval_desde": int(test.min()), "eval_hasta": int(test.max()),
                 "ultima_etiqueta_seleccion": sel_max, "primer_estado_test": int(test.min()),
                 "n_train": int(len(used))})
    # ajuste final y decisiones en test (solo X_test; T y etiquetas del test no intervienen)
    if label_time(fit_final).max() > test.min() - 1:
        raise AssertionError("ajuste final: etiqueta alcanza el primer momento de test")
    Ffit, Fte = F.iloc[fit_final].reset_index(drop=True), F.iloc[test].reset_index(drop=True)
    out = {"chosen": chosen, "cuts": cuts, "idx": {"dev": dev, "fit_final": fit_final, "test": test},
           "mu": {}, "choice": {}, "tied": {}, "pred_factual": {}}
    for model in MODELS:
        m = make_model(cfg, model, chosen[model]).fit(_design(Ffit, feats, model), Ffit["loss"])
        mu = np.column_stack([m.predict(_design(Fte, feats, model, t)) for t in T_LEVELS])
        c, tied = tie_choice(mu, cfg)
        out["mu"][model], out["choice"][model], out["tied"][model] = mu, c, tied
        out["pred_factual"][model] = m.predict(_design(Fte, feats, model))
        cuts.append({"rama": "sintetico", "id": None, "fold": "final_test", "modelo": model,
                     "hiperparametros": json.dumps(chosen[model]),
                     "train_desde": int(fit_final.min()), "train_hasta": int(fit_final.max()),
                     "eval_desde": int(test.min()), "eval_hasta": int(test.max()),
                     "ultima_etiqueta_train": int(label_time(fit_final).max()),
                     "n_train": len(fit_final), "n_eval": len(test)})
    return out


def evaluate(res, F, V, cfg, seed):
    """Única función que usa la tabla de verdad V."""
    sc = cfg["synthetic"]
    te, fit = res["idx"]["test"], res["idx"]["fit_final"]
    Vt = V.iloc[te].reset_index(drop=True); Ft = F.iloc[te].reset_index(drop=True)
    q = Vt[[f"q_{t}" for t in T_LEVELS]].to_numpy()
    _, opt_set = tie_choice(q, cfg)                    # conjunto óptimo ex ante (con la misma tolerancia)
    qmin = q.min(axis=1)
    rows_n = np.arange(len(te))
    freq = lambda s: {f"freq_{t}": float(np.mean(np.asarray(s) == t)) for t in T_LEVELS}
    obs_train = F["treatment"].iloc[fit].value_counts()
    obs_test = Ft["treatment"].value_counts()
    common = {"escenario": sc["scenario"], "semilla": seed, "n_train": len(fit), "n_test": len(te),
              **{f"train_obs_{t}": int(obs_train.get(t, 0)) for t in T_LEVELS},
              **{f"test_obs_{t}": int(obs_test.get(t, 0)) for t in T_LEVELS}}
    policies = {}
    for model in MODELS:
        policies[model] = res["choice"][model]
    for k, t in enumerate(T_LEVELS):
        policies[f"siempre_{t}"] = np.full(len(te), k)
    policies["oraculo_exante"] = np.argmax(opt_set, axis=1)
    rows = []
    for name, c in policies.items():
        qc = q[rows_n, c]
        r = {**common, "politica": name,
             "hiperparametros": json.dumps(res["chosen"][name]) if name in MODELS else "",
             "rmse_factual_test": rmse(Ft["loss"], res["pred_factual"][name]) if name in MODELS else np.nan,
             "regret_esperado_medio": float(np.mean(qc - qmin)),
             "regret_esperado_max": float(np.max(qc - qmin)),
             "perdida_esperada_media": float(np.mean(qc)),
             "coincidencia_optima": float(np.mean(opt_set[rows_n, c])),
             **freq(np.array(T_LEVELS)[c])}
        if name in MODELS:
            tied = res["tied"][name]
            r["n_empates"] = int((tied.sum(1) > 1).sum())
            r["n_planas"] = int((tied.sum(1) == 3).sum())
            # diagnóstico: el desempate hacia T0 no es una recomendación informativa
            nt = tied.sum(1) == 1
            r["n_sin_empate"] = int(nt.sum())
            r["regret_esperado_medio_filas_sin_empate"] = float(np.mean((qc - qmin)[nt])) if nt.any() else np.nan
            r["regret_siempre_T0_mismas_filas_sin_empate"] = float(np.mean((q[:, 0] - qmin)[nt])) if nt.any() else np.nan
        else:
            r["n_empates"] = np.nan; r["n_planas"] = np.nan
        rows.append(r)
    # tabla de predicciones (columnas eval_* nunca entran al ajuste)
    P = pd.DataFrame({"escenario": sc["scenario"], "semilla": seed, "t": Ft["t"], "date": Ft["date"],
                      **{c: Ft[c] for c in sc["base_features"]}, "T_obs": Ft["treatment"],
                      "loss_factual": Ft["loss"]})
    for model in MODELS:
        for k, t in enumerate(T_LEVELS):
            P[f"{model}_mu_{t}"] = res["mu"][model][:, k]
        P[f"{model}_accion"] = np.array(T_LEVELS)[res["choice"][model]]
        P[f"{model}_empate"] = [tied_label(m) for m in res["tied"][model]]
        P[f"{model}_pred_factual"] = res["pred_factual"][model]
    for k, t in enumerate(T_LEVELS):
        P[f"eval_q_{t}"] = q[:, k]
    for k, t in enumerate(T_LEVELS):
        P[f"eval_loss_pot_{t}"] = Vt[f"loss_pot_{t}"]
    for k, t in enumerate(T_LEVELS):
        P[f"eval_p_behavior_{t}"] = Vt[f"p_behavior_{t}"]
    P["eval_conjunto_optimo"] = [tied_label(m) for m in opt_set]
    return rows, P


def run_seed(seed, cfg):
    from .sim_exante import generate
    F, V = generate(seed, cfg["synthetic"])
    res = fit_and_decide(F, cfg)
    for c in res["cuts"]:
        c["id"] = f"seed{seed}"
    rows, P = evaluate(res, F, V, cfg, seed)
    return F, V, res, rows, P
