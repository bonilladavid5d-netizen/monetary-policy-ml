"""
Etapas del bloque 2. Cada etapa escribe en <run_dir> y falla con excepción (código != 0).
Uso: python -I -m b2.stages <etapa> <run_dir>      (con src/ en PYTHONPATH; ver run_bloque2.sh)

Etapas: inputs, derive, synthetic, real, legacy_ipw, determinism, figures   (tests/finalize: tests_b2.py)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from .common import ROOT, T_LEVELS, check_inputs, load_config, make_model, rmse, sha256, tie_choice


def st_inputs(run, cfg):
    h = check_inputs(cfg)
    (run / "estado").mkdir(exist_ok=True)
    (run / "estado" / "huellas_insumos_inicio.json").write_text(json.dumps(h, indent=2))
    print(json.dumps(h, indent=2))


def st_derive(run, cfg):
    from . import real
    p = real.build_derived(run / "derivados", cfg)
    # comparación informativa con la copia del bloque 1 (no se usa como insumo)
    ref = ROOT / "data/derived/base_macro_features.csv"
    info = {"base_macro_features_sha256": sha256(p),
            "igual_a_data_derived_B1": (sha256(p) == sha256(ref)) if ref.exists() else "referencia B1 no disponible"}
    (run / "derivados" / "procedencia.json").write_text(json.dumps(info, indent=2))
    print(info)


def st_synthetic(run, cfg):
    from .synthetic import run_seed
    T = run / "tablas"; T.mkdir(exist_ok=True)
    rows, preds, cuts, facts = [], [], [], []
    for seed in cfg["synthetic"]["seeds"]:
        F, V, res, r, P = run_seed(seed, cfg)
        rows += r; preds.append(P); cuts += res["cuts"]
        d = run / "sintetico" / f"seed{seed}"; d.mkdir(parents=True, exist_ok=True)
        F.to_csv(d / "factual.csv", index=False)          # lo único que ven los modelos
        V.to_csv(d / "verdad_reservada.csv", index=False)  # q, pérdidas potenciales, p de comportamiento
        print(f"seed {seed}: " + ", ".join(f"{x['politica']}={x['regret_esperado_medio']:.4f}" for x in r))
    S = pd.DataFrame(rows); S.to_csv(T / "sintetico_por_semilla.csv", index=False)
    pd.concat(preds, ignore_index=True).to_csv(T / "sintetico_predicciones.csv", index=False)
    pd.DataFrame(cuts).to_csv(T / "cortes_sintetico.csv", index=False)
    # resumen 10 semillas + diferencias emparejadas Ridge-RF
    g = S.groupby("politica")
    summ = pd.concat({m: g[m].agg(["mean", "std", "min", "max"]) for m in
                      ["regret_esperado_medio", "perdida_esperada_media", "coincidencia_optima", "rmse_factual_test"]}, axis=1)
    summ.columns = [f"{a}_{b}" for a, b in summ.columns]
    summ.reset_index().to_csv(T / "sintetico_resumen_semillas.csv", index=False)
    piv = S.pivot(index="semilla", columns="politica", values="regret_esperado_medio")
    pair = pd.DataFrame({"semilla": piv.index, "regret_ridge": piv["ridge"], "regret_rf": piv["rf"],
                         "dif_ridge_menos_rf": piv["ridge"] - piv["rf"],
                         "mejor_referencia_fija": piv[[f"siempre_{t}" for t in T_LEVELS]].idxmin(axis=1),
                         "regret_mejor_referencia_fija": piv[[f"siempre_{t}" for t in T_LEVELS]].min(axis=1)})
    pair.to_csv(T / "sintetico_diferencias_emparejadas.csv", index=False)
    print(summ.round(4).to_string())


def st_real(run, cfg):
    from . import real
    T = run / "tablas"; T.mkdir(exist_ok=True)
    df = real.load_base(run / "derivados" / "base_macro_features.csv", cfg)
    infos, cuts, preds = {}, [], []
    for tag, L in cfg["real"]["variants"].items():
        info, c, P = real.run_variant(df, L, cfg)
        infos[tag] = info; cuts += c; preds.append(P)
    P = pd.concat(preds, ignore_index=True)
    P.to_csv(T / "real_predicciones.csv", index=False)
    M = real.metrics(P); M.to_csv(T / "real_metricas.csv", index=False)
    pd.DataFrame(cuts).to_csv(T / "cortes_real.csv", index=False)
    (run / "estado" / "real_info.json").write_text(json.dumps(infos, indent=2, default=str))
    print(M[M.bloque.isin(["test", "folds_desarrollo_agregado"])].round(4).to_string(index=False))


def st_cortes(run):
    T = run / "tablas"
    C = pd.concat([pd.read_csv(T / "cortes_sintetico.csv"), pd.read_csv(T / "cortes_real.csv")], ignore_index=True)
    C.to_csv(T / "cortes_y_tuning.csv", index=False)


def st_legacy_ipw(run, cfg):
    """Diagnóstico del IPW legado (benchmark_module original). No rankea políticas."""
    sys.path.insert(0, str(ROOT / "src"))
    import benchmark_module as bm
    src = ROOT / cfg["inputs"]["legacy_outputs_real"]
    df = bm.load_real(run / "derivados" / "base_macro_features.csv")
    _, _, te = bm.temporal_split(len(df), buffer=bm.REAL_BUFFER)
    te_df = df.iloc[te].reset_index(drop=True)
    dec = pd.read_csv(src / "decisions.csv", parse_dates=["date"])
    prop = pd.read_csv(src / "propensity_test.csv")
    met = pd.read_csv(src / "metrics.csv").set_index("model")
    mu_rf = pd.read_csv(src / "predictions_rf.csv")[[f"mu_{t}" for t in T_LEVELS]].to_numpy()
    if not (te_df.date.values == dec.date.values).all():
        raise AssertionError("fechas de test legado no coinciden con la base regenerada")
    L = te_df[bm.REAL_TARGET].to_numpy(); Tobs = dec.T_obs.to_numpy()
    Pm = prop[[f"p_{t}" for t in T_LEVELS]].to_numpy()
    e = Pm[np.arange(len(Tobs)), [T_LEVELS.index(t) for t in Tobs]]; w = 1 / e
    pols = {"h_ridge": dec.h_ridge.to_numpy(), "h_rf": dec.h_rf.to_numpy(),
            **{f"siempre_{t}": np.full(len(Tobs), t, dtype=object) for t in T_LEVELS}}
    key = {"h_ridge": ("ridge", "ipw_value"), "h_rf": ("rf", "ipw_value"),
           **{f"siempre_{t}": ("ridge", f"ipw_value_baseline_{t}") for t in T_LEVELS}}
    rows = []
    for name, pol in pols.items():
        m = Tobs == pol
        contrib = m * L * w / len(Tobs)
        num = float(contrib.sum())
        top = np.sort(contrib)[::-1]
        rows.append({
            "politica_legada": name,
            "ipw_legado_guardado": float(met.loc[key[name]]),
            "ipw_legado_recalculado": float(bm._ipw_value(L, Tobs, pol, Pm)),
            "valor_interpretable": np.nan,  # NA: no validado como valor causal en ningún caso
            "n_coincidencias": int(m.sum()), "n_test": len(Tobs),
            "test_obs_T0": int((Tobs == "T0").sum()), "test_obs_T1": int((Tobs == "T1").sum()),
            "test_obs_T2": int((Tobs == "T2").sum()),
            "ess_kish": float((m * w).sum() ** 2 / (m * w ** 2).sum()) if m.any() else np.nan,
            "peso_max": float((m * w).max()) if m.any() else np.nan,
            "n_ponderaciones_recortadas_1e-3": int((m & np.isclose(e, 1e-3)).sum()),
            "share_top1_numerador": float(top[0] / num) if num > 0 else np.nan,
            "share_top2_numerador": float(top[:2].sum() / num) if num > 0 else np.nan,
            "estado": ("NO_EVALUABLE_SIN_COINCIDENCIAS (el 0 es solo el resultado del cálculo legado)"
                       if not m.any() else "NO_VALIDADO_COMO_VALOR_CAUSAL"),
        })
    D = pd.DataFrame(rows)
    D["nota"] = ("T es función de dfedfunds incluido en X; soporte insuficiente; "
                 f"RF legado: rango max de mu entre acciones en test = {float((mu_rf.max(1) - mu_rf.min(1)).max()):.2e}")
    D.to_csv(run / "tablas" / "diagnostico_ipw_legacy.csv", index=False)
    print(D.drop(columns=["nota"]).to_string(index=False))


def st_determinism(run, cfg):
    """Repite el ajuste elegido de la semilla sintética y del modelo real principal y compara."""
    from . import real, synthetic
    from .sim_exante import generate
    dc = cfg["determinism"]; tol = dc["atol_pred"]; T = run / "tablas"
    out = []
    seed = dc["synthetic_seed"]
    F, V = generate(seed, cfg["synthetic"])
    Fs = pd.read_csv(run / "sintetico" / f"seed{seed}" / "factual.csv")
    cols = cfg["synthetic"]["base_features"] + ["loss"]
    F_b, V_b = generate(seed, cfg["synthetic"])
    # dos generaciones en memoria: idénticas bit a bit; frente al CSV guardado: tolerancia de redondeo CSV
    gen_equal = bool(F[cols].equals(F_b[cols]) and V.equals(V_b) and (F.treatment == Fs.treatment).all()
                     and np.allclose(F[cols].to_numpy(), Fs[cols].to_numpy(), rtol=0, atol=tol))
    S = pd.read_csv(T / "sintetico_por_semilla.csv"); S = S[S.semilla == seed].set_index("politica")
    P = pd.read_csv(T / "sintetico_predicciones.csv"); P = P[P.semilla == seed]
    Fd = synthetic._with_dummies(F)
    _, fit, te = synthetic.split_indices(len(F), cfg["synthetic"])
    for model in synthetic.MODELS:
        params = json.loads(S.loc[model, "hiperparametros"])
        for rep in (1, 2):
            m = make_model(cfg, model, params).fit(synthetic._design(Fd.iloc[fit], cfg["synthetic"]["base_features"], model),
                                                   Fd["loss"].iloc[fit])
            mu = np.column_stack([m.predict(synthetic._design(Fd.iloc[te], cfg["synthetic"]["base_features"], model, t))
                                  for t in T_LEVELS])
            c, _ = tie_choice(mu, cfg)
            ref = P[[f"{model}_mu_{t}" for t in T_LEVELS]].to_numpy()
            out.append({"rama": "sintetico", "id": f"seed{seed}", "modelo": model, "repeticion": rep,
                        "generador_identico": gen_equal,
                        "max_dif_pred": float(np.abs(mu - ref).max()),
                        "decisiones_identicas": bool((np.array(T_LEVELS)[c] == P[f"{model}_accion"].to_numpy()).all()),
                        "tolerancia": tol})
    df = real.load_base(run / "derivados" / "base_macro_features.csv", cfg)
    L = cfg["real"]["variants"][dc["real_variant"]]
    dev, test, feats, _ = real.prepare(df, L, cfg)
    info = json.loads((run / "estado" / "real_info.json").read_text())[dc["real_variant"]]
    R = pd.read_csv(T / "real_predicciones.csv")
    R = R[(R.variante == dc["real_variant"]) & (R.bloque == "test")]
    for model in real.MODELS:
        for rep in (1, 2):
            pr = real._fit_predict(model, info["hiperparametros"][model], cfg, dev, test, feats)
            out.append({"rama": "real", "id": dc["real_variant"], "modelo": model, "repeticion": rep,
                        "generador_identico": np.nan,
                        "max_dif_pred": float(np.abs(pr - R[f"pred_{model}"].to_numpy()).max()),
                        "decisiones_identicas": np.nan,
                        "rmse_repetido": rmse(test.Y, pr), "tolerancia": tol})
    D = pd.DataFrame(out)
    D["pass"] = (D.max_dif_pred <= tol) & D.decisiones_identicas.fillna(True).astype(bool) & \
        D.generador_identico.fillna(True).astype(bool)
    D.to_csv(T / "determinismo_b2.csv", index=False)
    print(D.to_string(index=False))
    if not D["pass"].all():
        raise AssertionError("Determinismo: diferencias fuera de tolerancia")


def st_figures(run, cfg):
    from .figures_b2 import make_all
    make_all(run, cfg)


STAGES = {"inputs": st_inputs, "derive": st_derive, "synthetic": st_synthetic, "real": st_real,
          "cortes": lambda run, cfg: st_cortes(run), "legacy_ipw": st_legacy_ipw,
          "determinism": st_determinism, "figures": st_figures}

if __name__ == "__main__":
    stage, run = sys.argv[1], Path(sys.argv[2])
    cfg = load_config(run / "config_b2_usada.json")
    STAGES[stage](run, cfg)
