"""
Pruebas dirigidas del bloque 2 (sección C del protocolo) y cierre de la ejecución.

  tests    -> tablas/pruebas_b2.csv (prueba, esperado, observado, PASS/FAIL, evidencia); falla si alguna es FAIL
  finalize -> estado/run_status.json = COMPLETA + MANIFEST.sha256, solo si pruebas_b2.csv está todo en PASS
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from .common import ROOT, T_LEVELS, load_config, mae, rmse, sha256, tie_choice


def _row(rows, n, nombre, esperado, observado, ok, evidencia):
    rows.append({"prueba": n, "nombre": nombre, "esperado": esperado, "observado": observado,
                 "resultado": "PASS" if ok else "FAIL", "evidencia": evidencia})


# 1 ------------------------------------------------------------------------------
def t1_shocks(rows, run, cfg):
    from . import sim_exante as se
    sc = cfg["synthetic"]; p = se.make_params(sc)
    states = [np.array([2.0, 4.5, 2.5, 0.0]), np.array([5.0, 3.5, 1.0, 0.8]), np.array([0.5, 7.0, 4.0, -1.0])]
    rng = np.random.default_rng(123)
    worst, same_a = 0.0, True
    for x in states:
        for u in (0.05, 0.37, 0.71, 0.98):
            ref = None
            for _ in range(20):
                shocks = rng.normal(0, 5.0, 4)                  # choques futuros muy distintos
                q, probs, a, losses, _ = se.step(x, p, sc, u, shocks)
                if ref is None:
                    ref = (probs.copy(), a, losses.copy())
                else:
                    worst = max(worst, float(np.abs(probs - ref[0]).max()))
                    same_a &= a == ref[1]
    # en una trayectoria generada, p guardada = función de X observado (no de choques)
    F = pd.read_csv(run / "sintetico" / "seed42" / "factual.csv")
    V = pd.read_csv(run / "sintetico" / "seed42" / "verdad_reservada.csv")
    X = F[sc["base_features"]].to_numpy()
    rec = np.array([se.behavior_probs(se.expected_loss(x, p, sc), sc) for x in X])
    d = float(np.abs(rec - V[[f"p_behavior_{t}" for t in T_LEVELS]].to_numpy()).max())
    ok = worst == 0.0 and same_a and d < 1e-12
    _row(rows, 1, "Cambiar choques futuros a X fijo no cambia p(a|X) ni la acción",
         "dif. max p = 0; acción idéntica; p recalculada desde X == p guardada",
         f"dif. max p = {worst:.1e}; acción idéntica = {same_a}; recalculo desde X dif = {d:.1e}", ok,
         "src/b2/sim_exante.py:step; sintetico/seed42/")


# 2 ------------------------------------------------------------------------------
def t2_q_montecarlo(rows, run, cfg):
    from . import sim_exante as se
    sc = cfg["synthetic"]; p = se.make_params(sc); sig = se.sigma_vec(p, sc["noise_level"])
    rng = np.random.default_rng(2026); N = 200_000
    states = [np.array([2.0, 4.5, 2.5, 0.0]), np.array([4.5, 3.8, 1.5, 0.6]),
              np.array([0.8, 6.5, 3.8, -0.9]), np.array([3.0, 5.0, 2.0, 0.3])]
    zmax, rows_mc = 0.0, []
    for x in states:
        q = se.expected_loss(x, p, sc)
        S = rng.normal(0, sig, size=(N, 4))
        for a in range(3):
            m = np.array([se.v0._next_state(x, a, np.zeros(4), p)])[0]
            pi = m[0] + S[:, 0]; u = m[1] + S[:, 1]          # choques aditivos (mismas ecuaciones)
            L = (pi - sc["inflation_target"]) ** 2 + sc["lambda_u"] * (u - sc["unemployment_target"]) ** 2 \
                + sc["mu_r"] * p.dr[a] ** 2
            # verificación cruzada con _next_state completo en una submuestra
            sub = np.array([se.v0._loss(se.v0._next_state(x, a, S[i], p), p.dr[a], sc["inflation_target"],
                                        sc["unemployment_target"], sc["lambda_u"], sc["mu_r"]) for i in range(200)])
            assert np.allclose(sub, L[:200], atol=1e-12)
            se_ = L.std(ddof=1) / np.sqrt(N); z = abs(L.mean() - q[a]) / se_
            zmax = max(zmax, z); rows_mc.append({"x": x.tolist(), "a": T_LEVELS[a], "q": q[a],
                                                 "media_mc": L.mean(), "ee": se_, "z": z})
    pd.DataFrame(rows_mc).to_csv(run / "tablas" / "prueba2_q_montecarlo.csv", index=False)
    _row(rows, 2, "q analítico = pérdida esperada simulada (N=200000 por estado-acción)",
         "|media MC − q| <= 4 errores estándar en 12 casos (prob. de falso FAIL < 1e-3)",
         f"max |z| = {zmax:.2f}", zmax <= 4.0, "tablas/prueba2_q_montecarlo.csv")


# 3 ------------------------------------------------------------------------------
def t3_separation(rows, run, cfg):
    from . import real
    C = pd.read_csv(run / "tablas" / "cortes_y_tuning.csv")
    s = C[(C.rama == "sintetico") & C.ultima_etiqueta_train.notna()]
    ok_s = bool((s.ultima_etiqueta_train.astype(float) <= s.eval_desde.astype(float) - 1).all())
    r = C[(C.rama == "real") & C.ultima_etiqueta_train.notna()]
    det, ok_r = [], True
    for _, x in r.iterrows():
        L = cfg["real"]["variants"][x.id]
        okx = pd.Timestamp(x.ultima_etiqueta_train) <= pd.Timestamp(x.eval_desde) - pd.DateOffset(months=L)
        ok_r &= bool(okx)
        det.append(f"{x.id}/{x.fold}: etiquetas<= {x.ultima_etiqueta_train} vs eval {x.eval_desde}")
    # holgura mínima observada y verificación de que los predictores son los valores en t-L
    df = real.load_base(run / "derivados" / "base_macro_features.csv", cfg)
    ok_f = True
    for L in cfg["real"]["variants"].values():
        X, feats = real.build_features(df, L, cfg)
        idx = np.arange(30, len(df))
        ok_f &= bool(np.allclose(X[f"infl_m_t{-L}"].iloc[idx], df["infl_m"].iloc[idx - L]))
        ok_f &= bool(np.allclose(X[f"unrate_t{-(L + 11)}"].iloc[idx], df["unrate"].iloc[idx - L - 11]))
        sd = df["infl_m"].iloc[idx[0] - L - 11: idx[0] - L + 1].std(ddof=1)
        ok_f &= bool(np.isclose(X[f"infl_m_std12_t{-L}"].iloc[idx[0]], sd))
    P = pd.read_csv(run / "tablas" / "real_predicciones.csv", parse_dates=["date", "info_hasta", "etiquetas_train_hasta"])
    ok_p = True
    for var, L in cfg["real"]["variants"].items():
        Q = P[P.variante == var]
        ok_p &= bool((Q.info_hasta == Q.date - pd.DateOffset(months=L)).all())
        ok_p &= bool((Q.etiquetas_train_hasta <= Q.groupby("bloque").date.transform("min") - pd.DateOffset(months=L)).all())
    ok = ok_s and ok_r and ok_f and ok_p
    _row(rows, 3, "Separación por fechas en todos los folds y en test (sintético gap=1; real lag1 y lag2)",
         "sint: última etiqueta train <= primer evaluado − 1; real: label_end <= primer origen eval − L meses; X = valores en t−L",
         f"sintético {len(s)} cortes OK={ok_s}; real {len(r)} cortes OK={ok_r}; features t−L OK={ok_f}; predicciones OK={ok_p}",
         ok, "tablas/cortes_y_tuning.csv; tablas/real_predicciones.csv")
    # 3b: selección sintética (no estricta), recalculada desde los folds que usa fit_and_decide
    from . import synthetic
    from sklearn.model_selection import TimeSeriesSplit
    sc = cfg["synthetic"]
    dev, fit, te = synthetic.split_indices(sc["n_periods"], sc)
    folds = list(TimeSeriesSplit(n_splits=sc["cv_n_splits"], gap=sc["gap"]).split(dev))
    used = np.unique(np.concatenate([np.r_[a, b] for a, b in folds]))
    sel = int(synthetic.label_time(dev[used]).max())
    g = C[(C.rama == "sintetico") & (C.fold == "seleccion_global")]
    rep_ok = len(g) == len(cfg["synthetic"]["seeds"]) and bool((g.ultima_etiqueta_seleccion.astype(float) == sel).all()) \
        and bool((g.ultima_etiqueta_seleccion.astype(float) <= g.primer_estado_test.astype(float)).all())
    fit_strict = int(synthetic.label_time(fit).max()) < int(te.min())
    ok3b = sel <= int(te.min()) and rep_ok and fit_strict
    _row(rows, "3b", "Selección sintética: max(label_time usado en selección) <= primer estado de test; ajuste estricto",
         "selección <= 480 (fila 479 permitida); ajuste final < 480; registrado en cada semilla",
         f"selección max label_time = {sel}, primer estado test = {int(te.min())}; ajuste final max = "
         f"{int(synthetic.label_time(fit).max())}; filas seleccion_global coherentes = {rep_ok}", ok3b,
         "src/b2/synthetic.py:fit_and_decide; tablas/cortes_y_tuning.csv (fold = seleccion_global)")


# 4 ------------------------------------------------------------------------------
def t4_perturb(rows, run, cfg):
    from . import real, synthetic
    from .sim_exante import generate
    F, V = generate(42, cfg["synthetic"])
    r1 = synthetic.fit_and_decide(F, cfg)
    F2 = F.copy(); te = r1["idx"]["test"]
    F2.loc[te, "loss"] = F2.loc[te, "loss"] * 3 + 100.0          # etiquetas de test alteradas
    V2 = V.copy(); V2[[f"q_{t}" for t in T_LEVELS]] = V2[[f"q_{t}" for t in T_LEVELS]].to_numpy()[:, ::-1]
    r2 = synthetic.fit_and_decide(F2, cfg)
    same_hp = r1["chosen"] == r2["chosen"]
    same_mu = all(np.array_equal(r1["mu"][m], r2["mu"][m]) for m in synthetic.MODELS)
    same_dec = all(np.array_equal(r1["choice"][m], r2["choice"][m]) for m in synthetic.MODELS)
    e1, _ = synthetic.evaluate(r1, F, V, cfg, 42); e2, _ = synthetic.evaluate(r1, F2, V2, cfg, 42)
    eval_changed = e1[0]["regret_esperado_medio"] != e2[0]["regret_esperado_medio"]
    # real: alterar Y de test no cambia hiperparámetros ni predicciones
    df = real.load_base(run / "derivados" / "base_macro_features.csv", cfg)
    i1, _, P1 = real.run_variant(df, 1, cfg)
    df2 = df.copy(); mask = (df2.date >= cfg["real"]["test_start"]) & (df2.date <= cfg["real"]["test_end"])
    df2.loc[mask, "Y"] = df2.loc[mask, "Y"] + 50.0
    i2, _, P2 = real.run_variant(df2, 1, cfg)
    cols = ["pred_media", "pred_persistencia", "pred_ridge", "pred_rf"]
    same_real = i1["hiperparametros"] == i2["hiperparametros"] and np.array_equal(P1[cols].to_numpy(), P2[cols].to_numpy())
    real_eval_changed = not np.array_equal(P1.Y.to_numpy(), P2.Y.to_numpy())
    ok = same_hp and same_mu and same_dec and eval_changed and same_real and real_eval_changed
    _row(rows, 4, "Alterar etiquetas de test o la tabla de verdad no cambia ajuste, hiperparámetros ni decisiones",
         "hp, mu y decisiones idénticos (sint. seed 42 y real lag1); solo cambian métricas de evaluación",
         f"sint: hp={same_hp}, mu={same_mu}, decisiones={same_dec}, evaluación cambió={eval_changed}; "
         f"real: hp+predicciones idénticos={same_real}, Y evaluado cambió={real_eval_changed}", ok,
         "src/b2/synthetic.py:fit_and_decide/evaluate; src/b2/real.py:fit_and_predict")


# 5 ------------------------------------------------------------------------------
def t5_ties(rows, run, cfg):
    M = np.array([[1.0, 1.0 + 1e-9, 2.0],      # empate T0|T1 -> T0
                  [2.0, 1.0, 1.0 + 5e-7],      # empate T1|T2 -> T1 (tol = 1e-8 + 1e-6)
                  [3.0, 3.0, 3.0],             # plana -> T0, marcada
                  [2.0, 1.0, 1.0 + 1e-5],      # sin empate -> T1
                  [5.0, 5.0 + 4e-6, 5.0 + 6e-6]])  # tol relativa: min=5 -> 5.01e-6 : T0|T1
    c, tied = tie_choice(M, cfg)
    exp_c = [0, 1, 0, 1, 0]; exp_n = [2, 2, 3, 1, 2]
    ok_unit = [int(v) for v in c] == exp_c and [int(v) for v in tied.sum(1)] == exp_n
    c2, tied2 = tie_choice(M + 1e-13, cfg)
    ok_stable = np.array_equal(c, c2) and np.array_equal(tied, tied2)
    S = pd.read_csv(run / "tablas" / "sintetico_por_semilla.csv")
    P = pd.read_csv(run / "tablas" / "sintetico_predicciones.csv")
    ok_rep = True
    for m in ("ridge", "rf"):
        n_flat = P.groupby("semilla")[f"{m}_empate"].apply(lambda s: (s == "T0|T1|T2").sum())
        rep = S[S.politica == m].set_index("semilla")["n_planas"]
        ok_rep &= bool((n_flat.sort_index().to_numpy() == rep.sort_index().to_numpy()).all())
    tot_flat = int(S[S.politica.isin(["ridge", "rf"])].n_planas.sum())
    ok = ok_unit and ok_stable and ok_rep
    _row(rows, 5, "Empates cercanos: mismo desempate, conjunto registrado, predicción plana visible en el reporte",
         f"elecciones {exp_c}, tamaños {exp_n}; estable ante 1e-13; n_planas del reporte = conteo en predicciones",
         f"elecciones {[int(v) for v in c]}, tamaños {[int(v) for v in tied.sum(1)]}; estable={ok_stable}; reporte coincide={ok_rep}; "
         f"filas planas totales ridge+rf = {tot_flat}", ok, "src/b2/common.py:tie_choice; tablas/sintetico_por_semilla.csv")


# 6 ------------------------------------------------------------------------------
def t6_failure(rows, run, cfg):
    tmp = Path(tempfile.mkdtemp(prefix="b2_fallo_"))
    try:
        for d in ("src", "config"):
            shutil.copytree(ROOT / d, tmp / d, ignore=shutil.ignore_patterns("__pycache__"))
        (tmp / "data").mkdir(); shutil.copytree(ROOT / "data/raw", tmp / "data/raw")
        shutil.copy(ROOT / "run_bloque2.sh", tmp / "run_bloque2.sh")
        with open(tmp / "src/b2/synthetic.py", "a") as f:     # falla inyectada SOLO en la copia
            f.write("\nraise RuntimeError('falla inyectada por la prueba 6')\n")
        env = {**os.environ, "PY": sys.executable}
        r = subprocess.run(["bash", str(tmp / "run_bloque2.sh"), "prueba_fallo"], cwd=tmp, env=env,
                           capture_output=True, text=True, timeout=900)
        rd = tmp / "outputs/b2/prueba_fallo"
        status = json.loads((rd / "estado/run_status.json").read_text())
        done = (rd / "estado/etapas_ok.txt").read_text().split() if (rd / "estado/etapas_ok.txt").exists() else []
        later = [s for s in ("real", "cortes", "legacy_ipw", "determinism", "figures", "tests", "finalize")
                 if (rd / "logs" / f"{s}.log").exists()]
        no_manifest = not (rd / "MANIFEST.sha256").exists()
        ev = run / "logs" / "prueba6_fallo_controlado"; ev.mkdir(parents=True, exist_ok=True)
        (ev / "salida_runner.txt").write_text(f"returncode={r.returncode}\n--- stdout\n{r.stdout}\n--- stderr\n{r.stderr}")
        for f in (rd / "logs").glob("*.log"):
            shutil.copy(f, ev / f.name)
        shutil.copy(rd / "estado/run_status.json", ev / "run_status.json")
        orig_untouched = "falla inyectada" not in (ROOT / "src/b2/synthetic.py").read_text()
        ok6 = (r.returncode != 0 and status["estado"] == "FALLIDA" and status["etapa_fallida"] == "synthetic"
               and done == ["inputs", "derive"] and not later and no_manifest and orig_untouched)
        # 6b: runner histórico corregido (run_all.sh) también se detiene
        script = ("set -euo pipefail; LOG=$(mktemp -d); "
                  + _extract_step(ROOT / "run_all.sh")
                  + "; step falla sh -c 'exit 7'; echo NO_DEBERIA_LLEGAR")
        r2 = subprocess.run(["bash", "-c", script], capture_output=True, text=True)
        ok6b = r2.returncode == 7 and "NO_DEBERIA_LLEGAR" not in r2.stdout
        (ev / "run_all_step_fallo.txt").write_text(f"returncode={r2.returncode}\n{r2.stdout}\n{r2.stderr}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    _row(rows, 6, "Falla inyectada en una copia temporal (etapa synthetic)",
         "código != 0; estado FALLIDA en synthetic; solo inputs+derive OK; sin etapas posteriores ni MANIFEST; fuente intacta",
         f"código={r.returncode}; estado={status['estado']}/{status.get('etapa_fallida')}; etapas OK={done}; "
         f"logs posteriores={later}; sin manifest={no_manifest}; fuente intacta={orig_untouched}", ok6,
         "logs/prueba6_fallo_controlado/")
    # 6c: intérprete inexistente -> error, sin corrida EN_CURSO ni COMPLETA (copia temporal)
    tmp2 = Path(tempfile.mkdtemp(prefix="b2_py_"))
    try:
        shutil.copy(ROOT / "run_bloque2.sh", tmp2 / "run_bloque2.sh")
        for d in ("src", "config"):
            shutil.copytree(ROOT / d, tmp2 / d, ignore=shutil.ignore_patterns("__pycache__"))
        shutil.copytree(ROOT / "data/raw", tmp2 / "data/raw")
        r3 = subprocess.run(["bash", str(tmp2 / "run_bloque2.sh"), "prueba_py"], cwd=tmp2,
                            env={**os.environ, "PY": str(tmp2 / "no_existe" / "python")},
                            capture_output=True, text=True, timeout=120)
        st3 = tmp2 / "outputs/b2/prueba_py/estado/run_status.json"
        estado3 = json.loads(st3.read_text())["estado"] if st3.exists() else "sin_corrida_creada"
        (ev / "py_inexistente.txt").write_text(f"returncode={r3.returncode}\nestado={estado3}\n{r3.stdout}\n{r3.stderr}")
    finally:
        shutil.rmtree(tmp2, ignore_errors=True)
    ok6c = r3.returncode != 0 and estado3 not in ("EN_CURSO", "COMPLETA")
    _row(rows, "6c", "PY apunta a un intérprete inexistente (copia temporal)",
         "código != 0; ninguna corrida EN_CURSO ni COMPLETA",
         f"código={r3.returncode}; estado={estado3}", ok6c, "logs/prueba6_fallo_controlado/py_inexistente.txt")
    _row(rows, "6b", "run_all.sh (reproducción B1) corregido se detiene en la etapa que falla",
         "código 7 y no ejecuta lo siguiente", f"código={r2.returncode}; continuó={'NO_DEBERIA_LLEGAR' in r2.stdout}",
         ok6b, "logs/prueba6_fallo_controlado/run_all_step_fallo.txt; run_all_B1_original.sh (versión previa)")


def _extract_step(path):
    lines = path.read_text().splitlines()
    i = next(k for k, l in enumerate(lines) if l.startswith("step () {"))
    j = next(k for k in range(i, len(lines)) if lines[k] == "}")
    return "\n".join(lines[i:j + 1])


# 7 ------------------------------------------------------------------------------
def t7_integrity(rows, run, cfg):
    from .common import check_inputs
    h0 = json.loads((run / "estado" / "huellas_insumos_inicio.json").read_text())
    h1 = check_inputs(cfg)
    ok_h = h0 == h1
    # métricas sintéticas recalculadas desde las predicciones entregadas
    S = pd.read_csv(run / "tablas" / "sintetico_por_semilla.csv")
    P = pd.read_csv(run / "tablas" / "sintetico_predicciones.csv")
    qcols = [f"eval_q_{t}" for t in T_LEVELS]
    dmax = 0.0
    for seed, G in P.groupby("semilla"):
        q = G[qcols].to_numpy(); qmin = q.min(1)
        for m in ("ridge", "rf"):
            c = G[f"{m}_accion"].map({t: i for i, t in enumerate(T_LEVELS)}).to_numpy()
            reg = float(np.mean(q[np.arange(len(G)), c] - qmin))
            rm = rmse(G.loss_factual, G[f"{m}_pred_factual"])
            row = S[(S.semilla == seed) & (S.politica == m)].iloc[0]
            dmax = max(dmax, abs(reg - row.regret_esperado_medio), abs(rm - row.rmse_factual_test))
    # métricas reales recalculadas
    R = pd.read_csv(run / "tablas" / "real_predicciones.csv")
    M = pd.read_csv(run / "tablas" / "real_metricas.csv")
    for _, x in M[M.bloque == "test"].iterrows():
        B = R[(R.variante == x.variante) & (R.bloque == "test")]
        dmax = max(dmax, abs(rmse(B.Y, B[f"pred_{x.modelo}"]) - x.rmse), abs(mae(B.Y, B[f"pred_{x.modelo}"]) - x.mae))
    pc = ["pred_media", "pred_persistencia", "pred_ridge", "pred_rf"]
    same_rows = bool(R[pc + ["Y"]].notna().all().all()) and \
        bool((M.groupby(["variante", "bloque"]).n.nunique() == 1).all())
    syn_rows = bool(P[[f"{m}_mu_{t}" for m in ("ridge", "rf") for t in T_LEVELS]].notna().all().all()) and \
        bool((P.groupby("semilla").size() == S.n_test.iloc[0]).all())
    n_test_real = sorted(R[R.bloque == "test"].groupby("variante").size().unique().tolist())
    ok = ok_h and dmax < 1e-10 and same_rows and syn_rows
    _row(rows, 7, "Integridad: huellas de insumos, métricas recalculables, mismas filas entre modelos",
         "huellas inicio == fin; |métrica recalculada − reportada| < 1e-10; sin faltantes y mismo n por bloque",
         f"huellas iguales={ok_h}; dif. max={dmax:.1e}; filas reales completas={same_rows} (n test={n_test_real}); "
         f"filas sintéticas completas={syn_rows}", ok, "estado/huellas_insumos_inicio.json; tablas/*")


def run_tests(run):
    cfg = load_config(run / "config_b2_usada.json")
    rows = []
    for t in (t1_shocks, t2_q_montecarlo, t3_separation, t4_perturb, t5_ties, t6_failure, t7_integrity):
        try:
            t(rows, run, cfg)
        except Exception as e:  # una excepción es un FAIL registrado, no un PASS silencioso
            _row(rows, t.__name__, t.__name__, "sin excepción", f"{type(e).__name__}: {e}", False, "logs/tests.log")
    D = pd.DataFrame(rows)
    D.to_csv(run / "tablas" / "pruebas_b2.csv", index=False)
    pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 120)
    print(D[["prueba", "resultado", "observado"]].to_string(index=False))
    if (D.resultado != "PASS").any():
        raise SystemExit(1)


def finalize(run):
    D = pd.read_csv(run / "tablas" / "pruebas_b2.csv")
    if (D.resultado != "PASS").any():
        raise SystemExit("Pruebas con FAIL: no se marca COMPLETA")
    det = pd.read_csv(run / "tablas" / "determinismo_b2.csv")
    if not det["pass"].all():
        raise SystemExit("Determinismo con FAIL")
    run_id = run.name
    status = {"run_id": run_id, "estado": "COMPLETA", "pruebas": int(len(D)), "pruebas_pass": int((D.resultado == "PASS").sum())}
    (run / "estado" / "run_status.json").write_text(json.dumps(status, indent=2))
    # se excluyen los tres archivos que el runner sigue escribiendo después de esta etapa
    volatile = {"MANIFEST.sha256", "estado/etapas_ok.txt", "logs/etapas.log", "logs/finalize.log"}
    files = sorted(p for p in run.rglob("*") if p.is_file() and p.relative_to(run).as_posix() not in volatile)
    lines = [f"# run_id {run_id}"] + [f"{sha256(p)}  {p.relative_to(run).as_posix()}" for p in files]
    (run / "MANIFEST.sha256").write_text("\n".join(lines) + "\n")
    print(json.dumps(status))


def main(stage, run):
    run_tests(run) if stage == "tests" else finalize(run)
