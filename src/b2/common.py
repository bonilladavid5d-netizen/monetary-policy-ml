"""Utilidades comunes del bloque 2: configuración, cuadrícula, selección, desempates y huellas."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
T_LEVELS = ("T0", "T1", "T2")


def load_config(path=None):
    return json.loads(Path(path or ROOT / "config" / "b2.json").read_text())


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def check_inputs(cfg):
    """Falla si falta un insumo o su huella no coincide con la configuración congelada."""
    out = {}
    d = ROOT / cfg["inputs"]["fred_dir"]
    for fn, expected in cfg["inputs"]["files"].items():
        p = d / fn
        if not p.exists():
            raise FileNotFoundError(f"Insumo faltante: {p}")
        got = sha256(p)
        if got != expected:
            raise ValueError(f"Huella distinta en {p}: {got} != {expected}")
        out[fn] = got
    sim = ROOT / cfg["inputs"]["simulator_base"]
    got = sha256(sim)
    exp = cfg["inputs"].get("simulator_base_sha256")
    if exp and got != exp:
        raise ValueError(f"Huella distinta en {sim}: {got} != {exp}")
    out[cfg["inputs"]["simulator_base"]] = got
    return out


# ----------------------------------------------------------------- cuadrícula
def candidates(cfg, model):
    """Candidatos en ORDEN DE PREFERENCIA ante empate numérico (el primero gana)."""
    g = cfg["grid"]
    if model == "ridge":
        return [{"alpha": a} for a in sorted(g["ridge_alpha"], reverse=True)]  # mayor alpha primero
    pairs = [(d, l) for d, l in g["rf_pairs"]]
    pairs.sort(key=lambda p: (np.inf if p[0] is None else p[0], -p[1]))     # menor profundidad, mayor hoja
    return [{"max_depth": d, "min_samples_leaf": l} for d, l in pairs]


def make_model(cfg, model, params):
    g = cfg["grid"]
    if model == "ridge":
        # el escalado se aprende dentro de cada ajuste (fold o final), nunca con datos de evaluación
        return make_pipeline(StandardScaler(), Ridge(alpha=params["alpha"]))
    return RandomForestRegressor(n_estimators=g["rf_n_estimators"], random_state=g["rf_random_state"],
                                 n_jobs=g["rf_n_jobs"], **params)


def select(cfg, scores):
    """scores: lista (params, rmse) en orden de preferencia. Devuelve el primero dentro de la tolerancia del mínimo."""
    best = min(s for _, s in scores)
    tol = cfg["grid"]["tie_rel_tol"] * max(1.0, abs(best))
    for p, s in scores:
        if s <= best + tol:
            return p, s


def rmse(y, yhat):
    y, yhat = np.asarray(y, float), np.asarray(yhat, float)
    return float(np.sqrt(np.mean((y - yhat) ** 2)))


def mae(y, yhat):
    return float(np.mean(np.abs(np.asarray(y, float) - np.asarray(yhat, float))))


# ----------------------------------------------------------------- desempates
def tie_choice(values, cfg):
    """values: (n, 3) pérdidas predichas o esperadas por acción.
    Conjunto empatado = {a : v_a <= min + 1e-8 + 1e-6*max(1,|min|)}.
    Elección: T0 si pertenece; si no, la primera por orden T0/T1/T2.
    Devuelve (índice elegido, máscara del conjunto empatado)."""
    v = np.asarray(values, float)
    m = v.min(axis=1, keepdims=True)
    tol = cfg["decision_ties"]["abs"] + cfg["decision_ties"]["rel"] * np.maximum(1.0, np.abs(m))
    tied = v <= m + tol
    choice = np.argmax(tied, axis=1)  # primera True en orden T0, T1, T2 (T0 si pertenece)
    return choice, tied


def tied_label(mask_row):
    return "|".join(t for t, b in zip(T_LEVELS, mask_row) if b)
