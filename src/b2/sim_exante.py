"""
Escenario S0_B2_exante: simulador corregido del bloque 2.

Reutiliza SIN CAMBIOS las ecuaciones y parámetros estructurales de
src/simular_datos_v0_dr2.py (DGPParams, _initial_state, _next_state, _loss),
el simulador que generó la base sintética S0 original. Lo que cambia es el orden
de la información y el oráculo:

  v0 original                                   S0_B2_exante
  1) choques del periodo                        1) observar X_t y calcular q(X_t, a)
  2) pérdidas realizadas L^a con esos choques   2) asignar T_t ~ p(a|X_t) = 0.9*softmax(-1.2 q) + 0.1/3
  3) T_t por softmax sobre L^a + ruido          3) generar choques de t (flujo independiente)
     -> la asignación usa información futura    4) pérdidas potenciales con choques comunes
  oráculo = argmin de L^a realizadas (ex post)  oráculo = argmin_a q(X_t, a)  (ex ante)

q(X,a) = (m_pi - pi*)^2 + (s_pi*n)^2 + lambda*[(m_u - u*)^2 + (s_u*n)^2] + mu*(dr_a)^2,
con (m_pi, m_u) = transición con choque cero. Es la esperanza exacta de la pérdida v0
(L = (pi'-pi*)^2 + lambda (u'-u*)^2 + mu dr^2) porque los choques de pi y u son aditivos,
de media cero y varianza (sigma*noise)^2.

La política de comportamiento usa q por construcción (es el "banco central" del
laboratorio). Los modelos solo ven la tabla factual; q y las pérdidas potenciales
van a una tabla de verdad separada.
"""
from __future__ import annotations

import importlib.util
import sys

import numpy as np
import pandas as pd

from .common import ROOT, T_LEVELS


def _load_v0():
    name = "simular_datos_v0_dr2"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, ROOT / "src" / f"{name}.py")
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


v0 = _load_v0()


def make_params(sc):
    p = v0.DGPParams()
    p.pi_star = sc["inflation_target"]
    p.u_star = sc["unemployment_target"]
    return p


def sigma_vec(p, noise):
    return np.array([p.sigma_pi, p.sigma_u, p.sigma_y, p.sigma_r]) * noise


def expected_loss(x, p, sc):
    """q(X, a) para a = T0, T1, T2 (vector de 3)."""
    s = sigma_vec(p, sc["noise_level"])
    q = np.empty(3)
    for a in range(3):
        m = v0._next_state(x, a, np.zeros(4), p)        # media condicional (choque cero)
        q[a] = ((m[0] - sc["inflation_target"]) ** 2 + s[0] ** 2
                + sc["lambda_u"] * ((m[1] - sc["unemployment_target"]) ** 2 + s[1] ** 2)
                + sc["mu_r"] * p.dr[a] ** 2)
    return q


def behavior_probs(q, sc):
    """p(a|X) = mix*softmax(-tau*q) + floor/3; softmax estable restando el mínimo de q."""
    w = np.exp(-sc["behavior_tau"] * (q - q.min()))
    return sc["behavior_mix"] * w / w.sum() + sc["behavior_floor_total"] / 3.0


def draw_action(probs, u):
    """Acción por CDF inversa con un uniforme u del flujo de acciones (misma aleatoriedad -> misma acción)."""
    return int(min(np.searchsorted(np.cumsum(probs), u, side="right"), 2))


def step(x, p, sc, u_action, shocks):
    """Un periodo en el orden obligatorio. Las probabilidades y la acción se calculan
    ANTES de usar `shocks`; cambiar `shocks` no puede alterarlas (prueba 1)."""
    q = expected_loss(x, p, sc)
    probs = behavior_probs(q, sc)
    a = draw_action(probs, u_action)
    nxt = [v0._next_state(x, k, shocks, p) for k in range(3)]
    losses = np.array([v0._loss(nxt[k], p.dr[k], sc["inflation_target"], sc["unemployment_target"],
                                sc["lambda_u"], sc["mu_r"]) for k in range(3)])
    return q, probs, a, losses, nxt[a]


def generate(seed, sc):
    """Devuelve (factual, verdad). Flujos independientes init/acciones/choques derivados de la semilla."""
    ss_init, ss_act, ss_shock = np.random.SeedSequence(seed).spawn(3)
    rng_init, rng_act, rng_shock = (np.random.default_rng(s) for s in (ss_init, ss_act, ss_shock))
    p = make_params(sc)
    sig = sigma_vec(p, sc["noise_level"])
    x = v0._initial_state(rng_init, p)
    fact, truth = [], []
    for t in range(sc["n_periods"] + sc["burn_in"]):
        u = rng_act.random()
        shocks = rng_shock.normal(0.0, sig)
        q, probs, a, losses, x_next = step(x, p, sc, u, shocks)
        fact.append({"t": t, "inflation": x[0], "unemployment": x[1], "fedfunds": x[2], "activity": x[3],
                     "treatment": T_LEVELS[a], "loss": float(losses[a])})
        truth.append({"t": t, **{f"q_{T}": q[k] for k, T in enumerate(T_LEVELS)},
                      **{f"loss_pot_{T}": losses[k] for k, T in enumerate(T_LEVELS)},
                      **{f"p_behavior_{T}": probs[k] for k, T in enumerate(T_LEVELS)}})
        x = x_next
    b = sc["burn_in"]
    dates = pd.date_range(sc["start_date"], periods=sc["n_periods"], freq="MS")
    F = pd.DataFrame(fact).iloc[b:].reset_index(drop=True)
    V = pd.DataFrame(truth).iloc[b:].reset_index(drop=True)
    F["t"] = np.arange(len(F)); V["t"] = np.arange(len(V))
    F.insert(1, "date", dates); V.insert(1, "date", dates)
    return F, V
