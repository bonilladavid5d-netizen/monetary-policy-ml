"""
simular_datos.py
================

Simulación de datos macroeconómicos mensuales con un proceso generador (DGP)
conocido para un proyecto de política monetaria + machine learning.

Para cada mes t el módulo genera:
    - Estado macro X_t = (inflation, unemployment, fedfunds, activity).
    - Pérdidas contrafactuales L_t^T bajo cada tratamiento T in {T0, T1, T2}
      calculadas con los MISMOS choques (clean comparison).
    - Tratamiento óptimo verdadero T*(X_t) = argmin_T L_t^T.
    - Tratamiento observado T_t mediante una política softmax con ruido
      (mantiene positividad: P(T=k|X) > 0 para todo k).
    - Pérdida observada L_t = L_t^{T_t}.

El estado se avanza usando el tratamiento observado (no el óptimo), de modo
que la trayectoria sea realista. Se descarta un burn-in inicial.

Pérdida:
    L_t^T = (pi_{t+1}^T - pi*)^2
            + lambda_u * (u_{t+1}^T - u*)^2
            + mu_r    * (dr^T)^2     # penaliza la DECISIÓN, no la realización

No-linealidad introducida (canal tasa -> inflación):
    beta_pi_r_eff(pi_t) = beta_pi_r * (1 + 0.5 * tanh(pi_t - pi*))

Uso:
    python simular_datos.py
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------
# [PATCH-B1] ruta absoluta original reemplazada por ruta relativa al repo; override con argv[1]
import sys as _sys
DEFAULT_OUTPUT_DIR = (_sys.argv[1] if len(_sys.argv) > 1 else
                      os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "outputs", "repro", "simulated"))
OUTPUT_FILENAME = "base_sintetica.csv"

TREATMENT_LABELS = {0: "T0 (neutral)",
                    1: "T1 (contractiva)",
                    2: "T2 (expansiva)"}


# ---------------------------------------------------------------------------
# Parámetros estructurales del DGP
# ---------------------------------------------------------------------------
@dataclass
class DGPParams:
    """Parámetros estructurales del proceso generador."""
    # Shift de la tasa por tratamiento (T0, T1, T2) en puntos porcentuales
    dr: Tuple[float, float, float] = (0.0, 0.50, -0.50)

    # Persistencias AR(1)
    alpha_pi: float = 0.75
    alpha_u:  float = 0.85
    alpha_y:  float = 0.70
    alpha_r:  float = 0.90

    # Canales de transmisión (signos coherentes con teoría)
    beta_pi_r:  float = -0.35
    gamma_pi_y: float =  0.25
    beta_u_y:   float = -0.30
    gamma_u_r:  float =  0.20
    delta_y_r:  float =  0.40

    # Desv. estándar de los choques (antes de escalar por noise_level)
    sigma_pi: float = 0.30
    sigma_u:  float = 0.20
    sigma_y:  float = 0.25
    sigma_r:  float = 0.15

    # Medias de largo plazo
    pi_star: float = 2.0
    u_star:  float = 4.5
    r_star:  float = 2.5


# ---------------------------------------------------------------------------
# Núcleo del DGP
# ---------------------------------------------------------------------------
def _initial_state(rng, p):
    """Estado inicial cerca de las medias de largo plazo."""
    return np.array([
        p.pi_star + rng.normal(0.0, 0.5),
        p.u_star  + rng.normal(0.0, 0.5),
        p.r_star  + rng.normal(0.0, 0.5),
        0.0       + rng.normal(0.0, 0.5),
    ])


def _next_state(x, treatment_idx, shocks, p):
    """
    Aplica la dinámica del DGP dado un tratamiento.
    Incluye no-linealidad: beta_pi_r_eff = beta_pi_r * (1 + 0.5*tanh(pi_t - pi*))
    """
    pi_t, u_t, r_t, y_t = x
    dr = p.dr[treatment_idx]
    r_tilde = r_t + dr
    gap_r = r_tilde - p.r_star

    beta_pi_r_eff = p.beta_pi_r * (1.0 + 0.5 * np.tanh(pi_t - p.pi_star))

    pi_next = (p.alpha_pi * (pi_t - p.pi_star) + p.pi_star
               + beta_pi_r_eff * gap_r
               + p.gamma_pi_y * y_t
               + shocks[0])

    u_next = (p.alpha_u * (u_t - p.u_star) + p.u_star
              + p.beta_u_y * y_t
              + p.gamma_u_r * gap_r
              + shocks[1])

    y_next = (p.alpha_y * y_t
              - p.delta_y_r * gap_r
              + shocks[2])

    r_next = (p.alpha_r * r_t
              + (1.0 - p.alpha_r) * r_tilde
              + shocks[3])

    return np.array([pi_next, u_next, r_next, y_next])


def _loss(x_next, dr_t, inflation_target, unemployment_target, lambda_u, mu_r):
    """
    Pérdida macroeconómica con suavizamiento sobre la DECISIÓN de política
    (no sobre la realización ruidosa de la tasa):

        L = (pi_next - pi*)^2 + lambda_u*(u_next - u*)^2 + mu_r * dr_t^2

    donde dr_t es el shift de tasa inducido por el tratamiento elegido.
    """
    pi_n, u_n, _, _ = x_next
    return ((pi_n - inflation_target) ** 2
            + lambda_u * (u_n - unemployment_target) ** 2
            + mu_r * dr_t ** 2)


def _behavior_policy(losses_hat, rng, tau):
    """Softmax sobre el negativo de la pérdida estimada (mantiene positividad)."""
    logits = -tau * (losses_hat - losses_hat.min())
    weights = np.exp(logits)
    probs = weights / weights.sum()
    return int(rng.choice(3, p=probs))


# ---------------------------------------------------------------------------
# Función principal
# ---------------------------------------------------------------------------
def generate_synthetic_macro_data(
    n_periods: int = 600,
    start_date: str = "1975-01-01",
    random_seed: int = 42,
    lambda_u: float = 1.0,
    mu_r: float = 2.0,
    inflation_target: float = 2.0,
    unemployment_target: float = 4.5,
    noise_level: float = 1.0,
    tau: float = 1.2,
    burn_in: int = 24,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    save: bool = True,
    params: DGPParams | None = None,
) -> pd.DataFrame:
    """Genera el panel sintético y lo guarda como CSV."""
    rng = np.random.default_rng(random_seed)
    p = params if params is not None else DGPParams()
    p.pi_star = inflation_target
    p.u_star = unemployment_target

    sigma = np.array([p.sigma_pi, p.sigma_u, p.sigma_y, p.sigma_r]) * noise_level
    x = _initial_state(rng, p)
    total_periods = n_periods + burn_in

    records = []
    for t in range(total_periods):
        # 1) Choques comunes a los 3 contrafactuales del periodo
        shocks = rng.normal(0.0, sigma)

        # 2) Contrafactuales bajo cada tratamiento
        losses = np.empty(3)
        x_next_by_T = [None, None, None]
        for k in range(3):
            x_next_k = _next_state(x, k, shocks, p)
            x_next_by_T[k] = x_next_k
            losses[k] = _loss(
                x_next_k, dr_t=p.dr[k],
                inflation_target=inflation_target,
                unemployment_target=unemployment_target,
                lambda_u=lambda_u, mu_r=mu_r,
            )

        true_optimal = int(np.argmin(losses))

        # 3) Tratamiento observado (banco central no-omnisciente)
        losses_hat = losses + rng.normal(0.0, 0.5, size=3)
        treatment = _behavior_policy(losses_hat, rng, tau=tau)

        records.append({
            "inflation":              x[0],
            "unemployment":           x[1],
            "fedfunds":               x[2],
            "activity":               x[3],
            "treatment":              treatment,
            "loss":                   float(losses[treatment]),
            "loss_T0":                float(losses[0]),
            "loss_T1":                float(losses[1]),
            "loss_T2":                float(losses[2]),
            "true_optimal_treatment": true_optimal,
            "is_optimal":             int(treatment == true_optimal),
        })

        # 4) Avanzar el estado con el tratamiento OBSERVADO
        x = x_next_by_T[treatment]

    df = pd.DataFrame.from_records(records).iloc[burn_in:].reset_index(drop=True)
    df.insert(0, "date", pd.date_range(start=start_date, periods=len(df), freq="MS"))

    if save:
        os.makedirs(output_dir, exist_ok=True)
        out_path = os.path.join(output_dir, OUTPUT_FILENAME)
        df.to_csv(out_path, index=False)
        print(f"[OK] CSV guardado en: {out_path}")

    return df


# ---------------------------------------------------------------------------
# Diagnóstico
# ---------------------------------------------------------------------------
def _print_diagnostics(df: pd.DataFrame) -> None:
    print("\n=== Distribución de tratamientos observados ===")
    print(df["treatment"].map(TREATMENT_LABELS).value_counts(normalize=True).round(3))

    print("\n=== Distribución de tratamientos óptimos verdaderos ===")
    print(df["true_optimal_treatment"].map(TREATMENT_LABELS).value_counts(normalize=True).round(3))

    print("\n=== Pérdida promedio por tratamiento observado ===")
    print(df.groupby("treatment")["loss"].mean().round(4).rename(index=TREATMENT_LABELS))

    match_rate = df["is_optimal"].mean()
    print(f"\n=== % observado == óptimo: {match_rate:.2%} ===")

    print("\n=== Estadísticos del estado ===")
    print(df[["inflation", "unemployment", "fedfunds", "activity"]].describe().round(3))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    df = generate_synthetic_macro_data(
        n_periods=600,
        start_date="1975-01-01",
        random_seed=42,
        lambda_u=1.0,
        mu_r=2.0,
        inflation_target=2.0,
        unemployment_target=4.5,
        noise_level=1.0,
        tau=1.2,
        burn_in=24,
        output_dir=DEFAULT_OUTPUT_DIR,
        save=True,
    )
    _print_diagnostics(df)
    print(f"\nShape: {df.shape}")
