"""Tres figuras del bloque 2 (PNG estáticos, sin títulos que anticipen ganador).
Colores por entidad, fijos en las tres figuras: Ridge azul, RF naranja, media aqua,
persistencia amarillo; referencias fijas y oráculo en grises (paleta de referencia dataviz)."""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

C = {"ridge": "#2a78d6", "rf": "#eb6834", "media": "#1baf7a", "persistencia": "#eda100",
     "observado": "#0b0b0b", "ref": "#8a8986", "oraculo": "#52514e"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
NAMES = {"ridge": "Ridge", "rf": "Random Forest", "media": "Media de entrenamiento",
         "persistencia": "Persistencia", "siempre_T0": "Siempre T0\n(mantener)", "siempre_T1": "Siempre T1\n(+0,50 pp)",
         "siempre_T2": "Siempre T2\n(−0,50 pp)", "oraculo_exante": "Oráculo ex ante\n(mejor acción)"}


def _style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def fig_regret(run, cfg):
    S = pd.read_csv(run / "tablas" / "sintetico_por_semilla.csv")
    order = ["ridge", "rf", "siempre_T0", "siempre_T1", "siempre_T2", "oraculo_exante"]
    fig, ax = plt.subplots(figsize=(8.5, 5.0), dpi=150)
    rng = np.random.default_rng(0)
    for i, p in enumerate(order):
        v = S[S.politica == p]["regret_esperado_medio"].to_numpy()
        col = C.get(p, C["oraculo"] if p == "oraculo_exante" else C["ref"])
        ax.scatter(i + rng.uniform(-0.12, 0.12, len(v)), v, s=28, color=col, edgecolor="white", linewidth=1, zorder=3)
        ax.hlines(v.mean(), i - 0.25, i + 0.25, color=INK, linewidth=2, zorder=4)
        ax.annotate(f"{v.mean():.3f}", (i + 0.28, v.mean()), fontsize=8, color=INK, va="center")
    ax.set_xticks(range(len(order)), [NAMES[p] for p in order], fontsize=9, color=INK)
    ax.set_ylabel("Regret esperado medio en test\n(pérdida adicional frente a la mejor acción; menor es mejor)",
                  fontsize=9, color=INK)
    ax.set_title("Laboratorio S0_B2_exante: regret esperado a un paso por política", fontsize=10, color=INK, loc="left")
    _style(ax)
    fig.text(0.01, 0.055, "Cada punto es una repetición simulada (semillas 42–51), no un intervalo sobre la economía real; "
             "línea negra = media de las 10. T0 = mantener la tasa;\nT1 = subir 0,50 pp; T2 = bajar 0,50 pp (desplazamiento "
             "de la tasa usada por los canales del simulador). Regret = q(X, acción) − min_a q(X, a).", fontsize=7, color=MUTED)
    fig.text(0.01, 0.015, "Fuente: simulación S0_B2_exante (ecuaciones de simular_datos_v0_dr2.py); tablas/sintetico_por_semilla.csv",
             fontsize=7, color=MUTED)
    fig.tight_layout(rect=(0, 0.11, 1, 1))
    fig.savefig(run / "figuras" / "fig1_regret_sintetico.png"); plt.close(fig)


def fig_real_series(run, cfg):
    P = pd.read_csv(run / "tablas" / "real_predicciones.csv", parse_dates=["date"])
    P = P[(P.variante == cfg["real"]["main_variant"]) & (P.bloque == "test")]
    fig, ax = plt.subplots(figsize=(8.5, 5.2), dpi=150)
    ax.plot(P.date, P.Y, color=C["observado"], linewidth=2.2, label="Y observado", zorder=5)
    for k in ("ridge", "rf", "media", "persistencia"):
        ax.plot(P.date, P[f"pred_{k}"], color=C[k], linewidth=2, label=NAMES[k],
                linestyle="--" if k in ("media", "persistencia") else "-")
    ax.set_ylabel("Índice Y_t (puntos porcentuales)", fontsize=9, color=INK)
    ax.set_xlabel("Origen de predicción t (mes)", fontsize=9, color=INK)
    ax.set_title("Evaluación histórica reutilizada 2021-03 a 2024-01, información hasta t−1: observado y pronósticos",
                 fontsize=10, color=INK, loc="left")
    ax.legend(frameon=False, fontsize=8, ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.16))
    _style(ax)
    fig.text(0.01, 0.012, "Y_t = suma de las inflaciones mensuales de t+1 a t+12 + promedio del desempleo de t+1 a t+12 "
             "(índice construido para el ejercicio,\nno una medida oficial ni de bienestar). Cada punto del eje es un origen; "
             "los horizontes de orígenes vecinos comparten 11 meses.\nFuente: FRED CPIAUCSL, UNRATE, FEDFUNDS, INDPRO "
             "(descarga revisada, sin vintages); tablas/real_predicciones.csv", fontsize=7, color=MUTED,
             linespacing=1.4)
    fig.tight_layout(rect=(0, 0.11, 1, 1))
    fig.savefig(run / "figuras" / "fig2_real_observado_vs_pronostico.png"); plt.close(fig)


def fig_real_errors(run, cfg):
    M = pd.read_csv(run / "tablas" / "real_metricas.csv")
    M = M[M.bloque == "test"]
    order = ["ridge", "rf", "media", "persistencia"]
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.8), dpi=150, sharey=True)
    for ax, met, lab in zip(axes, ["rmse", "mae"], ["RMSE", "MAE"]):
        for i, m in enumerate(order):
            for var, mk, fc in (("lag1", "o", None), ("lag2", "D", "white")):
                v = M[(M.modelo == m) & (M.variante == var)][met].iloc[0]
                ax.scatter(v, i + (-0.12 if var == "lag1" else 0.12), marker=mk, s=60, color=C[m],
                           facecolor=fc or C[m], linewidth=1.8, zorder=3,
                           label=f"{'t−1' if var == 'lag1' else 't−2'}" if i == 0 else None)
                ax.annotate(f"{v:.2f}", (v, i + (-0.12 if var == "lag1" else 0.12)), xytext=(6, -3),
                            textcoords="offset points", fontsize=7, color=INK)
        ax.set_yticks(range(len(order)), [NAMES[m] for m in order], fontsize=9, color=INK)
        ax.set_xlabel(f"{lab} en test (puntos del índice Y)", fontsize=9, color=INK)
        ax.grid(axis="x", color=GRID, linewidth=0.8); ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.tick_params(colors=MUTED, labelsize=9)
    h = [plt.Line2D([], [], marker="o", color=MUTED, linestyle="", markersize=7, label="información hasta t−1 (principal)"),
         plt.Line2D([], [], marker="D", color=MUTED, markerfacecolor="white", linestyle="", markersize=7,
                    label="información hasta t−2 (sensibilidad)")]
    fig.legend(handles=h, frameon=False, fontsize=8, ncol=2, loc="lower center", bbox_to_anchor=(0.5, 0.04))
    fig.suptitle("Errores de pronóstico en 2021-03 a 2024-01 (n = 35 orígenes solapados) por disponibilidad de información",
                 fontsize=10, color=INK, x=0.01, ha="left")
    fig.text(0.01, 0.005, "Fuente: tablas/real_metricas.csv (bloque = test). Mismas 35 filas para los cuatro pronósticos.",
             fontsize=7, color=MUTED)
    fig.tight_layout(rect=(0, 0.12, 1, 0.95))
    fig.savefig(run / "figuras" / "fig3_real_errores_lag1_lag2.png"); plt.close(fig)


def make_all(run, cfg):
    (run / "figuras").mkdir(exist_ok=True)
    fig_regret(run, cfg); fig_real_series(run, cfg); fig_real_errors(run, cfg)
