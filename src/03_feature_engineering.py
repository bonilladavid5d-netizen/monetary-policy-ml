"""
03_feature_engineering.py
-------------------------
Construcción de variables para el modelo de pérdida macroeconómica.

Input:
    base_macro_fred_unida.csv  (columnas: date, unrate, cpi, fedfunds, indpro)

Output:
    base_macro_features.csv    (panel mensual con predictores y targets)

Variables construidas
=====================
Transformaciones contemporáneas
    - infl_m      : inflación mensual del CPI, 100*(CPI_t / CPI_{t-1} - 1)
    - infl_yoy    : inflación interanual del CPI, 100*(CPI_t / CPI_{t-12} - 1)
    - indpro_g_m  : crecimiento mensual de INDPRO
    - indpro_g_yoy: crecimiento interanual de INDPRO

Rezagos (k in {1, 3, 6, 12})  -> sufijo _lagK
    - infl_m_lagK, infl_yoy_lagK
    - unrate_lagK
    - fedfunds_lagK
    - indpro_lagK           (nivel)
    - indpro_g_m_lagK       (crecimiento mensual)
    - indpro_g_yoy_lagK     (crecimiento interanual)

Targets (forward, horizonte H = 12 meses)
    - infl_m_sum_fwd12   : Σ_{h=1..12} π^m_{t+h}     (inflación acumulada próximos 12m)
    - infl_yoy_fwd12     : π^a_{t+12}                 (inflación interanual dentro de 12m)
    - unrate_avg_fwd12   : (1/12) Σ_{h=1..12} UNRATE_{t+h}
    - loss_cum_lamLL     : infl_m_sum_fwd12 + λ * unrate_avg_fwd12
    - loss_yoy_lamLL     : infl_yoy_fwd12   + λ * unrate_avg_fwd12
      (LL = 050, 100, 200 para λ = 0.5, 1.0, 2.0)

Notas metodológicas
-------------------
- Todo en porcentaje para facilitar interpretación.
- El target usa SOLO información futura; los predictores usan valor actual o
  rezagos. No se mezclan ambos lados, evitando data leakage.
- Las últimas 12 filas quedan con target NaN: se dejan en el CSV para inspección,
  pero al entrenar hay que filtrarlas (ver función `get_train_frame`).
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Configuración
# ---------------------------------------------------------------------------
# [PATCH-B1] rutas absolutas originales -> argv[1] (input) y argv[2] (output)
import sys as _sys
_REPO = Path(__file__).resolve().parent.parent
INPUT_PATH = Path(_sys.argv[1]) if len(_sys.argv) > 1 else _REPO / "data" / "derived" / "base_macro_fred_unida.csv"
OUTPUT_PATH = Path(_sys.argv[2]) if len(_sys.argv) > 2 else _REPO / "outputs" / "repro" / "pipeline" / "base_macro_features.csv"

LAGS = [1, 3, 6, 12]
HORIZON = 12
LAMBDAS = {"050": 0.5, "100": 1.0, "200": 2.0}


# ---------------------------------------------------------------------------
# Carga y limpieza
# ---------------------------------------------------------------------------
def load_data(path: Path) -> pd.DataFrame:
    """Lee el CSV, parsea fechas, fuerza numérico y ordena por fecha."""
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    df["date"] = pd.to_datetime(df["date"])
    for col in ["unrate", "cpi", "fedfunds", "indpro"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.sort_values("date").reset_index(drop=True)

    # Chequeo de frecuencia mensual regular
    freq_check = df["date"].diff().dt.days.dropna()
    if not freq_check.between(27, 33).all():
        gaps = df.loc[freq_check[~freq_check.between(27, 33)].index, "date"]
        print(f"[warn] gaps no mensuales detectados cerca de: {gaps.tolist()[:5]}")

    return df


# ---------------------------------------------------------------------------
# Transformaciones contemporáneas
# ---------------------------------------------------------------------------
def add_transformations(df: pd.DataFrame) -> pd.DataFrame:
    """Inflación y crecimiento de INDPRO (mensual e interanual)."""
    df = df.copy()
    df["infl_m"] = 100 * (df["cpi"] / df["cpi"].shift(1) - 1)
    df["infl_yoy"] = 100 * (df["cpi"] / df["cpi"].shift(12) - 1)
    df["indpro_g_m"] = 100 * (df["indpro"] / df["indpro"].shift(1) - 1)
    df["indpro_g_yoy"] = 100 * (df["indpro"] / df["indpro"].shift(12) - 1)
    return df


# ---------------------------------------------------------------------------
# Rezagos
# ---------------------------------------------------------------------------
def add_lags(df: pd.DataFrame, lags: list[int]) -> pd.DataFrame:
    """Crea rezagos para los predictores."""
    df = df.copy()
    vars_to_lag = [
        "infl_m",
        "infl_yoy",
        "unrate",
        "fedfunds",
        "indpro",
        "indpro_g_m",
        "indpro_g_yoy",
    ]
    for v in vars_to_lag:
        for k in lags:
            df[f"{v}_lag{k}"] = df[v].shift(k)
    return df


# ---------------------------------------------------------------------------
# Target: pérdida macroeconómica hacia adelante
# ---------------------------------------------------------------------------
def add_forward_targets(
    df: pd.DataFrame, horizon: int, lambdas: dict[str, float]
) -> pd.DataFrame:
    """
    Construye la pérdida macro objetivo usando únicamente información futura.

        infl_m_sum_fwd12 = Σ_{h=1..H} π^m_{t+h}
        infl_yoy_fwd12   = π^a_{t+H}
        unrate_avg_fwd12 = (1/H) Σ_{h=1..H} UNRATE_{t+h}
        loss_cum = infl_m_sum_fwd12 + λ * unrate_avg_fwd12
        loss_yoy = infl_yoy_fwd12   + λ * unrate_avg_fwd12
    """
    df = df.copy()

    # Σ inflación mensual futura (acumulada, h=1..H)
    infl_fwd_terms = [df["infl_m"].shift(-h) for h in range(1, horizon + 1)]
    df[f"infl_m_sum_fwd{horizon}"] = sum(infl_fwd_terms)

    # Inflación interanual dentro de H meses
    df[f"infl_yoy_fwd{horizon}"] = df["infl_yoy"].shift(-horizon)

    # Promedio de desempleo en los próximos H meses
    unrate_fwd_terms = [df["unrate"].shift(-h) for h in range(1, horizon + 1)]
    df[f"unrate_avg_fwd{horizon}"] = sum(unrate_fwd_terms) / horizon

    # Pérdidas combinadas
    for tag, lam in lambdas.items():
        df[f"loss_cum_lam{tag}"] = (
            df[f"infl_m_sum_fwd{horizon}"] + lam * df[f"unrate_avg_fwd{horizon}"]
        )
        df[f"loss_yoy_lam{tag}"] = (
            df[f"infl_yoy_fwd{horizon}"] + lam * df[f"unrate_avg_fwd{horizon}"]
        )

    return df


# ---------------------------------------------------------------------------
# Helpers para entrenamiento
# ---------------------------------------------------------------------------
def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """Columnas que pueden usarse como X (nada con 'fwd' ni con 'loss')."""
    return [
        c
        for c in df.columns
        if c != "date"
        and not c.startswith("loss_")
        and "fwd" not in c
    ]


def get_train_frame(
    df: pd.DataFrame, target: str, feature_cols: list[str] | None = None
) -> pd.DataFrame:
    """Devuelve frame limpio listo para fit: drop NaN en X e y."""
    if feature_cols is None:
        feature_cols = get_feature_columns(df)
    cols = ["date"] + feature_cols + [target]
    return df[cols].dropna().reset_index(drop=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    df = load_data(INPUT_PATH)
    print(f"[info] cargadas {len(df)} filas, de {df['date'].min().date()} "
          f"a {df['date'].max().date()}")

    df = add_transformations(df)
    df = add_lags(df, LAGS)
    df = add_forward_targets(df, HORIZON, LAMBDAS)

    df.to_csv(OUTPUT_PATH, index=False)
    print(f"[ok] escrito {OUTPUT_PATH} con shape {df.shape}")

    # Reporte rápido
    feat_cols = get_feature_columns(df)
    train = get_train_frame(df, target="loss_cum_lam100")
    print(f"[info] features disponibles: {len(feat_cols)}")
    print(f"[info] muestra entrenable (loss_cum_lam100, sin NaN): "
          f"{len(train)} filas, {train['date'].min().date()} -> "
          f"{train['date'].max().date()}")

    print("\n[preview] estadísticas de los targets:")
    target_cols = [c for c in df.columns if c.startswith("loss_") or "fwd" in c]
    print(df[target_cols].describe().round(3).T[["mean", "std", "min", "max"]])


if __name__ == "__main__":
    main()
