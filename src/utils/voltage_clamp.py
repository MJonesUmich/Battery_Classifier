"""Shared helpers for enforcing the 3.0–3.6 V production-safe voltage window."""

from __future__ import annotations

from typing import Optional

import pandas as pd

SAFE_VOLTAGE_MIN: float = 3.0
SAFE_VOLTAGE_MAX: float = 3.6


def clamp_voltage_column(
    df: pd.DataFrame,
    column: str = "Voltage(V)",
    vmin: Optional[float] = SAFE_VOLTAGE_MIN,
    vmax: Optional[float] = SAFE_VOLTAGE_MAX,
) -> pd.DataFrame:
    """Clamp a dataframe's voltage column to the shared safe range in place."""

    if column not in df.columns:
        return df

    if vmin is not None or vmax is not None:
        df[column] = df[column].clip(lower=vmin, upper=vmax)
    return df

