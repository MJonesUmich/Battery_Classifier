#!/usr/bin/env python
"""Minimal helper to replicate the notebook's logistic-regression preprocessing.

Call ``main(file_path=...)`` with a single CSV (or the folder that contains the
charge/discharge CSVs). The script will:
1. Locate charge/discharge files living next to the provided path.
2. Aggregate the ≈100 sequential rows into summary statistics.
3. Drop the unused columns so the output matches the notebook's ``strip_df``.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd

DESIRED_FEATURES = [
    "cycle_index",
    "normalized_time",
    "voltage_v",
    "c_rate",
    "temperature_k",
]

STRIP_COLS = [
    "charge_cycle_index_min",
    "charge_cycle_index_max",
    "charge_cycle_index_mean",
    "charge_cycle_index_std",
    "charge_normalized_time_min",
    "charge_normalized_time_max",
    "discharge_c_rate_min",
    "discharge_c_rate_max",
    "discharge_c_rate_std",
    "charge_c_rate_min",
    "charge_c_rate_max",
    "charge_c_rate_std",
    "discharge_temperature_k_max",
    "discharge_temperature_k_min",
    "discharge_temperature_k_std",
    "charge_temperature_k_max",
    "charge_temperature_k_min",
    "charge_temperature_k_std",
    "discharge_temperature_k_mean",
    "file",
    "discharge_cycle_index_mean",
    "discharge_cycle_index_max",
    "discharge_cycle_index_min",
    "discharge_cycle_index_std",
    "discharge_normalized_time_max",
    "discharge_normalized_time_min",
    "discharge_normalized_time_std",
    "charge_normalized_time_std",
    "charge_normalized_time_mean",
    "discharge_normalized_time_mean",
]


def summarize_features(
    df: pd.DataFrame,
    chemistry: str,
    filepath: Path,
    desired_features: List[str],
    clip: bool,
) -> Optional[pd.DataFrame]:
    """Aggregate the sequential measurements into summary statistics."""

    available_cols = [col for col in desired_features if col in df.columns]
    if not available_cols:
        raise ValueError(
            f"None of the desired features {desired_features} exist in {filepath}"
        )

    tdf = df[available_cols].copy()
    tdf = tdf.apply(pd.to_numeric, errors="coerce")

    if clip and "voltage_v" in tdf.columns:
        tdf = tdf[(tdf["voltage_v"] >= 3.0) & (tdf["voltage_v"] <= 3.6)]

    if tdf.empty:
        return None

    agg_data = {}
    for col in available_cols:
        series = tdf[col]
        agg_data[f"{col}_mean"] = series.mean()
        agg_data[f"{col}_std"] = series.std()
        agg_data[f"{col}_min"] = series.min()
        agg_data[f"{col}_max"] = series.max()

    agg_data["chemistry"] = chemistry
    agg_data["file"] = str(filepath)

    return pd.DataFrame([agg_data])


def _rename_for_phase(df: pd.DataFrame, phase: str) -> pd.DataFrame:
    """Prefix columns for charge/discharge while leaving metadata untouched."""
    rename_map = {
        col: f"{phase}_{col}"
        for col in df.columns
        if col not in {"chemistry", "file"}
    }
    return df.rename(columns=rename_map)


def build_sample_from_files(
    chemistry: str,
    charge_file: Optional[Path],
    discharge_file: Optional[Path],
    clip: bool,
) -> Optional[pd.DataFrame]:
    """Create a single row of aggregated features from the specified files."""

    frames = []

    if charge_file:
        charge_df = pd.read_csv(charge_file)
        summarized = summarize_features(
            charge_df, chemistry, charge_file, DESIRED_FEATURES, clip
        )
        if summarized is not None:
            frames.append(_rename_for_phase(summarized, "charge"))

    if discharge_file:
        discharge_df = pd.read_csv(discharge_file)
        summarized = summarize_features(
            discharge_df, chemistry, discharge_file, DESIRED_FEATURES, clip
        )
        if summarized is not None:
            frames.append(_rename_for_phase(summarized, "discharge"))

    if not frames:
        return None

    combined = pd.concat(frames, axis=1)
    # Remove duplicate metadata columns that arise when both phases are present.
    combined = combined.loc[:, ~combined.columns.duplicated()]
    combined["chemistry"] = chemistry
    return combined


def scrub_values(input_df: pd.DataFrame) -> pd.DataFrame:
    """Drop unused columns to match the notebook's `strip_df`."""
    strip_df = input_df.drop(columns=STRIP_COLS, errors="ignore").copy()

    # Remove any lingering file metadata columns (e.g., file, file.1).
    file_cols = [
        col for col in strip_df.columns if col.split(".")[0].lower() == "file"
    ]
    if file_cols:
        strip_df = strip_df.drop(columns=file_cols)

    return strip_df


def _detect_phase_files(path: Path) -> Tuple[Optional[Path], Optional[Path]]:
    """Find charge/discharge CSVs given a file or folder path."""

    if path.is_dir():
        folder = path
        seed_file = None
    else:
        folder = path.parent
        seed_file = path

    charge_file = None
    discharge_file = None

    for csv_path in folder.glob("*.csv"):
        name = csv_path.name.lower()
        if "error_log" in name:
            continue
        if "discharge" in name:
            discharge_file = csv_path
        elif "charge" in name:
            charge_file = csv_path

    # If user passed a single CSV without charge/discharge in its name, treat it
    # as the only available phase.
    if seed_file and not ("charge" in seed_file.name.lower() or "discharge" in seed_file.name.lower()):
        charge_file = seed_file

    return charge_file, discharge_file


def _infer_chemistry(path: Path, override: Optional[str]) -> str:
    """Guess the chemistry label based on the folder structure."""

    if override:
        return override

    # e.g., .../LFP/lfo_cell_1/charge.csv -> chemistry = LFP
    try:
        return path.resolve().parents[1].name
    except IndexError:
        return "UNKNOWN"


def process_single_path(
    path: Path,
    chemistry: Optional[str],
    clip: bool,
) -> pd.DataFrame:
    """Collapse a single battery folder/file into the logistic-regression features."""

    charge_file, discharge_file = _detect_phase_files(path)
    label = _infer_chemistry(path if path.is_file() else charge_file or discharge_file or path, chemistry)

    sample = build_sample_from_files(
        chemistry=label,
        charge_file=charge_file,
        discharge_file=discharge_file,
        clip=clip,
    )
    if sample is None:
        raise RuntimeError(f"Unable to summarize any CSVs located near {path}")

    strip_df = scrub_values(sample).dropna().reset_index(drop=True)
    if strip_df.empty:
        raise RuntimeError("Processed dataframe is empty after cleaning.")

    return strip_df


def main(file_path: str, chemistry: Optional[str] = None, clip: bool = True, output: Optional[str] = None) -> pd.DataFrame:
    """Convenience wrapper so users can call `main('path/to/file.csv')` directly."""

    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(file_path)

    dataset = process_single_path(path, chemistry=chemistry, clip=clip)

    output_path = Path(output) if output else Path("artifacts/logreg_sample.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(output_path, index=False)
    print(f"Saved {len(dataset)} row(s) to {output_path}")
    return dataset


if __name__ == "__main__":
    default_path = r"assets\processed\LCO\Capacity_25C\Capacity_25C_charge_aggregated_data.csv"
    main(file_path=default_path)

