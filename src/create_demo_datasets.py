"""Generate 100-point demo CSVs for the React app upload workflow.

This script follows the workflow documented in `frontend/instruction.md`:
it samples evenly along `sample_index`, preserves the required columns, and
emits lightweight charge/discharge CSVs that live under
`frontend/battery-best/public/datasets/`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

# Columns the frontend pipeline expects to exist in every demo CSV.
REQUIRED_COLUMNS: List[str] = [
    "battery_id",
    "chemistry",
    "cycle_index",
    "sample_index",
    "normalized_time",
    "elapsed_time_s",
    "voltage_v",
    "current_a",
    "c_rate",
    "temperature_k",
]

DEFAULT_SAMPLE_COUNT = 100
OUTPUT_DIR = Path("frontend/battery-best/public/datasets")

# Source files pulled from two different chemistries.
DEMO_SOURCES: List[Dict[str, str]] = [
    {
        "label": "LCO_sample",
        "charge": "assets/processed/LCO/Capacity_25C/Capacity_25C_charge_aggregated_data.csv",
        "discharge": "assets/processed/LCO/Capacity_25C/Capacity_25C_discharge_aggregated_data.csv",
    },
    {
        "label": "LFP_sample",
        "charge": "assets/processed/LFP/lfp_cell_1/lfp_cell_1_charge_aggregated_data.csv",
        "discharge": "assets/processed/LFP/lfp_cell_1/lfp_cell_1_discharge_aggregated_data.csv",
    },
]


def sample_csv(input_path: Path, output_path: Path, n_samples: int = DEFAULT_SAMPLE_COUNT) -> None:
    """Down-sample the CSV while keeping the required columns and row order."""

    df = pd.read_csv(input_path)
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"{input_path} is missing columns: {missing}")

    subset = df[REQUIRED_COLUMNS].copy()
    if len(subset) > n_samples:
        # Evenly spaced indices based on sample_index order.
        subset = subset.sort_values("sample_index")
        idx = np.linspace(0, len(subset) - 1, n_samples, dtype=int)
        subset = subset.iloc[idx]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    subset.to_csv(output_path, index=False)
    print(f"Wrote {len(subset)} rows to {output_path}")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for source in DEMO_SOURCES:
        label = source["label"]
        for phase in ("charge", "discharge"):
            input_path = Path(source[phase])
            output_path = OUTPUT_DIR / f"{label}_{phase}.csv"
            sample_csv(input_path, output_path)


if __name__ == "__main__":
    main()

