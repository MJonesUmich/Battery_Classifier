"""Export a self-contained logistic regression bundle for JavaScript inference.

This script rebuilds the tabular training dataset, retrains the logistic
regression model (with StandardScaler + LabelEncoder), and serializes the
weights, intercepts, scaler statistics, and metadata to JSON. The resulting
JSON can be imported by a React/Node project to replicate the same prediction
logic using plain JavaScript (see README snippet in assistant response).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler

from prepare_logreg_dataset import build_sample_from_files, scrub_values

ASSETS_DIR = Path("assets/processed")
OUTPUT_PATH = Path("frontend/battery-best/src/assets/logreg_model.json")
TRAINING_COLUMN = "chemistry"


def collect_dataset(root_dir: Path, clip: bool = True) -> pd.DataFrame:
    """Walk the processed assets tree and rebuild the training dataframe."""

    all_rows: List[pd.DataFrame] = []
    for chemistry_dir in sorted(root_dir.iterdir()):
        if not chemistry_dir.is_dir():
            continue
        chemistry = chemistry_dir.name
        for battery_dir in sorted(chemistry_dir.iterdir()):
            if not battery_dir.is_dir():
                continue

            charge_file: Optional[Path] = None
            discharge_file: Optional[Path] = None
            for csv_path in sorted(battery_dir.glob("*.csv")):
                name = csv_path.name.lower()
                if "error_log" in name:
                    continue
                if "discharge" in name:
                    discharge_file = csv_path
                elif "charge" in name:
                    charge_file = csv_path

            if not (charge_file or discharge_file):
                continue

            sample = build_sample_from_files(
                chemistry=chemistry,
                charge_file=charge_file,
                discharge_file=discharge_file,
                clip=clip,
            )
            if sample is not None:
                all_rows.append(sample)

    if not all_rows:
        raise RuntimeError("No samples found while rebuilding the dataset.")

    raw_df = pd.concat(all_rows, ignore_index=True)
    clean_df = scrub_values(raw_df).dropna().reset_index(drop=True)
    return clean_df


def export_bundle(df: pd.DataFrame, output_path: Path) -> None:
    y = df[TRAINING_COLUMN]
    X = df.drop(columns=[TRAINING_COLUMN])
    feature_names = list(X.columns)

    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = LogisticRegression(
        solver="lbfgs",
        multi_class="auto",
        max_iter=2000,
        random_state=42,
    )
    model.fit(X_scaled, y_encoded)

    bundle = {
        "feature_names": feature_names,
        "classes": encoder.classes_.tolist(),
        "coef": model.coef_.tolist(),
        "intercept": model.intercept_.tolist(),
        "scaler": {
            "mean": scaler.mean_.tolist(),
            "scale": scaler.scale_.tolist(),
        },
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(bundle, indent=2))
    print(f"Exported bundle with {len(feature_names)} features to {output_path}")


def main() -> None:
    df = collect_dataset(ASSETS_DIR, clip=True)
    print(f"Rebuilt dataset with {len(df)} rows.")
    export_bundle(df, OUTPUT_PATH)


if __name__ == "__main__":
    main()

