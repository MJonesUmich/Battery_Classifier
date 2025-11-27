"""Generate combined (charge + discharge) demo CSVs for the React upload flow.

Each output file contains ~200 rows: the first half are evenly sampled charge
points, the second half are discharge points. A `phase` column is added so the
frontend can split the rows without needing two separate uploads.
"""

from __future__ import annotations

import json
import random
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

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
REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = REPO_ROOT / "frontend" / "battery-best" / "public" / "datasets"
DATA_ROOT = REPO_ROOT / "assets" / "processed"


def downsample(input_path: Path, n_samples: int) -> pd.DataFrame:
    df = pd.read_csv(input_path)
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"{input_path} is missing columns: {missing}")

    subset = df[REQUIRED_COLUMNS].copy()
    subset = subset.sort_values("sample_index")
    if len(subset) > n_samples:
        idx = np.linspace(0, len(subset) - 1, n_samples, dtype=int)
        subset = subset.iloc[idx]
    return subset.reset_index(drop=True)


def build_demo_file(charge_path: Path, discharge_path: Path, output_path: Path) -> None:
    charge_subset = downsample(charge_path, DEFAULT_SAMPLE_COUNT).assign(phase="charge")
    discharge_subset = downsample(discharge_path, DEFAULT_SAMPLE_COUNT).assign(phase="discharge")

    columns = ["phase", *REQUIRED_COLUMNS]
    combined = pd.concat([charge_subset, discharge_subset], ignore_index=True)[columns]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output_path, index=False)
    print(f"Wrote {len(combined)} rows to {output_path}")


def slugify(value: str) -> str:
    return re.sub(r"[^0-9a-zA-Z\-_.]+", "-", value)


def discover_pool() -> List[Dict[str, str]]:
    entries: List[Dict[str, str]] = []
    if not DATA_ROOT.exists():
        return entries

    for chemistry_dir in sorted(p for p in DATA_ROOT.iterdir() if p.is_dir()):
        chemistry = chemistry_dir.name
        for battery_dir in sorted(p for p in chemistry_dir.iterdir() if p.is_dir()):
            files = [
                csv_path
                for csv_path in battery_dir.glob("*.csv")
                if "error_log" not in csv_path.name.lower()
            ]
            charge_path = next((f for f in files if "charge" in f.name.lower()), None)
            discharge_path = next((f for f in files if "discharge" in f.name.lower()), None)
            if not (charge_path and discharge_path):
                continue

            label = slugify(f"{chemistry}_{battery_dir.name}_sample")
            title = f"{chemistry.upper()} · {battery_dir.name}"
            description = f"{chemistry} / {battery_dir.name} combined cycle (auto-generated)."
            entries.append(
                {
                    "label": label,
                    "title": title,
                    "description": description,
                    "charge": str((charge_path).resolve()),
                    "discharge": str((discharge_path).resolve()),
                }
            )
    return entries


def select_sources(pool: List[Dict[str, str]], indices: List[int]) -> List[Dict[str, str]]:
    return [pool[idx] for idx in indices]


def main(random_choices: Optional[List[int]] = None) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pool = discover_pool()
    if not pool:
        raise RuntimeError(f"No valid charge/discharge pairs found under {DATA_ROOT}")

    if random_choices is None:
        count = min(4, len(pool))
        random_choices = random.sample(range(len(pool)), count)

    manifest_entries = []
    for source in select_sources(pool, random_choices):
        output_path = OUTPUT_DIR / f"{source['label']}.csv"
        build_demo_file(Path(source["charge"]), Path(source["discharge"]), output_path)
        size_kb = output_path.stat().st_size / 1024
        manifest_entries.append(
            {
                "id": source["label"],
                "title": source["title"],
                "description": source["description"],
                "file": output_path.name,
                "size": f"{size_kb:.1f} KB CSV",
            }
        )

    manifest_path = OUTPUT_DIR / "datasets.json"
    manifest_payload = {
        "generated_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "datasets": manifest_entries,
    }
    manifest_path.write_text(json.dumps(manifest_payload, indent=2))
    print(f"Wrote manifest with {len(manifest_entries)} entries to {manifest_path}")


if __name__ == "__main__":
    random.seed(345)#1515
    #999 lfp 全变成了nca, nmc 变成了LCO
    main()

