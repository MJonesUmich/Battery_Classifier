"""Generate feature-only demo CSVs for the React upload flow.

Each output file contains a single row with the 11 model features derived from
paired charge/discharge CSVs.
"""

from __future__ import annotations

import json
import random

random.seed(2)
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

# Columns required to compute features.
REQUIRED_COLUMNS: List[str] = ["voltage_v", "c_rate", "temperature_k"]

FEATURE_COLUMNS: List[str] = [
    "charge_voltage_v_mean",
    "charge_voltage_v_std",
    "charge_voltage_v_min",
    "charge_voltage_v_max",
    "charge_c_rate_mean",
    "charge_temperature_k_mean",
    "discharge_voltage_v_mean",
    "discharge_voltage_v_std",
    "discharge_voltage_v_min",
    "discharge_voltage_v_max",
    "discharge_c_rate_mean",
]
REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = REPO_ROOT / "frontend" / "battery-best" / "public" / "datasets"
DATA_ROOT = REPO_ROOT / "assets" / "processed"


def _calc_stats(series: pd.Series) -> Dict[str, float]:
    return {
        "mean": series.mean(),
        "std": series.std(),
        "min": series.min(),
        "max": series.max(),
    }


def _summarize_phase(df: pd.DataFrame, filepath: Path, clip: bool = True) -> Dict[str, float]:
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"{filepath} is missing columns: {missing}")

    tdf = df[REQUIRED_COLUMNS].apply(pd.to_numeric, errors="coerce")
    if clip:
        tdf = tdf[(tdf["voltage_v"] >= 3.0) & (tdf["voltage_v"] <= 3.6)]
    if tdf.empty:
        raise ValueError(f"{filepath} has no rows after cleaning")

    voltage_stats = _calc_stats(tdf["voltage_v"])
    c_rate_stats = _calc_stats(tdf["c_rate"])
    temp_mean = tdf["temperature_k"].mean()

    return {
        "voltage": voltage_stats,
        "c_rate": c_rate_stats,
        "temperature_mean": temp_mean,
    }


def build_demo_file(charge_path: Path, discharge_path: Path, output_path: Path) -> None:
    charge_df = pd.read_csv(charge_path)
    discharge_df = pd.read_csv(discharge_path)

    charge_stats = _summarize_phase(charge_df, charge_path, clip=True)
    discharge_stats = _summarize_phase(discharge_df, discharge_path, clip=True)

    row = {
        "charge_voltage_v_mean": charge_stats["voltage"]["mean"],
        "charge_voltage_v_std": charge_stats["voltage"]["std"],
        "charge_voltage_v_min": charge_stats["voltage"]["min"],
        "charge_voltage_v_max": charge_stats["voltage"]["max"],
        "charge_c_rate_mean": charge_stats["c_rate"]["mean"],
        "charge_temperature_k_mean": charge_stats["temperature_mean"],
        "discharge_voltage_v_mean": discharge_stats["voltage"]["mean"],
        "discharge_voltage_v_std": discharge_stats["voltage"]["std"],
        "discharge_voltage_v_min": discharge_stats["voltage"]["min"],
        "discharge_voltage_v_max": discharge_stats["voltage"]["max"],
        "discharge_c_rate_mean": discharge_stats["c_rate"]["mean"],
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([row], columns=FEATURE_COLUMNS).to_csv(output_path, index=False)
    print(f"Wrote feature row to {output_path}")


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
                    "chemistry": chemistry,
                    "label": label,
                    "title": title,
                    "description": description,
                    "charge": str((charge_path).resolve()),
                    "discharge": str((discharge_path).resolve()),
                }
            )
    return entries


CHEMISTRY_PRIORITY = ["LCO", "LFP", "NCA", "NMC"]


def pick_unique_by_chemistry(
    pool: List[Dict[str, str]], preferred: Optional[List[str]] = None, k: int = 4
) -> List[Dict[str, str]]:
    buckets: Dict[str, List[Dict[str, str]]] = {}
    for entry in pool:
        chem = entry["chemistry"].upper()
        buckets.setdefault(chem, []).append(entry)

    selected: List[Dict[str, str]] = []
    ordered_chems = preferred or CHEMISTRY_PRIORITY
    for chem in ordered_chems:
        if chem in buckets and buckets[chem]:
            selected.append(random.choice(buckets[chem]))
            if len(selected) == k:
                return selected

    for chem, items in buckets.items():
        if len(selected) == k:
            break
        if any(choice["chemistry"].upper() == chem for choice in selected):
            continue
        selected.append(random.choice(items))

    if len(selected) < k:
        raise RuntimeError(
            f"Unable to assemble {k} unique chemistries from pool (found {len(selected)})"
        )
    return selected


def main(random_choices: Optional[List[int]] = None) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pool = discover_pool()
    if not pool:
        raise RuntimeError(f"No valid charge/discharge pairs found under {DATA_ROOT}")

    if random_choices is not None:
        chosen = [pool[idx] for idx in random_choices if idx < len(pool)]
    else:
        chosen = pick_unique_by_chemistry(pool, k=min(4, len(pool)))

    manifest_entries = []
    for source in chosen:
        output_path = OUTPUT_DIR / f"{source['label']}.csv"
        build_demo_file(Path(source["charge"]), Path(source["discharge"]), output_path)
        size_kb = output_path.stat().st_size / 1024
        manifest_entries.append(
            {
                "id": source["label"],
                "title": source["title"],
                "description": source["description"],
                "file": output_path.name,
                "size": "Feature CSV",
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
    
    main()

