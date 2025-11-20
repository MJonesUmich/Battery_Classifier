"""Generate voltage vs time plots from processed aggregation CSVs.

The plots are intended for downstream computer-vision experiments, so every
cycle is rendered as a voltage trace with a fixed 3.0–3.6 V y-range.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import re
import sys
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

DEFAULT_RC_PARAMS = {
    "figure.figsize": (8, 6),
    "figure.dpi": 120,
    "figure.facecolor": "#ffffff",
    "axes.facecolor": "#ffffff",
    "axes.edgecolor": "#000000",
    "axes.grid": True,
    "grid.color": "#d3d3d3",
    "grid.linestyle": "-",
    "grid.linewidth": 0.8,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "lines.linewidth": 2.0,
    "lines.antialiased": True,
    "savefig.facecolor": "#ffffff",
    "savefig.bbox": "tight",
}

plt.rcParams.update(DEFAULT_RC_PARAMS)

REQUIRED_COLUMNS = {
    "battery_id",
    "chemistry",
    "cycle_index",
    "sample_index",
    "normalized_time",
    "elapsed_time_s",
    "voltage_v",
    "current_a",
}


@dataclass
class PlotTask:
    csv_path: Path
    processed_root: Path
    output_root: Path


@dataclass
class PlotResult:
    status: str  # "success" or "error"
    csv_path: Path
    chemistry: str
    battery_id: str
    error: Optional[str] = None


REPO_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_ROOT = (REPO_ROOT / "assets" / "processed").resolve()
OUTPUT_ROOT = (REPO_ROOT / "assets" / "images" / "processed_voltage").resolve()
CHEMISTRY_FILTER: Optional[Sequence[str]] = None  # set to ("LCO",) to narrow runs
FILE_LIMIT: Optional[int] = None  # set to small int for smoke tests
WORKER_COUNT: Optional[int] = None  # default auto = cpu_count()-1 capped at 8
VERBOSE = False


def discover_csvs(processed_root: Path, chemistries: Optional[Iterable[str]]) -> List[Path]:
    if not processed_root.exists():
        raise FileNotFoundError(f"Processed root not found: {processed_root}")

    candidates: List[Path] = []
    if chemistries:
        for chem in chemistries:
            chem_dir = processed_root / chem
            if not chem_dir.exists():
                raise FileNotFoundError(f"Chemistry folder not found: {chem_dir}")
            candidates.extend(
                sorted(chem_dir.rglob("*_aggregated_data.csv"))
            )
    else:
        candidates = sorted(processed_root.rglob("*_aggregated_data.csv"))

    return [path for path in candidates if path.is_file()]


def safe_component(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9_-]", "_", value)
    return slug or "unknown"


def parse_metadata(csv_path: Path, processed_root: Path) -> Tuple[str, str, str]:
    rel_parts = csv_path.relative_to(processed_root).parts
    chemistry = safe_component(rel_parts[0]) if len(rel_parts) >= 1 else "unknown"
    battery_id = safe_component(rel_parts[1]) if len(rel_parts) >= 2 else safe_component(csv_path.stem)

    stem = csv_path.stem.lower()
    if "discharge" in stem:
        phase = "discharge"
    elif "charge" in stem:
        phase = "charge"
    else:
        phase = "unknown"

    return chemistry, battery_id, phase


def prepare_cycle(df: pd.DataFrame, cycle_index: int) -> pd.DataFrame:
    cycle_df = df[df["cycle_index"] == cycle_index].copy()
    cycle_df.sort_values("sample_index", inplace=True)
    cycle_df = cycle_df.dropna(subset=["normalized_time", "voltage_v"])
    if cycle_df.empty:
        return cycle_df
    # Ensure normalized time stays within [0, 1]
    cycle_df["normalized_time"] = cycle_df["normalized_time"].clip(0.0, 1.0)
    return cycle_df


def plot_voltage_trace(
    data: pd.DataFrame,
    output_path: Path,
    title: str,
    verbose: bool = False,
) -> None:
    fig, ax = plt.subplots()
    ax.plot(data["normalized_time"], data["voltage_v"], color="#1f77b4")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(2.9, 3.7)
    ax.set_xlabel("Normalized Time")
    ax.set_ylabel("Voltage (V)")
    ax.set_title(title)
    xticks = np.linspace(0.0, 1.0, 11)
    ax.set_xticks(xticks)
    yticks = np.round(np.linspace(2.9, 3.7, 9), 3)
    ax.set_yticks(yticks)
    ax.set_axisbelow(True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if verbose:
        print(f"Saving plot -> {output_path}")
    fig.savefig(output_path)
    plt.close(fig)


def build_output_path(
    output_root: Path,
    chemistry: str,
    battery_id: str,
    phase: str,
    cycle_index: int,
) -> Path:
    cycle_component = f"{cycle_index:03d}"
    filename = f"cycle_{cycle_component}_{phase}_battery_{battery_id}.png"
    return output_root / chemistry / battery_id / phase / filename


def process_csv(
    task: PlotTask,
    verbose: bool = False,
) -> PlotResult:
    chemistry, battery_id, phase = parse_metadata(task.csv_path, task.processed_root)
    try:
        df = pd.read_csv(task.csv_path)
    except Exception as exc:  # pragma: no cover - defensive
        return PlotResult(
            status="error",
            csv_path=task.csv_path,
            chemistry=chemistry,
            battery_id=battery_id,
            error=f"Failed to read CSV: {exc}",
        )

    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        return PlotResult(
            status="error",
            csv_path=task.csv_path,
            chemistry=chemistry,
            battery_id=battery_id,
            error=f"Missing columns: {sorted(missing)}",
        )

    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["voltage_v", "normalized_time"])

    cycle_values = sorted({int(idx) for idx in df["cycle_index"].dropna().unique()})
    if not cycle_values:
        return PlotResult(
            status="error",
            csv_path=task.csv_path,
            chemistry=chemistry,
            battery_id=battery_id,
            error="No cycle_index values found.",
        )

    for cycle_index in cycle_values:
        cycle_df = prepare_cycle(df, cycle_index)
        if cycle_df.empty:
            if verbose:
                print(f"Skipping empty cycle {cycle_index} in {task.csv_path}")
            continue
        output_path = build_output_path(task.output_root, chemistry, battery_id, phase, cycle_index)
        title = f"{chemistry} | {battery_id} | {phase} | cycle {cycle_index:03d}"
        plot_voltage_trace(cycle_df, output_path, title, verbose=verbose)

    return PlotResult(
        status="success",
        csv_path=task.csv_path,
        chemistry=chemistry,
        battery_id=battery_id,
    )


def determine_worker_count(requested: Optional[int]) -> int:
    if requested is not None:
        return max(1, requested)
    cpu_total = os.cpu_count() or 2
    return max(1, min(cpu_total - 1, 8))


def main() -> int:
    processed_root = PROCESSED_ROOT
    output_root = OUTPUT_ROOT

    csv_files = discover_csvs(processed_root, CHEMISTRY_FILTER)
    if FILE_LIMIT is not None:
        csv_files = csv_files[: FILE_LIMIT]

    if not csv_files:
        print("No processed CSV files found.")
        return 0

    print(f"Discovered {len(csv_files)} CSV file(s) to process.")

    worker_count = determine_worker_count(WORKER_COUNT)
    print(f"Using {worker_count} worker process(es).")

    tasks = [
        PlotTask(csv_path=path, processed_root=processed_root, output_root=output_root)
        for path in csv_files
    ]

    process_func = partial(process_csv, verbose=VERBOSE)

    results: List[PlotResult] = []
    with mp.Pool(processes=worker_count) as pool:
        for result in pool.imap_unordered(process_func, tasks):
            results.append(result)
            if VERBOSE and result.status == "error":
                print(f"[FAIL] {result.csv_path} -> {result.error}")

    successes = [r for r in results if r.status == "success"]
    failures = [r for r in results if r.status == "error"]

    print(f"Completed with {len(successes)} success(es) and {len(failures)} failure(s).")
    if failures:
        print("Failure details:")
        for entry in failures:
            print(f"  {entry.csv_path}: {entry.error}")

    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())

