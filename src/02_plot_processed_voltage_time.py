"""Batch-generate Voltage vs Normalized Time plots from processed CSV datasets.

This script mirrors the behavior of src/plot_interim_voltage_soc.py but:
- Reads CSVs from assets/processed instead of assets/interim
- Uses normalized time on the x-axis instead of SOC
"""

from __future__ import annotations

import argparse
import csv
import multiprocessing as mp
import os
import re
import sys
import traceback
from collections import defaultdict
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple, Union

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

DEFAULT_RC_PARAMS = {
    "figure.figsize": (8, 6),
    "figure.dpi": 100,
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


VOLTAGE_ALIASES = {
    "voltage",
    "voltage(v)",
    "voltage_v",
    "cell_voltage",
    "voltage[v]",
    "v",
}

CYCLE_ALIASES = {
    "cycle",
    "cycle_index",
    "cycle_number",
    "cycleid",
    "cycle_no",
}

C_RATE_ALIASES = {
    "c_rate",
    "crate",
    "c_rate_a",
    "c",
}

TEMP_ALIASES = {
    "temperature_k",
    "tempk",
    "temperature",
    "temp",
}

BATTERY_ALIASES = {
    "battery_id",
    "battery",
    "cell",
    "cell_id",
}

CHEMISTRY_ALIASES = {
    "chemistry",
    "chem",
    "chemistry_id",
}

# Prefer already-normalized time if present; otherwise use a raw time-like column and normalize within-cycle
TIME_NORMALIZED_ALIASES = {
    "normalized_time",
    "time_normalized",
    "normalized_time_fraction",
    "norm_time",
    "t_norm",
    "time_frac",
    "time_fraction",
}

TIME_ALIASES = {
    "time_s",
    "time",
    "test_time_s",
    "test_time",
    "timestamp_s",
    "timestamp",
    "elapsed_time_s",
    "elapsed_time",
    "t_s",
    "relative_time_s",
    "relative_time",
    "step_time_s",
    "step_time",
}


@dataclass
class PlotTask:
    csv_path: Path
    processed_root: Path
    output_root: Path
    clipped_output_root: Path


@dataclass
class PlotResult:
    status: str  # "success" or "error"
    csv_path: Path
    chemistry: str
    battery_id: str
    cycle: Optional[str] = None
    phase: Optional[str] = None
    output_path: Optional[Path] = None
    error: Optional[str] = None


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate Voltage vs Normalized Time plots from processed aggregated CSV files."
    )
    parser.add_argument(
        "--chemistry",
        "-c",
        action="append",
        help="Chemistry folder(s) inside assets/processed to process (can repeat).",
    )
    parser.add_argument(
        "--workers",
        "-w",
        type=int,
        default=None,
        help="Number of worker processes (defaults to cpu_count()-1, capped at 8).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process at most N files (useful for testing).",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Directory to store plots; defaults to <repo_root>/assets/images.",
    )
    parser.add_argument(
        "--processed-root",
        type=Path,
        default=None,
        help="Root directory containing processed CSVs; defaults to <repo_root>/assets/processed.",
    )
    parser.add_argument(
        "--error-root",
        type=Path,
        default=None,
        help="Directory for error logs; defaults to <repo_root>/processed_datasets.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Discover files and print planned actions without generating plots.",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="count",
        default=0,
        help="Increase logging verbosity (use -v or -vv).",
    )
    return parser.parse_args(argv)


def resolve_paths(args: argparse.Namespace) -> Tuple[Path, Path, Path]:
    repo_root = Path(__file__).resolve().parent.parent
    processed_root = args.processed_root or repo_root / "assets" / "processed"
    output_root = args.output_root or repo_root / "assets" / "images"
    error_root = args.error_root or repo_root / "processed_datasets"
    return processed_root.resolve(), output_root.resolve(), error_root.resolve()


def derive_clipped_output_root(output_root: Path) -> Path:
    return output_root.parent / f"{output_root.name}_clipped"


def normalize_column_name(name: str) -> str:
    cleaned = re.sub(r"[^a-z0-9]", "", name.lower())
    return cleaned


def pick_column(df: pd.DataFrame, aliases: Iterable[str], logical_name: str) -> str:
    normalized = {normalize_column_name(col): col for col in df.columns}
    for alias in aliases:
        key = normalize_column_name(alias)
        if key in normalized:
            return normalized[key]
    raise KeyError(f"Could not find a column for '{logical_name}' in {list(df.columns)}")


def pick_optional_column(df: pd.DataFrame, aliases: Iterable[str]) -> Optional[str]:
    try:
        return pick_column(df, aliases, logical_name="optional")
    except KeyError:
        return None


def safe_component(value: str) -> str:
    sanitized = re.sub(r"[^a-zA-Z0-9_-]", "_", value)
    return sanitized or "unknown"


def parse_metadata(csv_path: Path, processed_root: Path) -> Tuple[str, str, str, str, str, str]:
    rel_parts = csv_path.relative_to(processed_root).parts
    chemistry = rel_parts[0] if len(rel_parts) >= 1 else "unknown"
    battery_id = rel_parts[1] if len(rel_parts) >= 2 else csv_path.stem

    stem = csv_path.stem.lower()
    if "discharge" in stem:
        phase = "discharge"
    elif "charge" in stem:
        phase = "charge"
    else:
        phase = "unknown"

    cycle_match = re.search(r"cycle[_-]?(\d+)", stem)
    cycle = cycle_match.group(1) if cycle_match else "unknown"

    crate_match = re.search(r"crate[_-]?([0-9]+(?:\.[0-9]+)?)", stem)
    crate = crate_match.group(1) if crate_match else "unknown"

    temp_match = re.search(r"temp(?:k|_)?([0-9]+)", stem)
    if temp_match:
        temp_value = temp_match.group(1)
        temp = temp_value
    else:
        temp_c_match = re.search(r"([0-9]+)c", stem)
        if temp_c_match:
            temp_c = int(temp_c_match.group(1))
            temp = str(temp_c + 273)
        else:
            temp = "unknown"

    chemistry = safe_component(chemistry)
    battery_id = safe_component(battery_id)
    phase = safe_component(phase)
    cycle = safe_component(cycle)
    crate = safe_component(crate)
    temp = safe_component(temp)

    return chemistry, battery_id, phase, cycle, crate, temp


def build_output_filename(meta: Tuple[str, str, str, str, str, str]) -> str:
    chemistry, battery_id, phase, cycle, crate, temp = meta
    return f"Cycle_{cycle}_{phase}_Crate_{crate}_tempK_{temp}_batteryID_{battery_id}.png"


def prepare_time_voltage_data(
    df: pd.DataFrame,
    voltage_col: str,
) -> pd.DataFrame:
    # Prefer normalized time if present
    norm_col = pick_optional_column(df, TIME_NORMALIZED_ALIASES)
    if norm_col:
        t_series = pd.to_numeric(df[norm_col], errors="coerce")
        # If values look like percentages (0-100), scale to 0-1
        if t_series.max(skipna=True) and float(t_series.max()) > 1.01:
            t_series = t_series / float(t_series.max())
    else:
        time_col = pick_column(df, TIME_ALIASES, "time")
        raw_t = pd.to_numeric(df[time_col], errors="coerce")
        t_min = raw_t.min(skipna=True)
        t_max = raw_t.max(skipna=True)
        if pd.isna(t_min) or pd.isna(t_max) or t_max == t_min:
            raise ValueError("Cannot normalize time: invalid or constant time values.")
        t_series = (raw_t - t_min) / (t_max - t_min)

    voltage_series = pd.to_numeric(df[voltage_col], errors="coerce")
    data = pd.DataFrame({"t": t_series, "voltage": voltage_series}).dropna()
    if data.empty:
        raise ValueError("No valid time/voltage pairs after cleaning.")
    data = data.sort_values("t")
    return data


def plot_time_voltage(
    data: pd.DataFrame,
    output_path: Path,
    title: str,
    verbose: bool = False,
    y_limits: Tuple[float, float] = (2.0, 4.3),
) -> None:
    fig, ax = plt.subplots()
    ax.plot(data["t"], data["voltage"], color="#1f77b4")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(*y_limits)
    ax.set_xlabel("Normalized Time (fraction)")
    ax.set_ylabel("Voltage (V)")
    ax.set_title(title)
    xticks = np.linspace(0.0, 1.0, 11)
    ax.set_xticks(xticks)
    yticks = np.round(np.linspace(y_limits[0], y_limits[1], 12), 3)
    ax.set_yticks(yticks)
    ax.set_axisbelow(True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if verbose:
        print(f"Saving plot -> {output_path}")
    fig.savefig(output_path)
    plt.close(fig)


def format_cycle_value(value: object) -> str:
    try:
        numeric = int(float(value))
        return f"{numeric:03d}"
    except (TypeError, ValueError):
        return safe_component(str(value))


def cycle_sort_key(value: object) -> Tuple[int, Union[float, str]]:
    try:
        numeric = float(value)
        return (0, numeric)
    except (TypeError, ValueError):
        return (1, str(value))


def format_numeric_component(value: Optional[float], default: str = "unknown") -> str:
    if value is None:
        return default
    try:
        rounded = round(float(value), 4)
        text = f"{rounded:.4f}".rstrip("0").rstrip(".")
        return safe_component(text)
    except (TypeError, ValueError):
        return default


def extract_group_value(df: pd.DataFrame, aliases: Iterable[str]) -> Optional[float]:
    column = pick_optional_column(df, aliases)
    if not column:
        return None
    series = pd.to_numeric(df[column], errors="coerce").dropna()
    if series.empty:
        return None
    return float(series.median())


def discover_csvs(processed_root: Path, chemistries: Optional[Sequence[str]]) -> List[Path]:
    if not processed_root.exists():
        raise FileNotFoundError(f"Processed root not found: {processed_root}")

    if chemistries:
        candidates = []
        for chem in chemistries:
            chem_dir = processed_root / chem
            if not chem_dir.exists():
                raise FileNotFoundError(f"Chemistry folder not found: {chem_dir}")
            candidates.extend(sorted(chem_dir.rglob("*_aggregated_data.csv")))
        return candidates

    return sorted(processed_root.rglob("*_aggregated_data.csv"))


def process_csv(task: PlotTask, verbose: bool = False) -> List[PlotResult]:
    try:
        meta = parse_metadata(task.csv_path, task.processed_root)
    except Exception as exc:  # pragma: no cover - defensive
        return [
            PlotResult(
                status="error",
                csv_path=task.csv_path,
                chemistry="unknown",
                battery_id="unknown",
                error=f"Metadata parsing failed: {exc}",
            )
        ]

    chemistry, battery_id, phase, _, _, _ = meta

    try:
        df = pd.read_csv(task.csv_path)
        voltage_col = pick_column(df, VOLTAGE_ALIASES, "voltage")
        cycle_col = pick_column(df, CYCLE_ALIASES, "cycle")

        chemistry_col = pick_optional_column(df, CHEMISTRY_ALIASES)
        if chemistry_col:
            chemistry_value = df[chemistry_col].dropna()
            if not chemistry_value.empty:
                chemistry = safe_component(str(chemistry_value.iloc[0]))

        battery_col = pick_optional_column(df, BATTERY_ALIASES)
        if battery_col:
            battery_value = df[battery_col].dropna()
            if not battery_value.empty:
                battery_id = safe_component(str(battery_value.iloc[0]))

        unique_cycles = [c for c in pd.unique(df[cycle_col]) if pd.notna(c)]
        if not unique_cycles:
            raise ValueError("No cycle values found in dataset.")

        base_c_rate = extract_group_value(df, C_RATE_ALIASES)
        base_temp = extract_group_value(df, TEMP_ALIASES)

        results: List[PlotResult] = []
        for cycle_value in sorted(unique_cycles, key=cycle_sort_key):
            cycle_df = df[df[cycle_col] == cycle_value]
            try:
                data = prepare_time_voltage_data(cycle_df, voltage_col)
            except Exception as inner_exc:
                results.append(
                    PlotResult(
                        status="error",
                        csv_path=task.csv_path,
                        chemistry=chemistry,
                        battery_id=battery_id,
                        cycle=str(cycle_value),
                        phase=phase,
                        error=f"Cycle {cycle_value}: {inner_exc}",
                    )
                )
                continue

            cycle_component = format_cycle_value(cycle_value)
            cycle_c_rate = extract_group_value(cycle_df, C_RATE_ALIASES)
            cycle_temp = extract_group_value(cycle_df, TEMP_ALIASES)

            crate_component = format_numeric_component(cycle_c_rate if cycle_c_rate is not None else base_c_rate)
            temp_component = format_numeric_component(cycle_temp if cycle_temp is not None else base_temp)

            output_name = build_output_filename(
                (
                    chemistry,
                    battery_id,
                    phase,
                    cycle_component,
                    crate_component,
                    temp_component,
                )
            )
            output_path = task.output_root / chemistry / battery_id / output_name
            clipped_output_path = task.clipped_output_root / chemistry / battery_id / output_name

            title_parts = [part for part in [chemistry, battery_id, f"Cycle {cycle_component}", phase] if part != "unknown"]
            title = " - ".join(title_parts)

            plot_time_voltage(data, output_path, title, verbose=verbose)
            plot_time_voltage(
                data,
                clipped_output_path,
                title,
                verbose=verbose,
                y_limits=(3.0, 3.6),
            )

            results.append(
                PlotResult(
                    status="success",
                    csv_path=task.csv_path,
                    chemistry=chemistry,
                    battery_id=battery_id,
                    cycle=cycle_component,
                    phase=phase,
                    output_path=output_path,
                )
            )

        return results
    except Exception as exc:  # pragma: no cover - defensive
        message = f"{exc.__class__.__name__}: {exc}"
        if verbose:
            traceback.print_exc()
        return [
            PlotResult(
                status="error",
                csv_path=task.csv_path,
                chemistry=chemistry,
                battery_id=battery_id,
                phase=phase,
                error=message,
            )
        ]


def determine_worker_count(requested: Optional[int]) -> int:
    if requested is not None:
        return max(1, requested)
    cpu_total = os.cpu_count() or 2
    return max(1, min(cpu_total - 1, 8))


def write_error_logs(
    results: Sequence[PlotResult],
    error_root: Path,
    verbose: bool = False,
) -> List[Path]:
    grouped: defaultdict[Tuple[str, str], List[PlotResult]] = defaultdict(list)
    for result in results:
        if result.status == "error" and result.error:
            grouped[(result.chemistry, result.battery_id)].append(result)

    written_logs: List[Path] = []
    for (chemistry, battery_id), entries in grouped.items():
        chem_dir = error_root / chemistry
        chem_dir.mkdir(parents=True, exist_ok=True)
        log_path = chem_dir / f"error_log_{battery_id}.csv"
        if verbose:
            print(f"Writing error log -> {log_path}")
        with log_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["file", "cycle", "phase", "error"])
            writer.writeheader()
            for entry in entries:
                writer.writerow({
                    "file": str(entry.csv_path),
                    "cycle": entry.cycle or "",
                    "phase": entry.phase or "",
                    "error": entry.error or "",
                })
        written_logs.append(log_path)

    return written_logs


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    processed_root, output_root, error_root = resolve_paths(args)

    chemistries = args.chemistry or []
    csv_files = discover_csvs(processed_root, chemistries)
    if args.limit is not None:
        csv_files = csv_files[: args.limit]

    if not csv_files:
        print("No aggregated CSV files found with the provided filters.")
        return 0

    print(f"Discovered {len(csv_files)} CSV file(s) to process.")

    if args.dry_run:
        for path in csv_files:
            print(path)
        return 0

    worker_count = determine_worker_count(args.workers)
    print(f"Using {worker_count} worker process(es).")

    clipped_output_root = derive_clipped_output_root(output_root)
    tasks = [
        PlotTask(
            csv_path=path,
            processed_root=processed_root,
            output_root=output_root,
            clipped_output_root=clipped_output_root,
        )
        for path in csv_files
    ]

    process_func = partial(process_csv, verbose=args.verbose > 0)

    results: List[PlotResult] = []
    with mp.Pool(processes=worker_count) as pool:
        for batch in pool.imap_unordered(process_func, tasks):
            results.extend(batch)
            if args.verbose:
                for result in batch:
                    status = "OK" if result.status == "success" else "FAIL"
                    message = result.output_path if result.status == "success" else result.error
                    print(f"[{status}] {result.csv_path} -> {message}")

    successes = [r for r in results if r.status == "success"]
    failures = [r for r in results if r.status == "error"]

    print(f"Completed with {len(successes)} success(es) and {len(failures)} failure(s).")

    if failures:
        log_paths = write_error_logs(failures, error_root, verbose=args.verbose > 0)
        print("Failure details written to:")
        for log in log_paths:
            print(f"  {log}")

    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
# Completed with 82534 success(es) and 10 failure(s).

