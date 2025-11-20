import argparse
import csv
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class ProcessingConfig:
    """Configuration for Dataset_1_NCA_battery ingestion."""

    raw_data_rel_path: str = os.path.join("assets", "raw", "Dataset_1_NCA_battery")
    processed_rel_root: str = os.path.join("assets", "processed")
    chemistry: str = "NCA"
    sample_points: int = 100
    max_cycles: int = 100
    current_tolerance: float = 1e-4
    min_segment_samples: int = 100
    voltage_clip_min: float = 3.0
    voltage_clip_max: float = 3.6
    default_temperature_k: float = 298.15  # 25C assumption
    default_c_rate: float = float("nan")

    def get_raw_data_path(self, project_root: Path) -> Path:
        return Path(project_root, self.raw_data_rel_path)

    def get_processed_dir(self, project_root: Path, battery_id: Optional[str]) -> Path:
        base = Path(project_root, self.processed_rel_root, self.chemistry.lower())
        if battery_id:
            return base / battery_id.lower()
        return base


@dataclass
class ProcessingResult:
    battery_id: str
    charge_df: pd.DataFrame
    discharge_df: pd.DataFrame
    errors: List[str]


COLUMN_ALIASES: Dict[str, Tuple[str, float]] = {
    "time/s": ("Test_Time(s)", 1.0),
    "time(s)": ("Test_Time(s)", 1.0),
    "time_sec": ("Test_Time(s)", 1.0),
    "test_time(s)": ("Test_Time(s)", 1.0),
    "ecell/v": ("Voltage(V)", 1.0),
    "voltage(v)": ("Voltage(V)", 1.0),
    "voltage": ("Voltage(V)", 1.0),
    "<i>/ma": ("Current(A)", 1e-3),
    "current/ma": ("Current(A)", 1e-3),
    "current(ma)": ("Current(A)", 1e-3),
    "current(a)": ("Current(A)", 1.0),
    "cycle number": ("raw_cycle", 1.0),
    "cycle_number": ("raw_cycle", 1.0),
}

OUTPUT_COLUMNS = [
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Parse Dataset_1_NCA_battery raw CSV files into 100-point aggregates."
    )
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=None,
        help="Override path to Dataset_1_NCA_battery directory.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Override processed output root (defaults to assets/processed/<chemistry>/).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process at most N files (useful for smoke tests).",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="count",
        default=0,
        help="Increase logging verbosity.",
    )
    return parser.parse_args()


def discover_raw_files(raw_root: Path) -> List[Path]:
    if not raw_root.exists():
        raise FileNotFoundError(f"Raw directory not found: {raw_root}")
    return sorted(path for path in raw_root.glob("*.csv") if path.is_file())


def safe_component(value: str) -> str:
    slug = "".join(ch if ch.isalnum() else "_" for ch in value.strip())
    slug = slug.strip("_").lower()
    return slug or "unknown_cell"


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    scale_map: Dict[str, float] = {}
    rename_map: Dict[str, str] = {}

    for column in df.columns:
        normalized = column.strip().lower()
        if normalized in COLUMN_ALIASES:
            canonical, scale = COLUMN_ALIASES[normalized]
            rename_map[column] = canonical
            scale_map[canonical] = scale

    df = df.rename(columns=rename_map)

    for canonical, scale in scale_map.items():
        df[canonical] = pd.to_numeric(df[canonical], errors="coerce")
        if canonical == "Current(A)" and not np.isclose(scale, 1.0):
            df[canonical] = df[canonical] * scale
        elif canonical != "Current(A)" and not np.isclose(scale, 1.0):
            df[canonical] = df[canonical] * scale

    required = {"Test_Time(s)", "Voltage(V)", "Current(A)"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns after normalization: {missing}")

    df = df.dropna(subset=list(required)).reset_index(drop=True)
    return df


def assign_cycle_indices(df: pd.DataFrame) -> pd.DataFrame:
    if "raw_cycle" in df.columns:
        raw_cycle = pd.to_numeric(df["raw_cycle"], errors="coerce").fillna(method="ffill")
        unique = sorted({int(val) for val in raw_cycle.dropna().unique()})
        mapping = {value: idx + 1 for idx, value in enumerate(unique)}
        df["cycle_index"] = raw_cycle.map(lambda x: mapping.get(int(x)) if pd.notna(x) else np.nan)
    else:
        df["cycle_index"] = np.nan

    needs_detection = df["cycle_index"].isna().all()
    if needs_detection:
        df["cycle_index"] = detect_cycles_by_current(df["Current(A)"])

    df["cycle_index"] = df["cycle_index"].astype(pd.Int64Dtype())
    df = df.dropna(subset=["cycle_index"])
    return df.reset_index(drop=True)


def detect_cycles_by_current(current_series: pd.Series) -> pd.Series:
    values = np.abs(current_series.to_numpy(dtype=float))
    non_zero = values[values > 0]
    median = float(np.nanmedian(non_zero)) if non_zero.size else 0.0
    tolerance = max(1e-4, 0.05 * median)
    sign = np.sign(current_series.where(np.abs(current_series) >= tolerance, 0.0))

    cycle_ids: List[int] = []
    cycle = 0
    prev_sign = 0
    for value in sign:
        if value == 0:
            cycle_ids.append(cycle if cycle > 0 else np.nan)
            continue
        if prev_sign <= 0 and value > 0:
            cycle += 1
        prev_sign = value
        cycle_ids.append(cycle)
    return pd.Series(cycle_ids, index=current_series.index)


def split_cycle_segments(cycle_df: pd.DataFrame, tolerance: float) -> Tuple[pd.DataFrame, pd.DataFrame]:
    current = cycle_df["Current(A)"].to_numpy()
    positive = np.where(current > tolerance)[0]
    negative = np.where(current < -tolerance)[0]

    charge = pd.DataFrame()
    discharge = pd.DataFrame()
    if positive.size:
        charge = cycle_df.iloc[positive[0] : positive[-1] + 1].copy()
    if negative.size:
        discharge = cycle_df.iloc[negative[0] : negative[-1] + 1].copy()
    return charge, discharge


def prepare_segment(segment: pd.DataFrame, config: ProcessingConfig) -> Optional[pd.DataFrame]:
    if segment.empty:
        return None

    subset = segment[["Test_Time(s)", "Voltage(V)", "Current(A)"]].copy()
    subset.dropna(inplace=True)
    subset.drop_duplicates(subset="Test_Time(s)", inplace=True)
    subset.sort_values("Test_Time(s)", inplace=True)

    if len(subset) < config.min_segment_samples:
        return None

    times = subset["Test_Time(s)"].to_numpy()
    if np.isclose(times[-1] - times[0], 0.0):
        return None

    subset["Voltage(V)"] = subset["Voltage(V)"].clip(
        lower=config.voltage_clip_min, upper=config.voltage_clip_max
    )
    return subset.reset_index(drop=True)


def resample_segment(segment: pd.DataFrame, sample_points: int) -> pd.DataFrame:
    time_values = segment["Test_Time(s)"].to_numpy()
    rel_time = time_values - time_values[0]
    normalized_time = rel_time / rel_time[-1]

    target = np.linspace(0.0, 1.0, sample_points)
    elapsed = np.interp(target, normalized_time, rel_time)
    voltage = np.interp(target, normalized_time, segment["Voltage(V)"].to_numpy())
    current = np.interp(target, normalized_time, segment["Current(A)"].to_numpy())

    return pd.DataFrame(
        {
            "sample_index": np.arange(sample_points, dtype=int),
            "normalized_time": target,
            "elapsed_time_s": elapsed,
            "voltage_v": voltage,
            "current_a": current,
        }
    )


def aggregate_cycles(df: pd.DataFrame, battery_id: str, config: ProcessingConfig) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    errors: List[str] = []
    if df.empty:
        return pd.DataFrame(columns=OUTPUT_COLUMNS), pd.DataFrame(columns=OUTPUT_COLUMNS), ["No data rows after cleaning."]

    df.sort_values(["cycle_index", "Test_Time(s)"], inplace=True)
    unique_cycles = [int(val) for val in df["cycle_index"].dropna().unique()]
    unique_cycles.sort()

    min_samples = max(config.sample_points, config.min_segment_samples)
    collected_charge: List[pd.DataFrame] = []
    collected_discharge: List[pd.DataFrame] = []
    cycle_counter = 0

    for cycle_value in unique_cycles:
        if cycle_counter >= config.max_cycles:
            break

        cycle_df = df[df["cycle_index"] == cycle_value].copy()
        if cycle_df.empty:
            continue

        charge_raw, discharge_raw = split_cycle_segments(cycle_df, config.current_tolerance)

        charge_segment = prepare_segment(charge_raw, config)
        discharge_segment = prepare_segment(discharge_raw, config)
        if charge_segment is None or discharge_segment is None:
            errors.append(f"Cycle {cycle_value} missing qualifying charge/discharge segment.")
            continue

        if len(charge_segment) < min_samples or len(discharge_segment) < min_samples:
            errors.append(f"Cycle {cycle_value} discarded due to insufficient samples.")
            continue

        charge_resampled = resample_segment(charge_segment, config.sample_points)
        discharge_resampled = resample_segment(discharge_segment, config.sample_points)

        if charge_resampled.empty or discharge_resampled.empty:
            errors.append(f"Cycle {cycle_value} produced empty resample.")
            continue

        cycle_counter += 1
        for frame, bucket in (
            (charge_resampled, collected_charge),
            (discharge_resampled, collected_discharge),
        ):
            enriched = frame.copy()
            enriched["battery_id"] = battery_id
            enriched["chemistry"] = config.chemistry
            enriched["cycle_index"] = cycle_counter
            enriched["c_rate"] = config.default_c_rate
            enriched["temperature_k"] = config.default_temperature_k
            bucket.append(enriched[OUTPUT_COLUMNS])

    charge_df = pd.concat(collected_charge, ignore_index=True) if collected_charge else pd.DataFrame(columns=OUTPUT_COLUMNS)
    discharge_df = pd.concat(collected_discharge, ignore_index=True) if collected_discharge else pd.DataFrame(columns=OUTPUT_COLUMNS)
    return charge_df, discharge_df, errors


def write_outputs(result: ProcessingResult, config: ProcessingConfig, output_root: Path, verbose: bool = False) -> None:
    battery_dir = output_root / result.battery_id
    battery_dir.mkdir(parents=True, exist_ok=True)

    charge_path = battery_dir / f"{result.battery_id}_charge_aggregated_data.csv"
    discharge_path = battery_dir / f"{result.battery_id}_discharge_aggregated_data.csv"

    if not result.charge_df.empty:
        result.charge_df.to_csv(charge_path, index=False)
        if verbose:
            print(f"Wrote charge data -> {charge_path}")
    if not result.discharge_df.empty:
        result.discharge_df.to_csv(discharge_path, index=False)
        if verbose:
            print(f"Wrote discharge data -> {discharge_path}")

    if result.errors:
        log_path = battery_dir / f"error_log_{result.battery_id}.csv"
        with log_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["error"])
            for message in result.errors:
                writer.writerow([message])
        if verbose:
            print(f"Wrote error log -> {log_path}")


def process_file(file_path: Path, config: ProcessingConfig, output_root: Path, verbose: bool = False) -> ProcessingResult:
    battery_id = safe_component(file_path.stem)
    try:
        raw_df = pd.read_csv(file_path)
        normalized_df = normalize_columns(raw_df)
        normalized_df["Test_Time(s)"] = pd.to_numeric(normalized_df["Test_Time(s)"], errors="coerce")
        normalized_df["Voltage(V)"] = pd.to_numeric(normalized_df["Voltage(V)"], errors="coerce")
        normalized_df["Current(A)"] = pd.to_numeric(normalized_df["Current(A)"], errors="coerce")
        normalized_df.dropna(subset=["Test_Time(s)", "Voltage(V)", "Current(A)"], inplace=True)
        normalized_df.sort_values("Test_Time(s)", inplace=True)
        tagged_df = assign_cycle_indices(normalized_df)
        charge_df, discharge_df, errors = aggregate_cycles(tagged_df, battery_id, config)
    except Exception as exc:
        charge_df = pd.DataFrame(columns=OUTPUT_COLUMNS)
        discharge_df = pd.DataFrame(columns=OUTPUT_COLUMNS)
        errors = [f"{type(exc).__name__}: {exc}"]

    result = ProcessingResult(battery_id=battery_id, charge_df=charge_df, discharge_df=discharge_df, errors=errors)
    write_outputs(result, config, output_root, verbose=verbose)
    return result


def main() -> int:
    args = parse_args()
    config = ProcessingConfig()
    project_root = Path(__file__).resolve().parents[2]

    raw_root = args.raw_root or config.get_raw_data_path(project_root)
    if args.output_root:
        output_root = args.output_root
    else:
        output_root = config.get_processed_dir(project_root, None)

    output_root.mkdir(parents=True, exist_ok=True)

    files = discover_raw_files(raw_root)
    if args.limit is not None:
        files = files[: args.limit]

    if not files:
        print("No CSV files discovered under Dataset_1_NCA_battery.")
        return 0

    print(f"Discovered {len(files)} file(s) to process.")

    results: List[ProcessingResult] = []
    for path in files:
        if args.verbose:
            print(f"Processing {path.name}")
        results.append(process_file(path, config, output_root, verbose=args.verbose > 0))

    failures = [r for r in results if r.charge_df.empty or r.discharge_df.empty]
    if failures:
        print(f"{len(failures)} file(s) had incomplete outputs. Check their error logs for details.")
        return 1

    print("Completed Dataset_1_NCA_battery parsing successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

