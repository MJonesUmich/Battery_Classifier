import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd


def _ensure_import_path() -> None:
    """Ensure the src directory is available on sys.path for direct execution."""

    src_dir = Path(__file__).resolve().parents[1]
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))


_ensure_import_path()

try:
    from utils.help_function import load_meta_properties
    from utils.voltage_clamp import clamp_voltage_column
except ModuleNotFoundError:  # pragma: no cover - defensive fallback
    project_root = Path(__file__).resolve().parents[2]
    src_package = project_root / "src"
    if str(src_package) not in sys.path:
        sys.path.insert(0, str(src_package))
    from utils.help_function import load_meta_properties
    from utils.voltage_clamp import clamp_voltage_column


@dataclass
class ProcessingConfig:
    """Configuration for the Stanford parser pipeline."""

    raw_data_rel_path: str = os.path.join("assets", "raw", "Stanford")
    processed_rel_root: str = os.path.join("assets", "processed")
    sample_points: int = 100
    current_tolerance: float = 1e-4
    metadata_sheet: str = "General_Infos"
    max_cycles: int = 100

    def get_raw_base_path(self, project_root: str) -> str:
        return os.path.join(project_root, self.raw_data_rel_path)

    def get_processed_dir(
        self, project_root: str, chemistry: str, battery_key: Optional[str] = None
    ) -> str:
        base = os.path.join(project_root, self.processed_rel_root, chemistry)
        if battery_key:
            return os.path.join(base, battery_key)
        return base


@dataclass
class CellMetadata:
    """Metadata describing a Stanford cell configuration."""

    initial_capacity: float
    c_rate: float
    temperature_k: float
    vmax: float
    vmin: float
    chemistry: str


MIN_SEGMENT_SAMPLE_COUNT = 100


@dataclass
class StanfordFileInfo:
    """Descriptor for a single Stanford source file."""

    path: str
    file_name: str
    battery_id: str
    c_rate: float
    temperature_k: float
    chemistry: str


def extract_battery_info_from_filename(file_name: str) -> Tuple[str, float, float]:
    """Extract battery identifier, C-rate, and temperature from a filename."""

    base_name = os.path.splitext(file_name)[0]
    parts = base_name.split("_")
    if len(parts) < 4:
        raise ValueError(f"Unexpected Stanford filename format: {file_name}")

    battery_id = f"{parts[0]}_{parts[1]}"

    c_rate = 1.0
    if len(parts) >= 4 and parts[3].endswith("C") and "deg" not in parts[3]:
        c_rate = float(f"{parts[2]}.{parts[3][:-1]}")
    elif len(parts) >= 3 and parts[2].endswith("C"):
        c_rate = float(parts[2][:-1])

    temp_token = parts[-1]
    temp_c = None
    if "degC" in temp_token:
        try:
            temp_c = float(temp_token.split("degC")[0])
        except ValueError as exc:
            raise ValueError(f"Cannot parse temperature from {file_name}") from exc

    if temp_c is None:
        temp_c = 25.0

    temp_k = temp_c + 273.15
    return battery_id, c_rate, temp_k


def normalize_battery_key(battery_id: str, temperature_k: float) -> str:
    """Create a stable, lowercase identifier for output artifacts."""

    rounded_temp = int(round(temperature_k))
    return f"{battery_id.lower()}_{rounded_temp}k"


def load_stanford_excel(file_path: str) -> pd.DataFrame:
    """Load an Excel file, selecting a sheet that contains the required columns."""

    xls = pd.ExcelFile(file_path)
    required = {"Current(A)", "Voltage(V)", "Test_Time(s)"}
    chosen = None

    for sheet in xls.sheet_names:
        try:
            header = pd.read_excel(file_path, sheet_name=sheet, nrows=1)
        except Exception:
            continue
        columns = {str(col).strip() for col in header.columns}
        if required <= columns:
            chosen = sheet
            break

    if chosen is None:
        chosen = xls.sheet_names[0]

    df = pd.read_excel(file_path, sheet_name=chosen)
    df.columns = [str(col).strip() for col in df.columns]
    return df


def load_stanford_text(file_path: str) -> pd.DataFrame:
    """Load a legacy tab-delimited export and convert units to SI."""

    df = pd.read_csv(file_path, delimiter="\t")
    df.rename(
        columns={"Time": "Test_Time(s)", "mA": "Current(A)", "mV": "Voltage(V)"},
        inplace=True,
    )
    required = ["Current(A)", "Voltage(V)", "Test_Time(s)"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns {missing} in {file_path}")

    df = df[required]
    df["Voltage(V)"] = pd.to_numeric(df["Voltage(V)"], errors="coerce") / 1000.0
    df["Current(A)"] = pd.to_numeric(df["Current(A)"], errors="coerce") / 1000.0
    df["Test_Time(s)"] = pd.to_numeric(df["Test_Time(s)"], errors="coerce") * 60.0
    df.dropna(subset=required, inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def load_source_file(file_path: str) -> pd.DataFrame:
    """Load a Stanford source file regardless of its extension."""

    df = (
        load_stanford_text(file_path)
        if file_path.lower().endswith(".txt")
        else load_stanford_excel(file_path)
    )

    required = ["Current(A)", "Voltage(V)", "Test_Time(s)"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Required columns {missing} not found in {file_path}")

    df = df[required].apply(pd.to_numeric, errors="coerce")
    df.dropna(subset=required, inplace=True)
    df.sort_values("Test_Time(s)", inplace=True)
    df.drop_duplicates(subset="Test_Time(s)", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def compute_current_threshold(current: pd.Series, fallback: float = 1e-4) -> float:
    """Compute a tolerance-aware threshold to distinguish charge/discharge."""

    non_zero = np.abs(current[current != 0])
    if len(non_zero) == 0:
        return fallback
    return max(0.05 * np.nanmedian(non_zero), fallback)


def detect_charge_discharge_cycles(
    df: pd.DataFrame, tolerance: float
) -> List[Tuple[int, int, int, int]]:
    """Return cycle boundaries as (charge_start, charge_end, discharge_start, discharge_end)."""

    current = df["Current(A)"].to_numpy()
    sign = np.zeros_like(current, dtype=int)
    sign[current > tolerance] = 1
    sign[current < -tolerance] = -1

    cycles: List[Tuple[int, int, int, int]] = []
    idx = 0
    n = len(sign)

    while idx < n:
        while idx < n and sign[idx] != 1:
            idx += 1
        if idx >= n:
            break
        charge_start = idx
        while idx < n and sign[idx] == 1:
            idx += 1
        charge_end = idx

        while idx < n and sign[idx] == 0:
            idx += 1
        if idx >= n or sign[idx] != -1:
            break

        discharge_start = idx
        while idx < n and sign[idx] == -1:
            idx += 1
        discharge_end = idx

        cycles.append((charge_start, charge_end, discharge_start, discharge_end))

    return cycles


def extract_cycle_frames(df: pd.DataFrame, tolerance: float) -> List[pd.DataFrame]:
    """Extract individual charge/discharge cycles from the raw data."""

    cycles = detect_charge_discharge_cycles(df, tolerance)
    frames: List[pd.DataFrame] = []

    for charge_start, charge_end, discharge_start, discharge_end in cycles:
        start = charge_start
        end = discharge_end
        if end - start <= 1:
            continue
        frames.append(df.iloc[start:end].reset_index(drop=True))

    return frames


def restrict_voltage_range(
    df: pd.DataFrame,
    vmin: Optional[float],
    vmax: Optional[float],
    tolerance: float = 0.01,
) -> pd.DataFrame:
    """Clip the dataframe to the provided voltage window when metadata is available."""

    if vmin is None or vmax is None:
        return df

    mask = (df["Voltage(V)"] >= vmin - tolerance) & (
        df["Voltage(V)"] <= vmax + tolerance
    )
    filtered = df[mask]
    if filtered.empty:
        return df
    return filtered.reset_index(drop=True)


def resample_cycle_segment(
    segment_df: pd.DataFrame, sample_points: int
) -> pd.DataFrame:
    """Resample a segment to a fixed number of points with interpolation."""

    required = ["Test_Time(s)", "Voltage(V)", "Current(A)"]
    if segment_df is None or segment_df.empty:
        return pd.DataFrame()

    segment = segment_df[required].copy()
    segment.dropna(inplace=True)
    segment.drop_duplicates(subset="Test_Time(s)", inplace=True)
    segment.sort_values("Test_Time(s)", inplace=True)

    if segment.empty:
        return pd.DataFrame()

    time_values = segment["Test_Time(s)"].to_numpy()
    rel_time = time_values - time_values[0]

    result = pd.DataFrame(
        {
            "sample_index": np.arange(sample_points, dtype=int),
            "normalized_time": np.linspace(0.0, 1.0, sample_points),
        }
    )

    if len(segment) == 1 or np.isclose(rel_time[-1], 0.0):
        result["elapsed_time_s"] = np.zeros(sample_points)
        for column, target in [
            ("Voltage(V)", "voltage_v"),
            ("Current(A)", "current_a"),
        ]:
            result[target] = np.full(sample_points, segment[column].iloc[0])
        return result

    normalized_time = rel_time / rel_time[-1]
    target_time = result["normalized_time"].to_numpy()

    result["elapsed_time_s"] = np.interp(target_time, normalized_time, rel_time)
    result["voltage_v"] = np.interp(
        target_time, normalized_time, segment["Voltage(V)"].to_numpy()
    )
    result["current_a"] = np.interp(
        target_time, normalized_time, segment["Current(A)"].to_numpy()
    )
    return result


def split_cycle_segments(
    cycle_df: pd.DataFrame, tolerance: float
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split a cycle dataframe into charge and discharge segments."""

    if cycle_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    current = cycle_df["Current(A)"].to_numpy()
    positive_indices = np.where(current > tolerance)[0]
    negative_indices = np.where(current < -tolerance)[0]

    charge_segment = pd.DataFrame()
    discharge_segment = pd.DataFrame()

    if positive_indices.size:
        charge_start = positive_indices[0]
        charge_end = positive_indices[-1] + 1
        charge_segment = cycle_df.iloc[charge_start:charge_end].copy()

    if negative_indices.size:
        discharge_start = negative_indices[0]
        discharge_end = negative_indices[-1] + 1
        discharge_segment = cycle_df.iloc[discharge_start:discharge_end].copy()

    return charge_segment, discharge_segment


def prepare_cycle_segment(
    segment_df: pd.DataFrame, min_samples: int
) -> Optional[pd.DataFrame]:
    """Sanitize a raw segment and enforce minimum sample requirements."""

    required = ["Test_Time(s)", "Voltage(V)", "Current(A)"]

    if segment_df is None or segment_df.empty:
        return None

    if any(column not in segment_df.columns for column in required):
        return None

    sanitized = segment_df.copy()

    for column in required:
        sanitized[column] = pd.to_numeric(sanitized[column], errors="coerce")

    sanitized.dropna(subset=required, inplace=True)
    if sanitized.empty:
        return None

    sanitized.drop_duplicates(subset="Test_Time(s)", inplace=True)
    sanitized.sort_values("Test_Time(s)", inplace=True)

    if len(sanitized) < min_samples:
        return None

    times = sanitized["Test_Time(s)"].to_numpy()
    if np.isclose(times[-1] - times[0], 0.0):
        return None

    sanitized = sanitized.reset_index(drop=True)
    clamp_voltage_column(sanitized, column="Voltage(V)")
    return sanitized


def prepare_resampled_outputs(
    agg_df: pd.DataFrame,
    chemistry: str,
    battery_key: str,
    config: ProcessingConfig,
    default_temperature: float,
    default_c_rate: float,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Create resampled charge and discharge outputs per specification."""

    columns = [
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

    if agg_df.empty:
        empty = pd.DataFrame(columns=columns)
        return empty.copy(), empty.copy()

    charge_segments: List[pd.DataFrame] = []
    discharge_segments: List[pd.DataFrame] = []

    unique_cycles = sorted(int(idx) for idx in agg_df["cycle_index"].dropna().unique())
    min_samples_required = max(config.sample_points, MIN_SEGMENT_SAMPLE_COUNT)
    max_cycles = max(1, int(config.max_cycles))
    valid_cycle_count = 0

    for cycle_index in unique_cycles:
        if valid_cycle_count >= max_cycles:
            break

        cycle_df = agg_df[agg_df["cycle_index"] == cycle_index].copy()
        if cycle_df.empty:
            continue

        charge_segment_raw, discharge_segment_raw = split_cycle_segments(
            cycle_df, tolerance=config.current_tolerance
        )

        charge_segment = prepare_cycle_segment(
            charge_segment_raw, min_samples_required
        )
        discharge_segment = prepare_cycle_segment(
            discharge_segment_raw, min_samples_required
        )

        if charge_segment is None or discharge_segment is None:
            continue

        charge_resampled = resample_cycle_segment(
            charge_segment, config.sample_points
        )
        discharge_resampled = resample_cycle_segment(
            discharge_segment, config.sample_points
        )

        if charge_resampled.empty or discharge_resampled.empty:
            continue

        valid_cycle_count += 1

        for label, segment, resampled in (
            ("charge", charge_segment, charge_resampled),
            ("discharge", discharge_segment, discharge_resampled),
        ):
            resampled = resampled.copy()
            resampled["battery_id"] = battery_key
            resampled["chemistry"] = chemistry
            resampled["cycle_index"] = valid_cycle_count

            if "c_rate" in segment.columns:
                c_rate_series = segment["c_rate"].dropna()
                c_rate_value = (
                    float(c_rate_series.iloc[0])
                    if not c_rate_series.empty
                    else default_c_rate
                )
            else:
                c_rate_value = default_c_rate

            if "temperature_k" in segment.columns:
                temp_series = segment["temperature_k"].dropna()
                temperature_value = (
                    float(temp_series.iloc[0])
                    if not temp_series.empty
                    else default_temperature
                )
            else:
                temperature_value = default_temperature

            resampled["c_rate"] = c_rate_value
            resampled["temperature_k"] = temperature_value
            resampled = resampled[columns]

            if label == "charge":
                charge_segments.append(resampled)
            else:
                discharge_segments.append(resampled)

    empty = pd.DataFrame(columns=columns)

    if charge_segments:
        charge_df = pd.concat(charge_segments, ignore_index=True)
        charge_df = charge_df.sort_values(["cycle_index", "sample_index"])
        charge_df = charge_df.reset_index(drop=True)
    else:
        charge_df = empty.copy()

    if discharge_segments:
        discharge_df = pd.concat(discharge_segments, ignore_index=True)
        discharge_df = discharge_df.sort_values(["cycle_index", "sample_index"])
        discharge_df = discharge_df.reset_index(drop=True)
    else:
        discharge_df = empty.copy()

    return charge_df, discharge_df


def save_processed_data(
    charge_df: pd.DataFrame,
    discharge_df: pd.DataFrame,
    output_dir: str,
    battery_key: str,
) -> Tuple[str, str]:
    """Persist charge and discharge aggregates to CSV files."""

    os.makedirs(output_dir, exist_ok=True)

    charge_path = os.path.join(output_dir, f"{battery_key}_charge_aggregated_data.csv")
    discharge_path = os.path.join(
        output_dir, f"{battery_key}_discharge_aggregated_data.csv"
    )

    charge_df.to_csv(charge_path, index=False)
    discharge_df.to_csv(discharge_path, index=False)

    return charge_path, discharge_path


def save_error_log(
    error_dict: Dict[str, str], output_dir: str, battery_key: str
) -> None:
    """Persist parsing errors for a specific battery group."""

    if not error_dict:
        return

    os.makedirs(output_dir, exist_ok=True)
    error_df = pd.DataFrame(
        list(error_dict.items()), columns=["file_name", "error_message"]
    )
    error_path = os.path.join(output_dir, f"error_log_{battery_key}.csv")
    error_df.to_csv(error_path, index=False)


def find_mapper_entry(
    meta_df: pd.DataFrame, battery_id: str, temperature_k: float
) -> pd.DataFrame:
    """Locate metadata rows matching a battery ID and temperature."""

    temp_c = round(temperature_k - 273.15)
    candidates = [
        f"{battery_id}_{temp_c:02d}degC",
        f"{battery_id}_{temp_c}degC",
        battery_id,
    ]

    normalized = meta_df.copy()
    normalized["_id_norm"] = (
        normalized["Battery_ID"].astype(str).str.strip().str.lower()
    )

    for candidate in candidates:
        rows = normalized[normalized["_id_norm"] == candidate.strip().lower()]
        if not rows.empty:
            return rows

    rows = normalized[normalized["_id_norm"].str.contains(battery_id.lower(), na=False)]
    return rows


def get_cell_metadata(
    meta_df: pd.DataFrame, battery_id: str, temperature_k: float, chemistry: str
) -> CellMetadata:
    """Build cell metadata, falling back to chemistry defaults when necessary."""

    rows = find_mapper_entry(meta_df, battery_id, temperature_k)

    defaults = {
        "LFP": {"capacity": 2.5, "vmax": 3.6, "vmin": 2.0},
        "NCA": {"capacity": 3.0, "vmax": 4.2, "vmin": 2.5},
        "NMC": {"capacity": 3.0, "vmax": 4.2, "vmin": 2.5},
    }
    default = defaults.get(chemistry.upper(), defaults["LFP"])

    if rows.empty:
        return CellMetadata(
            initial_capacity=default["capacity"],
            c_rate=1.0,
            temperature_k=temperature_k,
            vmax=default["vmax"],
            vmin=default["vmin"],
            chemistry=chemistry,
        )

    vmax_series = rows.get("Max_Voltage")
    vmin_series = rows.get("Min_Voltage")
    capacity_series = rows.get("Initial_Capacity_Ah")

    vmax_value = (
        float(vmax_series.iloc[0])
        if vmax_series is not None and not pd.isna(vmax_series.iloc[0])
        else default["vmax"]
    )
    vmin_value = (
        float(vmin_series.iloc[0])
        if vmin_series is not None and not pd.isna(vmin_series.iloc[0])
        else default["vmin"]
    )
    capacity_value = (
        float(capacity_series.iloc[0])
        if capacity_series is not None and not pd.isna(capacity_series.iloc[0])
        else default["capacity"]
    )

    return CellMetadata(
        initial_capacity=capacity_value,
        c_rate=1.0,
        temperature_k=temperature_k,
        vmax=vmax_value,
        vmin=vmin_value,
        chemistry=chemistry,
    )


def collect_stanford_files(
    chemistry_path: str, chemistry: str
) -> List[StanfordFileInfo]:
    """Discover all valid Stanford source files for a chemistry."""

    infos: List[StanfordFileInfo] = []
    for root, _, files in os.walk(chemistry_path):
        for file_name in files:
            if not file_name.lower().endswith((".xlsx", ".txt")):
                continue
            file_path = os.path.join(root, file_name)
            try:
                battery_id, c_rate, temp_k = extract_battery_info_from_filename(
                    file_name
                )
            except ValueError as exc:
                print(f"Skipping {file_name}: {exc}")
                continue
            infos.append(
                StanfordFileInfo(
                    path=file_path,
                    file_name=file_name,
                    battery_id=battery_id,
                    c_rate=c_rate,
                    temperature_k=temp_k,
                    chemistry=chemistry,
                )
            )

    return infos


def group_files_by_battery(
    infos: Iterable[StanfordFileInfo],
) -> Dict[Tuple[str, float], List[StanfordFileInfo]]:
    """Group files by battery identifier and temperature."""

    groups: Dict[Tuple[str, float], List[StanfordFileInfo]] = {}
    for info in infos:
        key = (info.battery_id, round(info.temperature_k, 2))
        groups.setdefault(key, []).append(info)
    return groups


def process_file_group(
    files: List[StanfordFileInfo],
    config: ProcessingConfig,
    cell_meta: CellMetadata,
) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """Parse and aggregate all files for a battery/temperature grouping."""

    aggregated_cycles: List[pd.DataFrame] = []
    errors: Dict[str, str] = {}
    cycle_index = 0

    ordered_files = sorted(files, key=lambda item: item.c_rate)

    reached_cycle_limit = False

    for info in ordered_files:
        try:
            raw_df = load_source_file(info.path)
            filtered_df = restrict_voltage_range(raw_df, cell_meta.vmin, cell_meta.vmax)
            working_df = filtered_df if not filtered_df.empty else raw_df

            threshold = compute_current_threshold(
                working_df["Current(A)"], config.current_tolerance
            )
            cycles = extract_cycle_frames(
                working_df, max(threshold, config.current_tolerance)
            )
            if not cycles:
                raise ValueError("No complete charge/discharge cycles detected")

            for cycle_df in cycles:
                if cycle_df.empty:
                    continue

                if cycle_index >= config.max_cycles:
                    reached_cycle_limit = True
                    break

                cycle_index += 1
                cycle_df = cycle_df.copy()
                cycle_df["cycle_index"] = cycle_index
                cycle_df["c_rate"] = info.c_rate
                cycle_df["temperature_k"] = info.temperature_k
                aggregated_cycles.append(cycle_df)

            if reached_cycle_limit:
                break

        except Exception as exc:
            errors[info.file_name] = str(exc)

        if reached_cycle_limit:
            break

    if reached_cycle_limit:
        errors.setdefault(
            "cycle_limit",
            f"Reached maximum cycle limit of {config.max_cycles}; remaining cycles were skipped",
        )

    if not aggregated_cycles:
        return pd.DataFrame(), errors

    agg_df = pd.concat(aggregated_cycles, ignore_index=True)
    return agg_df, errors


def process_chemistry(
    chemistry: str,
    raw_base_path: str,
    project_root: str,
    meta_df: pd.DataFrame,
    config: ProcessingConfig,
) -> None:
    """Process every Stanford battery for a given chemistry."""

    chemistry_path = os.path.join(raw_base_path, chemistry)
    if not os.path.isdir(chemistry_path):
        return

    print(f"Processing chemistry {chemistry}")

    file_infos = collect_stanford_files(chemistry_path, chemistry)
    file_groups = group_files_by_battery(file_infos)

    if not file_groups:
        print(f"No Stanford files found for chemistry {chemistry}")
        return

    for (battery_id, _temp_key), files in sorted(file_groups.items()):
        temperature_k = files[0].temperature_k
        battery_key = normalize_battery_key(battery_id, temperature_k)
        output_dir = config.get_processed_dir(project_root, chemistry, battery_key)

        cell_meta = get_cell_metadata(meta_df, battery_id, temperature_k, chemistry)

        agg_df, errors = process_file_group(files, config, cell_meta)
        if agg_df.empty:
            save_error_log(errors, output_dir, battery_key)
            print(
                f"No aggregated data produced for {battery_key}; see error log if present."
            )
            continue

        charge_df, discharge_df = prepare_resampled_outputs(
            agg_df,
            chemistry,
            battery_key,
            config,
            cell_meta.temperature_k,
            cell_meta.c_rate,
        )

        if charge_df.empty and discharge_df.empty:
            save_error_log(errors, output_dir, battery_key)
            print(f"No charge or discharge segments available for {battery_key}")
            continue

        save_processed_data(charge_df, discharge_df, output_dir, battery_key)
        save_error_log(errors, output_dir, battery_key)


def main(config: Optional[ProcessingConfig] = None) -> None:
    """Entry point for the refactored Stanford parser."""

    if config is None:
        config = ProcessingConfig()

    start_time = time.time()
    print("Starting Stanford data aggregation")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.join(script_dir, "..", "..")

    raw_base_path = config.get_raw_base_path(project_root)
    processed_root = config.get_processed_dir(project_root, "")
    os.makedirs(processed_root, exist_ok=True)

    meta_df = load_meta_properties(sheet_name=config.metadata_sheet)

    chemistries = [
        name
        for name in os.listdir(raw_base_path)
        if os.path.isdir(os.path.join(raw_base_path, name))
    ]

    for chemistry in sorted(chemistries):
        process_chemistry(chemistry, raw_base_path, project_root, meta_df, config)

    elapsed = time.time() - start_time
    hours = int(elapsed // 3600)
    minutes = int((elapsed % 3600) // 60)
    seconds = int(elapsed % 60)

    print("Stanford parsing finished")
    print(f"Total processing time: {hours:02d}:{minutes:02d}:{seconds:02d}")


if __name__ == "__main__":
    main()

# Stanford parsing finished
# Total processing time: 00:10:07