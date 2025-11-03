import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.io import loadmat


def _ensure_import_path() -> None:
    """Ensure the project src directory is on sys.path for direct execution."""

    src_dir = Path(__file__).resolve().parents[1]
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))


_ensure_import_path()


@dataclass
class ProcessingConfig:
    """Configuration for TU Finland aggregation workflow."""

    raw_data_rel_path: str = os.path.join("assets", "raw", "TU_Finland")
    processed_rel_root: str = os.path.join("assets", "processed")
    chemistries: Tuple[str, ...] = ("LFP", "NCA", "NMC")
    sample_points: int = 100
    current_tolerance: float = 1e-4

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
    """Container for per-cell metadata used during aggregation."""

    initial_capacity: float
    c_rate: float
    temperature: float
    vmax: float
    vmin: float
    chemistry: str


@dataclass
class MatFileInfo:
    """Descriptor for a TU Finland MAT source file."""

    path: str
    battery_id: str
    chemistry: str

    @property
    def battery_key(self) -> str:
        return self.battery_id.lower()

    @property
    def file_name(self) -> str:
        return os.path.basename(self.path)


DEFAULT_METADATA: Dict[str, Dict[str, float]] = {
    "LFP": {"initial_capacity": 2.5, "c_rate": 1.0, "temperature": 298.15, "vmax": 3.6, "vmin": 2.5},
    "NCA": {"initial_capacity": 3.0, "c_rate": 1.0, "temperature": 298.15, "vmax": 4.2, "vmin": 2.7},
    "NMC": {"initial_capacity": 3.0, "c_rate": 1.0, "temperature": 298.15, "vmax": 4.2, "vmin": 2.7},
}


def extract_battery_specs_from_mat(file_path: str, percentile: int = 95) -> Tuple[float, float, float, float]:
    """Estimate voltage limits, capacity, and C-rate from a TU Finland MAT file."""

    mat_data = loadmat(file_path)
    table = mat_data.get("table")
    if table is None:
        raise ValueError("MAT file does not contain 'table' key")

    all_voltages: List[float] = []
    charge_voltages: List[float] = []
    discharge_voltages: List[float] = []
    charge_currents: List[float] = []

    for entry in table[0]:
        voltage_raw = entry[1][0] if len(entry) > 1 and len(entry[1]) > 0 else np.array([])
        current_raw = entry[2][0] if len(entry) > 2 and len(entry[2]) > 0 else np.array([])

        if voltage_raw.size == 0 or current_raw.size == 0:
            continue

        all_voltages.extend(voltage_raw.tolist())

        non_zero_mask = np.abs(current_raw) > 0.01
        if np.any(non_zero_mask):
            charge_mask = current_raw > 0.01
            discharge_mask = current_raw < -0.01

            if np.any(charge_mask):
                charge_voltages.extend(voltage_raw[charge_mask].tolist())
                charge_currents.extend(current_raw[charge_mask].tolist())
            if np.any(discharge_mask):
                discharge_voltages.extend(voltage_raw[discharge_mask].tolist())

    if not all_voltages:
        return 3.6, 2.5, 2.5, 1.0

    if charge_voltages and discharge_voltages:
        vmax = float(np.percentile(charge_voltages, percentile))
        vmin = float(np.percentile(discharge_voltages, 100 - percentile))
    else:
        vmax = float(np.percentile(all_voltages, 98))
        vmin = float(np.percentile(all_voltages, 2))

    if charge_currents:
        max_charge_current = max(charge_currents)
        if max_charge_current > 0:
            if vmax > 4.0:
                estimated_capacity = max(2.5, min(5.0, max_charge_current / 0.8))
            else:
                estimated_capacity = max(1.5, min(4.0, max_charge_current / 0.8))
            c_rate = 1.0
        else:
            estimated_capacity = 2.5
            c_rate = 1.0
    else:
        estimated_capacity = 2.5
        c_rate = 1.0

    return vmax, vmin, float(estimated_capacity), float(c_rate)


def load_from_mat(file_path: str) -> pd.DataFrame:
    """Load a TU Finland MAT file into a tidy dataframe."""

    mat_data = loadmat(file_path, struct_as_record=False, squeeze_me=True)
    table = mat_data.get("table")
    if table is None:
        raise ValueError("MAT file does not contain 'table' key")

    entries = np.atleast_1d(table)
    frames: List[pd.DataFrame] = []
    time_offset = 0.0

    for idx, entry in enumerate(entries):
        voltage_raw = getattr(entry, "Voltage", None)
        current_raw = getattr(entry, "Current", None)
        temperature_raw = getattr(entry, "Temperature", None)

        voltage_array = (
            np.asarray(voltage_raw, dtype=float).ravel() if voltage_raw is not None else np.array([])
        )
        current_array = (
            np.asarray(current_raw, dtype=float).ravel() if current_raw is not None else np.array([])
        )
        temperature_array = (
            np.asarray(temperature_raw, dtype=float).ravel()
            if temperature_raw is not None
            else np.zeros_like(voltage_array)
        )

        if voltage_array.size == 0 or current_array.size == 0:
            continue

        if temperature_array.size not in (0, voltage_array.size):
            temperature_array = np.resize(temperature_array, voltage_array.size)

        time_raw = getattr(entry, "Time", None)
        time_array: Optional[np.ndarray] = None
        if time_raw is not None:
            try:
                time_candidate = np.asarray(time_raw, dtype=float).ravel()
            except Exception:  # MATLAB opaque duration/string, fallback to indices
                time_candidate = None
            if time_candidate is not None and time_candidate.size == voltage_array.size:
                time_array = time_candidate - float(time_candidate[0])

        if time_array is None:
            time_array = np.arange(voltage_array.size, dtype=float)

        global_time = time_array + time_offset

        frame = pd.DataFrame(
            {
                "Test_Time(s)": global_time,
                "Voltage(V)": voltage_array,
                "Current(A)": current_array,
                "Temperature(K)": temperature_array,
                "Entry_Index": idx,
            }
        )

        frames.append(frame)

        if len(time_array) > 1:
            delta = float(time_array[-1] - time_array[-2])
            if np.isclose(delta, 0.0):
                delta = 1.0
        else:
            delta = 1.0

        time_offset = float(global_time[-1] + delta)

    if not frames:
        raise ValueError("No valid voltage/current pairs found in MAT file")

    df = pd.concat(frames, ignore_index=True)
    df.dropna(subset=["Test_Time(s)", "Voltage(V)", "Current(A)"] , inplace=True)
    df.sort_values("Test_Time(s)", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def compute_current_threshold(current: pd.Series, fallback: float = 1e-4) -> float:
    """Derive a tolerance to distinguish charge and discharge segments."""

    non_zero = np.abs(current[current != 0])
    if len(non_zero) == 0:
        return fallback
    return max(0.05 * float(np.nanmedian(non_zero)), fallback)


def detect_charge_discharge_cycles(
    df: pd.DataFrame, tolerance: float
) -> List[Tuple[int, int, int, int]]:
    """Identify charge/discharge cycle boundaries."""

    current = df["Current(A)"].to_numpy()
    sign = np.zeros_like(current, dtype=int)
    sign[current > tolerance] = 1
    sign[current < -tolerance] = -1

    cycles: List[Tuple[int, int, int, int]] = []
    idx = 0
    n = len(sign)

    while idx < n:
        while idx < n and sign[idx] <= 0:
            idx += 1
        if idx >= n:
            break

        charge_start = idx
        while idx < n and sign[idx] >= 0:
            idx += 1
        charge_end = idx

        while idx < n and sign[idx] == 0:
            idx += 1
        if idx >= n or sign[idx] > 0:
            continue

        discharge_start = idx
        while idx < n and sign[idx] <= 0:
            idx += 1
        discharge_end = idx

        if discharge_end - charge_start > 1:
            cycles.append((charge_start, charge_end, discharge_start, discharge_end))

    return cycles


def extract_cycle_frames(df: pd.DataFrame, tolerance: float) -> List[pd.DataFrame]:
    """Return dataframes for each detected charge/discharge cycle."""

    cycles = detect_charge_discharge_cycles(df, tolerance)
    frames: List[pd.DataFrame] = []

    for charge_start, charge_end, discharge_start, discharge_end in cycles:
        start = charge_start
        end = discharge_end
        if end - start <= 1:
            continue
        frames.append(df.iloc[start:end].reset_index(drop=True))

    return frames


def clip_cycle_voltage_window(
    cycle_df: pd.DataFrame, vmax: Optional[float], vmin: Optional[float], tolerance: float = 0.01
) -> pd.DataFrame:
    """Clip a cycle to the provided voltage window, excluding CV holds when possible."""

    if cycle_df.empty or vmax is None or vmin is None:
        return cycle_df.reset_index(drop=True)

    voltage = cycle_df["Voltage(V)"].to_numpy()
    current = cycle_df["Current(A)"].to_numpy()

    current_samples = np.abs(current[np.abs(current) > 1e-6])
    if current_samples.size:
        current_threshold = max(0.05 * float(np.nanmedian(current_samples)), tolerance)
    else:
        current_threshold = tolerance

    discharge_start = None
    for idx, value in enumerate(current):
        if value < -current_threshold:
            discharge_start = idx
            break
    if discharge_start is None:
        discharge_start = len(cycle_df)

    charge_end_idx = discharge_start
    if discharge_start > 0:
        charge_voltage = voltage[:discharge_start]
        charge_current = current[:discharge_start]

        vmax_first = None
        for idx, value in enumerate(charge_voltage):
            if value >= vmax - tolerance:
                vmax_first = idx
                break

        if vmax_first is not None:
            cv_hold_start = None
            for idx in range(vmax_first, len(charge_voltage)):
                if idx + 5 >= len(charge_voltage):
                    break
                voltage_stable = np.all(
                    np.abs(charge_voltage[idx : idx + 5] - vmax) < tolerance
                )
                current_decreasing = (
                    charge_current[idx] < 0.5 * charge_current[vmax_first]
                    if charge_current[vmax_first] > 0
                    else False
                )
                if voltage_stable and current_decreasing:
                    cv_hold_start = idx
                    break
            charge_end_idx = cv_hold_start if cv_hold_start is not None else vmax_first + 1
        else:
            charge_end_idx = discharge_start

    discharge_end_idx = len(cycle_df)
    if discharge_start < len(cycle_df):
        discharge_voltage = voltage[discharge_start:]
        for idx, value in enumerate(discharge_voltage):
            if value <= vmin + tolerance:
                discharge_end_idx = discharge_start + idx + 1
                break

    indices: List[int] = []
    if charge_end_idx > 0:
        indices.extend(range(0, min(charge_end_idx, len(cycle_df))))
    if discharge_start < discharge_end_idx:
        indices.extend(range(discharge_start, min(discharge_end_idx, len(cycle_df))))

    if not indices:
        return cycle_df.reset_index(drop=True)

    filtered = cycle_df.iloc[indices].copy()
    return filtered.reset_index(drop=True)


def resample_cycle_segment(segment_df: pd.DataFrame, sample_points: int) -> pd.DataFrame:
    """Resample a charge or discharge segment to a fixed number of samples."""

    required = ["Test_Time(s)", "Voltage(V)", "Current(A)"]
    if segment_df is None or segment_df.empty:
        return pd.DataFrame()

    segment = segment_df[required].dropna().copy()
    segment.drop_duplicates(subset="Test_Time(s)", inplace=True)
    segment.sort_values("Test_Time(s)", inplace=True)
    if segment.empty:
        return pd.DataFrame()

    time_values = segment["Test_Time(s)"].to_numpy()
    relative_time = time_values - time_values[0]

    result = pd.DataFrame(
        {
            "sample_index": np.arange(sample_points, dtype=int),
            "normalized_time": np.linspace(0.0, 1.0, sample_points),
        }
    )

    if len(segment) == 1 or np.isclose(relative_time[-1], 0.0):
        result["elapsed_time_s"] = np.zeros(sample_points)
        result["voltage_v"] = np.full(sample_points, segment["Voltage(V)"].iloc[0])
        result["current_a"] = np.full(sample_points, segment["Current(A)"].iloc[0])
        return result

    normalized_time = relative_time / relative_time[-1]
    target = result["normalized_time"].to_numpy()

    result["elapsed_time_s"] = np.interp(target, normalized_time, relative_time)
    result["voltage_v"] = np.interp(target, normalized_time, segment["Voltage(V)"].to_numpy())
    result["current_a"] = np.interp(target, normalized_time, segment["Current(A)"].to_numpy())
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


def prepare_resampled_outputs(
    agg_df: pd.DataFrame, cell_meta: CellMetadata, config: ProcessingConfig, battery_key: str
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

    if agg_df.empty or "cycle_index" not in agg_df.columns:
        empty = pd.DataFrame(columns=columns)
        return empty.copy(), empty.copy()

    charge_segments: List[pd.DataFrame] = []
    discharge_segments: List[pd.DataFrame] = []

    for cycle in sorted(int(c) for c in agg_df["cycle_index"].dropna().unique()):
        cycle_df = agg_df[agg_df["cycle_index"] == cycle].copy()
        if cycle_df.empty:
            continue

        charge_segment, discharge_segment = split_cycle_segments(
            cycle_df, tolerance=config.current_tolerance
        )

        for label, segment in (("charge", charge_segment), ("discharge", discharge_segment)):
            if segment.empty:
                continue

            resampled = resample_cycle_segment(segment, config.sample_points)
            if resampled.empty:
                continue

            resampled["battery_id"] = battery_key
            resampled["chemistry"] = cell_meta.chemistry
            resampled["cycle_index"] = int(cycle)
            resampled["c_rate"] = cell_meta.c_rate
            resampled["temperature_k"] = cell_meta.temperature
            resampled = resampled[columns]

            if label == "charge":
                charge_segments.append(resampled)
            else:
                discharge_segments.append(resampled)

    charge_df = (
        pd.concat(charge_segments, ignore_index=True)
        if charge_segments
        else pd.DataFrame(columns=columns)
    )
    discharge_df = (
        pd.concat(discharge_segments, ignore_index=True)
        if discharge_segments
        else pd.DataFrame(columns=columns)
    )

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


def save_error_log(error_dict: Dict[str, str], output_dir: str, battery_key: str) -> None:
    """Write an error log CSV for a specific battery when issues occur."""

    error_path = os.path.join(output_dir, f"error_log_{battery_key}.csv")

    if not error_dict:
        if os.path.exists(error_path):
            os.remove(error_path)
        return

    os.makedirs(output_dir, exist_ok=True)
    error_df = pd.DataFrame(
        list(error_dict.items()), columns=["stage", "error_message"]
    )
    error_df.to_csv(error_path, index=False)


def get_cell_metadata_from_file(
    file_path: str, battery_type: str
) -> Tuple[CellMetadata, Optional[str]]:
    """Extract metadata from a MAT file, falling back to chemistry defaults if needed."""

    chemistry = battery_type.upper()
    defaults = DEFAULT_METADATA.get(chemistry, DEFAULT_METADATA["LFP"])

    try:
        vmax, vmin, capacity, c_rate = extract_battery_specs_from_mat(
            file_path, percentile=95
        )
        metadata = CellMetadata(
            initial_capacity=capacity,
            c_rate=c_rate,
            temperature=defaults["temperature"],
            vmax=vmax,
            vmin=vmin,
            chemistry=chemistry,
        )
        return metadata, None
    except Exception as exc:
        metadata = CellMetadata(
            initial_capacity=defaults["initial_capacity"],
            c_rate=defaults["c_rate"],
            temperature=defaults["temperature"],
            vmax=defaults["vmax"],
            vmin=defaults["vmin"],
            chemistry=chemistry,
        )
        return metadata, f"Metadata fallback applied: {exc}"


def discover_mat_files(chemistry_path: str, chemistry: str) -> List[MatFileInfo]:
    """Locate MAT files for the specified chemistry."""

    entries: List[MatFileInfo] = []
    if not os.path.isdir(chemistry_path):
        return entries

    for file_name in sorted(os.listdir(chemistry_path)):
        if not file_name.lower().endswith(".mat"):
            continue
        if file_name.lower().startswith("ocv"):
            continue
        path = os.path.join(chemistry_path, file_name)
        entries.append(
            MatFileInfo(
                path=path,
                battery_id=Path(file_name).stem,
                chemistry=chemistry,
            )
        )

    return entries


def process_single_mat_file(
    info: MatFileInfo, project_root: str, config: ProcessingConfig
) -> Dict[str, str]:
    """Process a single MAT file and persist charge/discharge aggregates."""

    errors: Dict[str, str] = {}

    output_dir = config.get_processed_dir(project_root, info.chemistry, info.battery_key)

    cell_meta, metadata_warning = get_cell_metadata_from_file(info.path, info.chemistry)
    if metadata_warning:
        errors["metadata"] = metadata_warning

    try:
        raw_df = load_from_mat(info.path)
    except Exception as exc:
        errors["load"] = str(exc)
        raw_df = pd.DataFrame()

    cycle_frames: List[pd.DataFrame] = []
    if raw_df.empty:
        errors.setdefault("cycles", "No data available after loading")
    else:
        threshold = max(
            compute_current_threshold(raw_df["Current(A)"], config.current_tolerance),
            config.current_tolerance,
        )
        cycles = extract_cycle_frames(raw_df, threshold)
        if not cycles:
            errors.setdefault("cycles", "No complete charge/discharge cycles detected")
        else:
            for idx, cycle_df in enumerate(cycles, start=1):
                trimmed = clip_cycle_voltage_window(
                    cycle_df, cell_meta.vmax, cell_meta.vmin
                )
                trimmed = trimmed[
                    ["Test_Time(s)", "Voltage(V)", "Current(A)"]
                ].dropna()
                trimmed.drop_duplicates(subset="Test_Time(s)", inplace=True)
                trimmed.sort_values("Test_Time(s)", inplace=True)
                trimmed.reset_index(drop=True, inplace=True)
                if trimmed.empty:
                    continue
                trimmed["cycle_index"] = idx
                trimmed["c_rate"] = cell_meta.c_rate
                trimmed["temperature_k"] = cell_meta.temperature
                cycle_frames.append(trimmed)

            if not cycle_frames:
                errors.setdefault(
                    "cycles", "No valid cycles remained after voltage clipping"
                )

    agg_df = (
        pd.concat(cycle_frames, ignore_index=True)
        if cycle_frames
        else pd.DataFrame(columns=["Test_Time(s)", "Voltage(V)", "Current(A)", "cycle_index"])
    )

    charge_df, discharge_df = prepare_resampled_outputs(
        agg_df, cell_meta, config, info.battery_key
    )

    save_processed_data(charge_df, discharge_df, output_dir, info.battery_key)
    save_error_log(errors, output_dir, info.battery_key)

    return errors


def process_chemistry(
    chemistry: str, raw_base_path: str, project_root: str, config: ProcessingConfig
) -> Dict[str, str]:
    """Process every MAT file for a given chemistry and return aggregated errors."""

    chemistry_path = os.path.join(raw_base_path, chemistry)
    mat_files = discover_mat_files(chemistry_path, chemistry)

    if not mat_files:
        print(f"No MAT files found for chemistry {chemistry}")
        return {}

    print(f"Processing chemistry {chemistry} with {len(mat_files)} files")

    chemistry_errors: Dict[str, str] = {}
    for info in mat_files:
        print(f"  Parsing {info.file_name}")
        errors = process_single_mat_file(info, project_root, config)
        if errors:
            chemistry_errors[info.battery_id] = "; ".join(
                f"{stage}: {message}" for stage, message in errors.items()
            )

    return chemistry_errors


def main(config: Optional[ProcessingConfig] = None) -> None:
    """Entry point for TU Finland MAT aggregation."""

    if config is None:
        config = ProcessingConfig()

    start_time = time.time()
    print("Starting TU Finland data aggregation")

    project_root_path = Path(__file__).resolve().parents[2]
    project_root = str(project_root_path)

    raw_base_path = config.get_raw_base_path(project_root)

    all_errors: Dict[str, Dict[str, str]] = {}

    for chemistry in config.chemistries:
        processed_dir = config.get_processed_dir(project_root, chemistry)
        os.makedirs(processed_dir, exist_ok=True)
        chemistry_errors = process_chemistry(
            chemistry, raw_base_path, project_root, config
        )
        if chemistry_errors:
            all_errors[chemistry] = chemistry_errors

    end_time = time.time()
    total_time = end_time - start_time

    print("\nProcessing summary")
    for chemistry in config.chemistries:
        if chemistry in all_errors:
            print(
                f"  {chemistry}: {len(all_errors[chemistry])} file(s) reported issues"
            )
        else:
            print(f"  {chemistry}: success")

    hours = int(total_time // 3600)
    minutes = int((total_time % 3600) // 60)
    seconds = int(total_time % 60)
    print(f"Total processing time: {hours:02d}:{minutes:02d}:{seconds:02d}")


if __name__ == "__main__":
    main()
    
# Total processing time: 00:02:14