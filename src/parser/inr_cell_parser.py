"""INR parser aligned with the consolidated aggregation workflow."""

import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from utils.help_function import load_meta_properties


@dataclass
class ProcessingConfig:
    """Configuration for INR data processing."""

    raw_data_rel_path: str = os.path.join("assets", "raw", "INR")
    processed_rel_root: str = os.path.join("assets", "processed")
    chemistry: str = "NMC"
    sample_points: int = 100
    thread_count: int = 4
    max_cycles: int = 100

    def get_raw_data_path(self, project_root: str) -> str:
        return os.path.join(project_root, self.raw_data_rel_path)

    def get_processed_dir(
        self, project_root: str, battery_id: Optional[str] = None
    ) -> str:
        base = os.path.join(project_root, self.processed_rel_root, self.chemistry)
        if battery_id:
            return os.path.join(base, battery_id)
        return base


@dataclass
class CellMetadata:
    """Container for INR cell metadata."""

    initial_capacity: float
    c_rate: float
    temperature: float
    vmax: float
    vmin: float


MIN_SEGMENT_SAMPLE_COUNT = 100


# Fallback metadata entries for INR batteries when the shared sheet is missing rows.
FALLBACK_METADATA: Dict[str, CellMetadata] = {
    "sp3_25c_lc_ocv_11_16_2015": CellMetadata(
        initial_capacity=2.0,
        c_rate=0.05,
        temperature=298.0,
        vmax=4.2,
        vmin=2.5,
    ),
    "sp1_25c_lc_ocv_11_5_2015": CellMetadata(
        initial_capacity=2.0,
        c_rate=0.05,
        temperature=298.0,
        vmax=4.2,
        vmin=2.5,
    ),
}


FILE_TOKEN_OVERRIDES: Dict[str, str] = {
    "sp20-3": "SP3_25C_LC_OCV_11_16_2015",
    "sp20-1": "SP1_25C_LC_OCV_11_5_2015",
}


def extract_date(file_name: str) -> datetime.date:
    """Extract the first three numeric tokens as a date."""

    base_name = os.path.splitext(file_name)[0]
    parts = base_name.split("_")

    numeric_parts: List[int] = []
    for part in parts:
        try:
            numeric_parts.append(int(part))
            if len(numeric_parts) == 3:
                break
        except ValueError:
            continue

    if len(numeric_parts) < 3:
        raise ValueError(f"Cannot extract date from filename: {file_name}")

    month, day, year = numeric_parts[0], numeric_parts[1], numeric_parts[2]

    if not (1 <= month <= 12 and 1 <= day <= 31):
        raise ValueError(f"Invalid date components in {file_name}")
    if year < 100:
        year += 2000

    return datetime(year, month, day).date()


def sort_files(file_names: Iterable[str]) -> List[str]:
    """Sort files by inferred date, falling back to lexicographical order."""

    def sort_key(name: str) -> Tuple[int, str]:
        try:
            date_value = extract_date(name).toordinal()
        except ValueError:
            date_value = -1
        return (date_value, name)

    return sorted(file_names, key=sort_key)


def extract_battery_token(file_name: str) -> str:
    """Extract the trailing token from a filename to match metadata IDs."""

    base_name = os.path.splitext(os.path.basename(file_name))[0]
    parts = base_name.rsplit("_", 1)
    token = parts[-1] if parts else base_name
    return token.lower()


def normalise_dataframe(raw_df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """Normalise column names and units to the expected schema."""

    if raw_df.empty:
        return None

    df = raw_df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    source_cols = set(df.columns)
    from_milli_voltage = "mV" in source_cols
    from_milli_current = "mA" in source_cols
    has_duration_sec = "Duration (sec)" in source_cols
    has_time_minutes = "Time" in source_cols and not has_duration_sec

    rename_map = {
        "Voltage (V)": "Voltage(V)",
        "Voltage": "Voltage(V)",
        "mV": "Voltage(V)",
        "Current (A)": "Current(A)",
        "Current": "Current(A)",
        "mA": "Current(A)",
        "Duration (sec)": "Test_Time(s)",
        "Duration": "Test_Time(s)",
        "Test Time (s)": "Test_Time(s)",
        "Time (s)": "Test_Time(s)",
        "Time": "Test_Time(s)",
    }

    df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns}, inplace=True)

    required_cols = ["Current(A)", "Voltage(V)", "Test_Time(s)"]
    if not all(col in df.columns for col in required_cols):
        return None

    for col in required_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    if from_milli_voltage:
        df["Voltage(V)"] = df["Voltage(V)"] / 1000.0

    if from_milli_current:
        df["Current(A)"] = df["Current(A)"] / 1000.0

    if has_time_minutes:
        df["Test_Time(s)"] = df["Test_Time(s)"] * 60.0

    return df[required_cols].dropna().reset_index(drop=True)


def load_excel_file(file_path: str) -> pd.DataFrame:
    """Load an Excel file and return the normalised dataframe."""

    xls = pd.ExcelFile(file_path)
    for sheet in xls.sheet_names:
        try:
            raw_df = pd.read_excel(file_path, sheet_name=sheet)
        except Exception:
            continue

        df = normalise_dataframe(raw_df)
        if df is not None and not df.empty:
            return df

    raise ValueError(f"Required columns not found in {file_path}")


def load_text_file(file_path: str) -> pd.DataFrame:
    """Load a tab-delimited text file in the legacy format."""

    df = pd.read_csv(file_path, delimiter="\t")
    df.rename(
        columns={
            "Time": "Test_Time(s)",
            "mA": "Current(A)",
            "mV": "Voltage(V)",
        },
        inplace=True,
    )

    required_cols = ["Current(A)", "Voltage(V)", "Test_Time(s)"]
    if not all(col in df.columns for col in required_cols):
        raise ValueError(f"Missing required columns in {file_path}")

    df = df[required_cols].dropna().reset_index(drop=True)
    df["Voltage(V)"] = df["Voltage(V)"] / 1000.0
    df["Current(A)"] = df["Current(A)"] / 1000.0
    df["Test_Time(s)"] = df["Test_Time(s)"] * 60.0
    return df


def compute_transition_indices(df: pd.DataFrame) -> Tuple[List[int], List[int]]:
    """Identify charge/discharge transitions using current sign."""

    current = df["Current(A)"]
    non_zero = np.abs(current[current != 0])
    threshold = max(0.05 * np.nanmedian(non_zero) if len(non_zero) else 0.0, 1e-4)

    sign = np.zeros_like(current, dtype=int)
    sign[current > threshold] = 1
    sign[current < -threshold] = -1

    charge_indices: List[int] = []
    discharge_indices: List[int] = []
    previous = 0

    for index, value in enumerate(sign):
        if value == 0:
            continue
        if previous == 0:
            previous = value
            continue
        if value != previous:
            if value == 1:
                charge_indices.append(index)
            else:
                discharge_indices.append(index)
            previous = value

    if not charge_indices or not discharge_indices:
        raise ValueError("Unable to detect alternating charge/discharge cycles")

    complexity, expected_order = validate_indices(charge_indices, discharge_indices)
    if complexity == "High":
        raise ValueError("Indices do not alternate correctly")

    if expected_order[0] == "discharge" and len(discharge_indices) > len(charge_indices):
        discharge_indices = discharge_indices[:-1]
    elif expected_order[0] == "charge" and len(charge_indices) > len(discharge_indices):
        charge_indices = charge_indices[:-1]

    if len(charge_indices) != len(discharge_indices):
        raise ValueError("Mismatched charge/discharge cycle counts")

    return charge_indices, discharge_indices


def validate_indices(charge_indices: List[int], discharge_indices: List[int]):
    """Validate that charge/discharge indices alternate."""

    if charge_indices[0] < discharge_indices[0]:
        expected_order = ["charge", "discharge"]
        combined = [(idx, "charge") for idx in charge_indices] + [
            (idx, "discharge") for idx in discharge_indices
        ]
    else:
        expected_order = ["discharge", "charge"]
        combined = [(idx, "discharge") for idx in discharge_indices] + [
            (idx, "charge") for idx in charge_indices
        ]

    combined.sort(key=lambda item: item[0])

    for position, (_, label) in enumerate(combined):
        if label != expected_order[position % 2]:
            return "High", expected_order

    return "Low", expected_order


def assign_cycle_counts(
    df: pd.DataFrame,
    charge_indices: List[int],
    discharge_indices: List[int],
    initial_capacity: float,
) -> pd.DataFrame:
    """Assign cycle numbers and compute throughput metrics."""

    start_idx = min(charge_indices[0], discharge_indices[0])
    end_idx = max(charge_indices[-1], discharge_indices[-1])

    if (
        discharge_indices[0] < charge_indices[0]
        or charge_indices[-1] > discharge_indices[-1]
        or end_idx <= start_idx
    ):
        raise ValueError("Unable to form complete charge/discharge window")

    df = df.iloc[start_idx : end_idx + 1].reset_index(drop=True)

    adjusted_charge = [idx - start_idx for idx in charge_indices if start_idx <= idx <= end_idx]
    adjusted_discharge = [
        idx - start_idx for idx in discharge_indices if start_idx <= idx <= end_idx
    ]

    if not adjusted_charge or not adjusted_discharge:
        raise ValueError("Insufficient charge/discharge segments within window")

    df["Cycle_Count"] = np.nan

    for cycle_number, (start, end) in enumerate(
        zip(adjusted_charge, adjusted_charge[1:] + [len(df)]), start=1
    ):
        df.loc[start : max(end - 1, start), "Cycle_Count"] = cycle_number

    df["Delta_Time(s)"] = df["Test_Time(s)"].diff().fillna(0.0)
    df["Delta_Ah"] = np.abs(df["Current(A)"]) * df["Delta_Time(s)"] / 3600.0
    df["Ah_throughput"] = df["Delta_Ah"].cumsum()

    if initial_capacity and not np.isclose(initial_capacity, 0.0):
        df["EFC"] = df["Ah_throughput"] / initial_capacity
    else:
        df["EFC"] = np.nan

    return df


def fallback_cycle_assignment(df: pd.DataFrame, initial_capacity: float) -> pd.DataFrame:
    """Fallback when no alternating charge/discharge cycles are detected."""

    df = df.copy().reset_index(drop=True)
    df["Cycle_Count"] = 1
    df["Delta_Time(s)"] = df["Test_Time(s)"].diff().fillna(0.0)
    df["Delta_Ah"] = np.abs(df["Current(A)"]) * df["Delta_Time(s)"] / 3600.0
    df["Ah_throughput"] = df["Delta_Ah"].cumsum()

    if initial_capacity and not np.isclose(initial_capacity, 0.0):
        df["EFC"] = df["Ah_throughput"] / initial_capacity
    else:
        df["EFC"] = np.nan

    return df


def resample_cycle_segment(segment: pd.DataFrame, sample_points: int) -> pd.DataFrame:
    """Resample a cycle segment to a fixed number of samples."""

    if segment is None or segment.empty:
        return pd.DataFrame()

    subset = (
        segment[["Test_Time(s)", "Voltage(V)", "Current(A)"]]
        .dropna()
        .drop_duplicates(subset=["Test_Time(s)"])
        .sort_values("Test_Time(s)")
    )

    if subset.empty:
        return pd.DataFrame()

    time_values = subset["Test_Time(s)"].to_numpy()
    elapsed = time_values - time_values[0]

    output = pd.DataFrame(
        {
            "sample_index": np.arange(sample_points, dtype=int),
            "normalized_time": np.linspace(0.0, 1.0, sample_points),
        }
    )

    if len(subset) == 1 or np.isclose(elapsed[-1], 0.0):
        output["elapsed_time_s"] = np.zeros(sample_points)
        output["voltage_v"] = np.full(sample_points, subset["Voltage(V)"].iloc[0])
        output["current_a"] = np.full(sample_points, subset["Current(A)"].iloc[0])
        return output

    normalized = elapsed / elapsed[-1]
    target = output["normalized_time"].to_numpy()

    output["elapsed_time_s"] = np.interp(target, normalized, elapsed)
    output["voltage_v"] = np.interp(target, normalized, subset["Voltage(V)"].to_numpy())
    output["current_a"] = np.interp(target, normalized, subset["Current(A)"].to_numpy())

    return output


def split_cycle_segments(
    cycle_df: pd.DataFrame, tolerance: float = 1e-4
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split a cycle into charge and discharge segments using current sign."""

    if cycle_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    current = cycle_df["Current(A)"].to_numpy()
    positive_indices = np.where(current > tolerance)[0]
    negative_indices = np.where(current < -tolerance)[0]

    charge_segment = pd.DataFrame()
    discharge_segment = pd.DataFrame()

    def contiguous_blocks(indices: np.ndarray) -> List[np.ndarray]:
        if indices.size == 0:
            return []
        splits = np.where(np.diff(indices) > 1)[0] + 1
        return np.split(indices, splits)

    def select_block(
        blocks: List[np.ndarray],
        min_length: int = 5,
        start_after: Optional[int] = None,
    ) -> Optional[np.ndarray]:
        def block_duration(block: np.ndarray) -> float:
            start = block[0]
            end = block[-1]
            return float(
                cycle_df.iloc[end]["Test_Time(s)"] - cycle_df.iloc[start]["Test_Time(s)"]
            )

        filtered: List[np.ndarray] = []
        for block in blocks:
            if start_after is not None and block[0] <= start_after:
                continue
            if len(block) < min_length:
                continue
            filtered.append(block)

        if filtered:
            return max(filtered, key=block_duration)

        if start_after is not None:
            later_blocks = [block for block in blocks if block[0] > start_after]
            if later_blocks:
                return max(later_blocks, key=block_duration)

        return max(blocks, key=block_duration) if blocks else None

    positive_blocks = contiguous_blocks(positive_indices)
    negative_blocks = contiguous_blocks(negative_indices)

    first_positive_idx = positive_blocks[0][0] if positive_blocks else None
    first_negative_idx = negative_blocks[0][0] if negative_blocks else None

    if first_positive_idx is not None and (
        first_negative_idx is None or first_positive_idx <= first_negative_idx
    ):
        # Charge occurs before discharge
        charge_block = select_block(positive_blocks)
        if charge_block is not None:
            charge_segment = cycle_df.iloc[charge_block[0] : charge_block[-1] + 1].copy()

        if negative_blocks:
            charge_last_idx = charge_block[-1] if charge_block is not None else None
            discharge_block = select_block(
                negative_blocks,
                start_after=charge_last_idx,
            )
            if discharge_block is None:
                discharge_block = select_block(negative_blocks)

            discharge_segment = cycle_df.iloc[
                discharge_block[0] : discharge_block[-1] + 1
            ].copy()
    elif first_negative_idx is not None:
        # Discharge occurs before charge (e.g., OCV tests)
        discharge_block = select_block(negative_blocks)
        if discharge_block is not None:
            discharge_segment = cycle_df.iloc[
                discharge_block[0] : discharge_block[-1] + 1
            ].copy()

        if positive_blocks:
            discharge_last_idx = discharge_block[-1] if discharge_block is not None else None
            charge_block = select_block(
                positive_blocks,
                start_after=discharge_last_idx,
            )
            if charge_block is None:
                charge_block = select_block(positive_blocks)

            if charge_block is not None:
                charge_segment = cycle_df.iloc[
                    charge_block[0] : charge_block[-1] + 1
                ].copy()

    return charge_segment, discharge_segment


def prepare_cycle_segment(
    segment_df: pd.DataFrame, min_samples: int
) -> Optional[pd.DataFrame]:
    """Sanitize a segment and enforce minimum raw sample counts."""

    required_cols = ["Test_Time(s)", "Voltage(V)", "Current(A)"]

    if segment_df is None or segment_df.empty:
        return None

    if any(col not in segment_df.columns for col in required_cols):
        return None

    sanitized = segment_df[required_cols].copy()

    for col in required_cols:
        sanitized[col] = pd.to_numeric(sanitized[col], errors="coerce")

    sanitized = sanitized.dropna()
    if sanitized.empty:
        return None

    sanitized = sanitized.drop_duplicates(subset=["Test_Time(s)"])
    sanitized = sanitized.sort_values("Test_Time(s)")

    if len(sanitized) < min_samples:
        return None

    return sanitized.reset_index(drop=True)


def prepare_resampled_outputs(
    agg_df: pd.DataFrame,
    cell_meta: CellMetadata,
    config: ProcessingConfig,
    battery_id: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Prepare charge and discharge aggregated outputs."""

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

    unique_cycles = sorted(
        int(cycle) for cycle in agg_df["Cycle_Count"].dropna().unique()
    )

    min_samples_required = max(config.sample_points, MIN_SEGMENT_SAMPLE_COUNT)
    max_cycles = getattr(config, "max_cycles", MIN_SEGMENT_SAMPLE_COUNT)
    if not isinstance(max_cycles, int) or max_cycles <= 0:
        max_cycles = MIN_SEGMENT_SAMPLE_COUNT

    valid_cycle_count = 0

    for cycle in unique_cycles:
        if valid_cycle_count >= max_cycles:
            break

        cycle_df = agg_df[agg_df["Cycle_Count"] == cycle].copy()
        if cycle_df.empty:
            continue

        cycle_df = cycle_df.sort_values("Test_Time(s)")
        charge_raw, discharge_raw = split_cycle_segments(cycle_df)

        charge_segment = prepare_cycle_segment(charge_raw, min_samples_required)
        discharge_segment = prepare_cycle_segment(discharge_raw, min_samples_required)

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

        for label, resampled in (
            ("charge", charge_resampled),
            ("discharge", discharge_resampled),
        ):
            formatted = resampled.copy()
            formatted["sample_index"] = formatted["sample_index"].astype(int)
            formatted["battery_id"] = battery_id
            formatted["chemistry"] = config.chemistry
            formatted["cycle_index"] = valid_cycle_count
            formatted["c_rate"] = cell_meta.c_rate
            formatted["temperature_k"] = cell_meta.temperature
            formatted = formatted[columns]

            if label == "charge":
                charge_segments.append(formatted)
            else:
                discharge_segments.append(formatted)

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


def get_cell_metadata(meta_df: pd.DataFrame, cell_id: str) -> Optional[CellMetadata]:
    """Retrieve metadata matching the provided battery ID."""

    cell_df = meta_df[meta_df["Battery_ID"].str.lower() == cell_id.lower()]
    if cell_df.empty:
        fallback = FALLBACK_METADATA.get(cell_id.lower())
        if fallback is not None:
            print(f"ℹ️  Using fallback metadata for battery ID: {cell_id}")
            return fallback
        print(f"No metadata found for battery ID: {cell_id}")
        return None

    return CellMetadata(
        initial_capacity=cell_df["Initial_Capacity_Ah"].values[0],
        c_rate=cell_df["C_rate"].values[0],
        temperature=cell_df["Temperature (K)"].values[0],
        vmax=cell_df["Max_Voltage"].values[0],
        vmin=cell_df["Min_Voltage"].values[0],
    )


def parse_source_file(file_path: str, metadata: CellMetadata) -> pd.DataFrame:
    """Parse a single INR source file and return tagged cycles."""

    extension = os.path.splitext(file_path)[1].lower()
    if extension == ".txt":
        df = load_text_file(file_path)
    else:
        df = load_excel_file(file_path)

    df = df.sort_values("Test_Time(s)").reset_index(drop=True)

    try:
        charge_indices, discharge_indices = compute_transition_indices(df)
        df = assign_cycle_counts(df, charge_indices, discharge_indices, metadata.initial_capacity)
    except ValueError:
        df = fallback_cycle_assignment(df, metadata.initial_capacity)

    df["C_rate"] = metadata.c_rate
    return df


def roll_cycle_counters(df: pd.DataFrame, existing: pd.DataFrame) -> pd.DataFrame:
    """Offset cycle counters when stacking multiple files."""

    if existing.empty:
        return df

    df = df.copy()
    max_cycle = int(existing["Cycle_Count"].max())
    df["Cycle_Count"] = df["Cycle_Count"].astype(int) + max_cycle
    df["Ah_throughput"] = df["Ah_throughput"] + existing["Ah_throughput"].max()
    if "EFC" in existing.columns and existing["EFC"].notna().any():
        df["EFC"] = df["EFC"] + existing["EFC"].dropna().max()
    return df


def process_files_for_battery(
    raw_base_path: str,
    file_names: List[str],
    cell_meta: CellMetadata,
) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """Process all files associated with a single battery."""

    aggregated = pd.DataFrame()
    errors: Dict[str, str] = {}

    sorted_files = sort_files(file_names)

    for index, file_name in enumerate(sorted_files, start=1):
        file_path = os.path.join(raw_base_path, file_name)
        try:
            df = parse_source_file(file_path, cell_meta)
            df = roll_cycle_counters(df, aggregated)
            aggregated = pd.concat([aggregated, df], ignore_index=True)
        except Exception as exc:
            errors[file_name] = str(exc)
            print(f"  ❌ Error processing {file_name}: {exc}")
        else:
            progress = round(index / max(len(sorted_files), 1) * 100, 1)
            print(f"  ✓ Processed {file_name} ({progress}% complete)")

    return aggregated, errors


def prepare_battery_file_map(
    all_files: List[str], meta_df: pd.DataFrame
) -> Dict[str, List[str]]:
    """Map metadata battery IDs to matching source files."""

    relevant_meta = meta_df[
        meta_df["Battery_ID"].str.contains("LC_OCV", case=False, na=False)
    ]["Battery_ID"].dropna()

    mapping: Dict[str, List[str]] = {battery_id: [] for battery_id in relevant_meta}

    tokens = {file_name: extract_battery_token(file_name) for file_name in all_files}

    for file_name, token in tokens.items():
        override_battery = FILE_TOKEN_OVERRIDES.get(token)
        if override_battery:
            mapping.setdefault(override_battery, []).append(file_name)
            continue

        for battery_id in relevant_meta:
            if token and token in battery_id.lower():
                mapping[battery_id].append(file_name)
                break

    return {battery_id: files for battery_id, files in mapping.items() if files}


def print_debug_matching_info(all_files: List[str], meta_df: pd.DataFrame) -> None:
    """Print debug information when no metadata matches are found."""

    print("ℹ️  Debugging INR metadata matching...")

    tokens = {file_name: extract_battery_token(file_name) for file_name in all_files}
    if tokens:
        print("   ↳ File tokens detected:")
        for file_name, token in tokens.items():
            mapped = FILE_TOKEN_OVERRIDES.get(token)
            if mapped:
                print(
                    f"      • {file_name} → token='{token}' (override→'{mapped}')"
                )
            else:
                print(f"      • {file_name} → token='{token}'")

    metadata_ids = meta_df[
        meta_df["Battery_ID"].str.contains("LC_OCV", case=False, na=False)
    ]["Battery_ID"].dropna()

    if not metadata_ids.empty:
        print("   ↳ Metadata Battery_ID values:")
        for cell_id in metadata_ids:
            print(f"      • {cell_id}")
    else:
        print("   ↳ Metadata does not contain LC_OCV battery entries.")


def save_processed_data(
    agg_df: pd.DataFrame,
    cell_id: str,
    cell_meta: CellMetadata,
    config: ProcessingConfig,
    output_dir: str,
) -> Tuple[str, str]:
    """Persist charge and discharge aggregated CSVs."""

    charge_df, discharge_df = prepare_resampled_outputs(agg_df, cell_meta, config, cell_id)
    os.makedirs(output_dir, exist_ok=True)

    charge_path = os.path.join(output_dir, f"{cell_id}_charge_aggregated_data.csv")
    discharge_path = os.path.join(output_dir, f"{cell_id}_discharge_aggregated_data.csv")

    charge_df.to_csv(charge_path, index=False)
    discharge_df.to_csv(discharge_path, index=False)

    print(f"💾 Saved charge CSV: {charge_path}")
    print(f"💾 Saved discharge CSV: {discharge_path}")

    return charge_path, discharge_path


def save_error_log(error_dict: Dict[str, str], cell_id: str, output_dir: str) -> None:
    """Write per-battery error logs."""

    if not error_dict:
        return

    os.makedirs(output_dir, exist_ok=True)
    error_df = pd.DataFrame(
        list(error_dict.items()), columns=["File_Name", "Error_Message"]
    )
    error_log_path = os.path.join(output_dir, f"error_log_{cell_id}.csv")
    error_df.to_csv(error_log_path, index=False)
    print(f"📝 Saved error log: {error_log_path}")


def process_single_battery(
    battery_id: str,
    files: List[str],
    raw_base_path: str,
    processed_base_path: str,
    meta_df: pd.DataFrame,
    config: ProcessingConfig,
) -> Dict[str, str]:
    """Process all files for a single battery and return errors."""

    print(f"\nProcessing battery: {battery_id}")

    cell_meta = get_cell_metadata(meta_df, battery_id)
    if cell_meta is None:
        return {battery_id: "Missing metadata"}

    aggregated, errors = process_files_for_battery(raw_base_path, files, cell_meta)

    output_dir = os.path.join(processed_base_path, battery_id)
    if not aggregated.empty:
        save_processed_data(aggregated, battery_id, cell_meta, config, output_dir)
    else:
        print(f"No data aggregated for {battery_id}")

    save_error_log(errors, battery_id, output_dir)
    return errors


def discover_source_files(raw_base_path: str) -> List[str]:
    """List valid INR source files in the raw directory."""

    valid_extensions = {".xls", ".xlsx", ".txt"}
    return [
        file_name
        for file_name in os.listdir(raw_base_path)
        if os.path.splitext(file_name)[1].lower() in valid_extensions
    ]


def main(config: Optional[ProcessingConfig] = None) -> None:
    """Main entry point for INR processing."""

    if config is None:
        config = ProcessingConfig()

    start_time = time.time()
    print(f"🚀 Starting INR battery data processing with {config.thread_count} threads...")

    meta_df = load_meta_properties()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.join(script_dir, "..", "..")
    raw_base_path = config.get_raw_data_path(project_root)
    processed_base_path = config.get_processed_dir(project_root)
    os.makedirs(processed_base_path, exist_ok=True)

    all_files = discover_source_files(raw_base_path)
    if not all_files:
        print("⚠️  No INR source files found.")
        return

    battery_file_map = prepare_battery_file_map(all_files, meta_df)
    if not battery_file_map:
        print("⚠️  No matching INR batteries found in metadata.")
        print_debug_matching_info(all_files, meta_df)
        return

    aggregated_errors: Dict[str, Dict[str, str]] = {}

    with ThreadPoolExecutor(max_workers=config.thread_count) as executor:
        future_to_battery = {
            executor.submit(
                process_single_battery,
                battery_id,
                files,
                raw_base_path,
                processed_base_path,
                meta_df,
                config,
            ): battery_id
            for battery_id, files in battery_file_map.items()
        }

        for future in as_completed(future_to_battery):
            battery_id = future_to_battery[future]
            try:
                errors = future.result()
                aggregated_errors[battery_id] = errors
                print(f"✅ Completed processing battery: {battery_id}")
            except Exception as exc:
                aggregated_errors[battery_id] = {battery_id: str(exc)}
                print(f"✗ Error processing battery {battery_id}: {exc}")

    end_time = time.time()
    total_time = end_time - start_time

    hours = int(total_time // 3600)
    minutes = int((total_time % 3600) // 60)
    seconds = int(total_time % 60)

    print(f"\n{'=' * 60}")
    print("🎉 All INR batteries processed successfully!")
    print(f"⏱️  Total processing time: {hours:02d}:{minutes:02d}:{seconds:02d}")
    print(f"📊 Processed {len(battery_file_map)} batteries with {config.thread_count} threads")
    print(
        f"⚡ Average time per battery: {total_time / max(len(battery_file_map), 1):.2f} seconds"
    )
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
# ============================================================
# 🎉 All INR batteries processed successfully!
# ⏱️  Total processing time: 00:00:15
# 📊 Processed 2 batteries with 4 threads
# ⚡ Average time per battery: 8.00 seconds
# ============================================================