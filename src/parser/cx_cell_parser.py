"""CX2 parser aligned with consolidated aggregation workflow."""

import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from utils.help_function import check_file_string, load_meta_properties


@dataclass
class ProcessingConfig:
    """Configuration for CX2 data processing."""

    raw_data_rel_path: str = os.path.join("assets", "raw", "CX2")
    processed_rel_root: str = os.path.join("assets", "processed")
    chemistry: str = "LCO"
    sample_points: int = 100
    thread_count: int = 20

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
    """Cell metadata container."""

    initial_capacity: float
    c_rate: float
    temperature: float
    vmax: float
    vmin: float


def extract_date(file_name: str) -> datetime.date:
    """Extract a date from CX2 filenames by scanning numeric suffixes."""

    name, _ = os.path.splitext(file_name)
    parts = name.split("_")

    numeric_parts: List[int] = []
    for part in reversed(parts):
        try:
            value = int(part)
        except ValueError:
            continue

        numeric_parts.insert(0, value)
        if len(numeric_parts) == 3:
            break

    if len(numeric_parts) < 3:
        raise ValueError(f"Cannot extract date from filename: {file_name}")

    month, day, year = numeric_parts[0], numeric_parts[1], numeric_parts[2]

    if not (1 <= month <= 12):
        raise ValueError(f"Invalid month extracted from {file_name}: {month}")
    if not (1 <= day <= 31):
        raise ValueError(f"Invalid day extracted from {file_name}: {day}")
    if year < 100:
        year = 2000 + year
    if not (2000 <= year <= 2099):
        raise ValueError(f"Invalid year extracted from {file_name}: {year}")

    return datetime(year, month, day).date()


def sort_files(file_names: List[str]) -> List[str]:
    """Sort filenames by the date encoded in their suffix."""

    dated_pairs = [(extract_date(name), name) for name in file_names]
    dated_pairs.sort(key=lambda item: item[0])
    return [name for _, name in dated_pairs]


def load_excel_file(file_path: str) -> pd.DataFrame:
    """Load an Excel file and return required columns as floats."""

    xls = pd.ExcelFile(file_path)
    chosen_sheet = None

    for sheet in xls.sheet_names:
        try:
            header = pd.read_excel(file_path, sheet_name=sheet, nrows=1)
        except Exception:
            continue

        columns = set(header.columns.astype(str))
        if {"Current(A)", "Voltage(V)", "Test_Time(s)"} <= columns:
            chosen_sheet = sheet
            break

    if chosen_sheet is None:
        chosen_sheet = xls.sheet_names[0]

    df = pd.read_excel(file_path, sheet_name=chosen_sheet)
    df.columns = [str(c).strip() for c in df.columns]

    required_cols = ["Current(A)", "Voltage(V)", "Test_Time(s)"]
    for col in required_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if not all(col in df.columns for col in required_cols):
        raise ValueError(
            f"Required columns not found in {file_path}. Columns: {df.columns.tolist()}"
        )

    return df[required_cols].dropna().reset_index(drop=True)


def load_text_file(file_path: str) -> pd.DataFrame:
    """Load a legacy text export and normalise units."""

    df = pd.read_csv(file_path, delimiter="\t")
    df.rename(
        columns={"Time": "Test_Time(s)", "mA": "Current(A)", "mV": "Voltage(V)"},
        inplace=True,
    )

    required_cols = ["Current(A)", "Voltage(V)", "Test_Time(s)"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing columns {missing_cols} in {file_path}")

    result = df[required_cols].dropna().reset_index(drop=True)
    result["Voltage(V)"] = result["Voltage(V)"] / 1000.0
    result["Current(A)"] = result["Current(A)"] / 1000.0
    result["Test_Time(s)"] = result["Test_Time(s)"] * 60.0
    return result


def compute_transition_indices(df: pd.DataFrame) -> Tuple[List[int], List[int]]:
    """Identify charge/discharge transition indices."""

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
    """Ensure charge and discharge indices alternate properly."""

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


def clip_and_tag_cycles(
    df: pd.DataFrame,
    charge_indices: List[int],
    discharge_indices: List[int],
    cell_initial_capacity: float,
    vmax: Optional[float] = None,
    vmin: Optional[float] = None,
    tolerance: float = 0.01,
) -> pd.DataFrame:
    """Restrict dataframe to valid cycles and compute cycle metrics."""

    df = df.iloc[charge_indices[0] : discharge_indices[-1] + 1].reset_index(drop=True)

    adjusted_charge = [idx - charge_indices[0] for idx in charge_indices]
    adjusted_discharge = [idx - charge_indices[0] for idx in discharge_indices]

    df["Cycle_Count"] = np.nan

    filtered: List[pd.DataFrame] = []

    for cycle_number, charge_start in enumerate(adjusted_charge, start=1):
        discharge_start = (
            adjusted_discharge[cycle_number - 1]
            if cycle_number - 1 < len(adjusted_discharge)
            else len(df)
        )

        next_charge_start = (
            adjusted_charge[cycle_number]
            if cycle_number < len(adjusted_charge)
            else len(df)
        )

        cycle_slice = df.iloc[charge_start:next_charge_start].copy()
        cycle_slice["Cycle_Count"] = cycle_number

        if vmax is not None and vmin is not None:
            cycle_slice = filter_voltage_range(cycle_slice, vmax, vmin, tolerance)

        if not cycle_slice.empty:
            filtered.append(cycle_slice)

    if filtered:
        df = pd.concat(filtered, ignore_index=True)
    else:
        for cycle_number, (start, end) in enumerate(
            zip(adjusted_charge, adjusted_charge[1:] + [len(df)]), start=1
        ):
            df.loc[start : max(end - 1, start), "Cycle_Count"] = cycle_number

    df["Delta_Time(s)"] = df["Test_Time(s)"].diff().fillna(0.0)
    df["Delta_Ah"] = np.abs(df["Current(A)"]) * df["Delta_Time(s)"] / 3600.0
    df["Ah_throughput"] = df["Delta_Ah"].cumsum()
    df["EFC"] = df["Ah_throughput"] / cell_initial_capacity

    return df


def filter_voltage_range(
    cycle_data: pd.DataFrame,
    vmax: float,
    vmin: float,
    tolerance: float = 0.01,
) -> pd.DataFrame:
    """Trim cycle data to exclude constant-voltage holds outside [vmin, vmax]."""

    if cycle_data.empty:
        return cycle_data

    voltage = cycle_data["Voltage(V)"].to_numpy()
    current = cycle_data["Current(A)"].to_numpy()

    discharge_start = next((i for i, value in enumerate(current) if value < -tolerance), None)
    vmax_index = next((i for i, value in enumerate(voltage) if value >= vmax - tolerance), None)
    vmin_index = None
    if discharge_start is not None:
        for idx in range(discharge_start, len(voltage)):
            if voltage[idx] <= vmin + tolerance:
                vmin_index = idx
                break

    if (
        discharge_start is None
        or vmax_index is None
        or vmin_index is None
        or vmax_index >= len(cycle_data)
        or vmin_index >= len(cycle_data)
    ):
        return cycle_data

    charge_portion = cycle_data.iloc[: vmax_index + 1]
    discharge_portion = cycle_data.iloc[discharge_start : vmin_index + 1]

    return pd.concat([charge_portion, discharge_portion], ignore_index=True)


def roll_cycle_counters(df: pd.DataFrame, existing: pd.DataFrame) -> pd.DataFrame:
    """Offset cycle metrics so consecutive files stack correctly."""

    if existing.empty:
        return df

    max_cycle = int(existing["Cycle_Count"].max())
    df["Cycle_Count"] = df["Cycle_Count"].astype(int) + max_cycle
    df["Ah_throughput"] = df["Ah_throughput"] + existing["Ah_throughput"].max()
    df["EFC"] = df["EFC"] + existing["EFC"].max()
    return df


def parse_source_file(
    file_path: str,
    metadata: CellMetadata,
) -> pd.DataFrame:
    """Parse a CX2 source file into tagged cycles."""

    extension = os.path.splitext(file_path)[1].lower()
    if extension == ".txt":
        df = load_text_file(file_path)
    else:
        df = load_excel_file(file_path)

    charge_indices, discharge_indices = compute_transition_indices(df)
    df = clip_and_tag_cycles(
        df,
        charge_indices,
        discharge_indices,
        metadata.initial_capacity,
        metadata.vmax,
        metadata.vmin,
    )
    df["C_rate"] = metadata.c_rate
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

    if positive_indices.size:
        charge_start = positive_indices[0]
        negative_after = negative_indices[negative_indices > charge_start]
        charge_end = negative_after[0] if negative_after.size else len(cycle_df)
        charge_segment = cycle_df.iloc[charge_start:charge_end].copy()

    if negative_indices.size:
        discharge_start = negative_indices[0]
        positive_after = positive_indices[positive_indices > discharge_start]
        discharge_end = positive_after[0] if positive_after.size else len(cycle_df)
        discharge_segment = cycle_df.iloc[discharge_start:discharge_end].copy()

    return charge_segment, discharge_segment


def prepare_resampled_outputs(
    agg_df: pd.DataFrame,
    cell_meta: CellMetadata,
    config: ProcessingConfig,
    battery_id: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Build charge and discharge aggregation tables."""

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

    for cycle in sorted(agg_df["Cycle_Count"].dropna().unique()):
        cycle_df = agg_df[agg_df["Cycle_Count"] == cycle].copy()
        if cycle_df.empty:
            continue

        cycle_df = cycle_df.sort_values("Test_Time(s)")
        charge_segment, discharge_segment = split_cycle_segments(cycle_df)

        for label, segment in (("charge", charge_segment), ("discharge", discharge_segment)):
            if segment.empty:
                continue

            resampled = resample_cycle_segment(segment, config.sample_points)
            if resampled.empty:
                continue

            resampled["battery_id"] = battery_id
            resampled["chemistry"] = config.chemistry
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


def list_valid_source_files(folder_path: str) -> List[str]:
    """Filter valid CX2 source files (size, suffix, and name quality)."""

    valid_extensions = {".xlsx", ".xls", ".txt"}
    candidates: List[str] = []

    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        if not os.path.isfile(file_path):
            continue

        if os.path.splitext(file_name)[1].lower() not in valid_extensions:
            continue

        if os.path.getsize(file_path) <= 200 * 1024:
            continue

        if check_file_string(file_name) == "bad":
            continue

        candidates.append(file_name)

    return candidates


def get_cell_metadata(meta_df: pd.DataFrame, cell_id: str) -> Optional[CellMetadata]:
    """Look up metadata for a given battery id."""

    cell_df = meta_df[meta_df["Battery_ID"].str.lower() == cell_id.lower()]
    if cell_df.empty:
        print(f"No metadata for {cell_id}. Available IDs: {meta_df['Battery_ID'].dropna().unique()}")
        return None

    return CellMetadata(
        initial_capacity=cell_df["Initial_Capacity_Ah"].values[0],
        c_rate=cell_df["C_rate"].values[0],
        temperature=cell_df["Temperature (K)"].values[0],
        vmax=cell_df["Max_Voltage"].values[0],
        vmin=cell_df["Min_Voltage"].values[0],
    )


def process_files_in_subfolder(
    folder_path: str, file_names: List[str], cell_meta: CellMetadata
) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """Parse and aggregate all files in a CX2 subfolder."""

    errors: Dict[str, str] = {}
    aggregated = pd.DataFrame()
    subfolder = os.path.basename(folder_path)

    sorted_files = sort_files(file_names)[::-1]
    print(f"📁 Processing {len(sorted_files)} files for {subfolder}")

    for index, file_name in enumerate(sorted_files, start=1):
        file_path = os.path.join(folder_path, file_name)
        try:
            df = parse_source_file(file_path, cell_meta)
            df = roll_cycle_counters(df, aggregated)
            aggregated = pd.concat([aggregated, df], ignore_index=True)
        except Exception as exc:
            errors[file_name] = str(exc)

        if index % 5 == 0 or index == len(sorted_files):
            progress = round(index / max(len(sorted_files), 1) * 100, 1)
            print(f"📁 {subfolder}: processed {index}/{len(sorted_files)} files ({progress}%)")

    return aggregated, errors


def save_processed_data(
    agg_df: pd.DataFrame,
    cell_id: str,
    cell_meta: CellMetadata,
    config: ProcessingConfig,
    output_dir: str,
) -> Tuple[str, str]:
    """Persist charge and discharge aggregates to disk."""

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
    """Persist per-battery error information."""

    if not error_dict:
        return

    os.makedirs(output_dir, exist_ok=True)
    error_df = pd.DataFrame(list(error_dict.items()), columns=["File_Name", "Error_Message"])
    error_path = os.path.join(output_dir, f"error_log_{cell_id}.csv")
    error_df.to_csv(error_path, index=False)
    print(f"📝 Saved error log: {error_path}")


def process_single_subfolder(
    subfolder: str,
    raw_base_path: str,
    processed_base_path: str,
    meta_df: pd.DataFrame,
    config: ProcessingConfig,
) -> None:
    """Process a single CX2 folder end-to-end."""

    folder_path = os.path.join(raw_base_path, subfolder)
    print(f"\nProcessing folder: {subfolder}")

    try:
        file_names = list_valid_source_files(folder_path)
        if not file_names:
            print(f"⚠️  No valid files found in {subfolder}")
            return

        cell_meta = get_cell_metadata(meta_df, subfolder)
        if cell_meta is None:
            return

        aggregated, errors = process_files_in_subfolder(folder_path, file_names, cell_meta)
        if aggregated.empty:
            print(f"No data aggregated for {subfolder}")
            save_error_log(errors, subfolder, os.path.join(processed_base_path, subfolder))
            return

        output_dir = os.path.join(processed_base_path, subfolder)
        save_processed_data(aggregated, subfolder, cell_meta, config, output_dir)
        save_error_log(errors, subfolder, output_dir)

    except Exception as exc:
        print(f"✗ Error processing {subfolder}: {exc}")


def get_cx2_subfolders(base_path: str) -> List[str]:
    """List CX2 subfolders located under the raw data root."""

    subfolders: List[str] = []
    for entry in os.listdir(base_path):
        full_path = os.path.join(base_path, entry)
        if os.path.isdir(full_path) and entry.upper().startswith("CX2_"):
            subfolders.append(entry)
    return subfolders


def main(config: Optional[ProcessingConfig] = None) -> None:
    """Entry point for multi-threaded CX2 dataset processing."""

    if config is None:
        config = ProcessingConfig()

    start_time = time.time()
    print(f"🚀 Starting CX2 battery data processing with {config.thread_count} threads...")

    meta_df = load_meta_properties()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.join(script_dir, "..", "..")
    raw_base_path = config.get_raw_data_path(project_root)
    processed_base_path = config.get_processed_dir(project_root)
    os.makedirs(processed_base_path, exist_ok=True)

    cx2_subfolders = get_cx2_subfolders(raw_base_path)
    print(f"📂 Found {len(cx2_subfolders)} CX2 batteries: {cx2_subfolders}")

    with ThreadPoolExecutor(max_workers=config.thread_count) as executor:
        future_to_subfolder = {
            executor.submit(
                process_single_subfolder,
                subfolder,
                raw_base_path,
                processed_base_path,
                meta_df,
                config,
            ): subfolder
            for subfolder in cx2_subfolders
        }

        for future in as_completed(future_to_subfolder):
            subfolder = future_to_subfolder[future]
            try:
                future.result()
                print(f"✅ Completed processing battery: {subfolder}")
            except Exception as exc:
                print(f"✗ Error processing subfolder {subfolder}: {exc}")

    end_time = time.time()
    total_time = end_time - start_time

    if cx2_subfolders:
        average_time = total_time / len(cx2_subfolders)
    else:
        average_time = 0.0

    hours = int(total_time // 3600)
    minutes = int((total_time % 3600) // 60)
    seconds = int(total_time % 60)

    print(f"\n{'=' * 60}")
    print("🎉 All CX2 subfolders processed successfully!")
    print(f"⏱️  Total processing time: {hours:02d}:{minutes:02d}:{seconds:02d}")
    print(f"📊 Processed {len(cx2_subfolders)} subfolders with {config.thread_count} threads")
    print(f"⚡ Average time per subfolder: {average_time:.2f} seconds")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
