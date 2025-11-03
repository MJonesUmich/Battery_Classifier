import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from utils.help_function import load_meta_properties


@dataclass
class ProcessingConfig:
    """Configuration class for CS2 data processing."""

    raw_data_rel_path: str = os.path.join("assets", "raw", "CS2")
    processed_rel_root: str = os.path.join("assets", "processed")
    chemistry: str = "LCO"
    sample_points: int = 100
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
    """Cell metadata container."""

    initial_capacity: float
    c_rate: float
    temperature: float
    vmax: float
    vmin: float


MIN_SEGMENT_SAMPLE_COUNT = 100


def extract_date(file_name, orientation="last"):
    # Extract MM, DD, YR from the file name
    name, extension = os.path.splitext(file_name)
    parts = name.split("_")

    # Find the last 3 numeric parts that could be valid date components
    # We need to be more careful about which numeric parts to use
    numeric_parts = []
    for i in range(len(parts) - 1, -1, -1):
        try:
            val = int(parts[i])
            # Only consider reasonable values for date components
            if 1 <= val <= 31:  # Day range
                numeric_parts.insert(0, val)
            elif 1 <= val <= 12:  # Month range
                numeric_parts.insert(0, val)
            elif 10 <= val <= 99:  # Year range (10-99, will be converted to 2010-2099)
                numeric_parts.insert(0, val)
            elif 2000 <= val <= 2099:  # Full year range
                numeric_parts.insert(0, val)

            if len(numeric_parts) == 3:
                break
        except ValueError:
            # Skip non-numeric parts (like "self discharge test")
            continue

    if len(numeric_parts) < 3:
        raise ValueError(f"Cannot extract date from filename: {file_name}")

    # The last 3 numeric parts should be: month, day, year
    year, day, month = numeric_parts[-1], numeric_parts[-2], numeric_parts[-3]

    # Fix year: if it's a 2-digit year, assume it's 20xx
    if year < 100:
        year = 2000 + year

    # Validate the date components
    if not (1 <= month <= 12):
        raise ValueError(f"Invalid month: {month}")
    if not (1 <= day <= 31):
        raise ValueError(f"Invalid day: {day}")
    if not (2000 <= year <= 2099):
        raise ValueError(f"Invalid year: {year}")

    print(month, day, year)
    return datetime(year, month, day).date()


def sort_files(file_names, orientation="last"):

    file_dates = []

    # Extract dates and sort files
    for file_name in file_names:
        file_date = extract_date(file_name, orientation)
        file_dates.append(file_date)

    # Sort files by their corresponding dates
    sorted_files = [file for _, file in sorted(zip(file_dates, file_names))]

    return sorted_files, file_dates


def load_file(file_path):

    # Read Excel (choose the sheet including Current/Voltage)
    xls = pd.ExcelFile(file_path)
    chosen = None

    # Look for sheets that contain the required columns
    for s in xls.sheet_names:
        try:
            # Read just the header to check columns
            cols = set(
                pd.read_excel(file_path, sheet_name=s, nrows=1).columns.astype(str)
            )
            if {"Current(A)", "Voltage(V)", "Test_Time(s)"} <= cols:
                chosen = s
                break
        except Exception as e:
            # Skip sheets that can't be read
            continue

    if chosen is None:
        # If no suitable sheet found, try the first sheet
        chosen = xls.sheet_names[0]

    df = pd.read_excel(file_path, sheet_name=chosen)
    df.columns = [str(c).strip() for c in df.columns]

    # Get the desired columns out:
    quant_cols = ["Current(A)", "Voltage(V)", "Test_Time(s)"]
    # In the dataframe force these columns to be float
    for col in quant_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Only keep the desired columns that exist
    available_cols = [col for col in quant_cols if col in df.columns]
    if len(available_cols) < 3:
        raise ValueError(
            f"Required columns not found. Available columns: {df.columns.tolist()}"
        )

    df = df[available_cols].dropna().reset_index(drop=True)
    return df


def load_from_text_file(file_path):
    # Load data from a text file
    df = pd.read_csv(file_path, delimiter="\t")
    # rename columns:
    df.rename(
        columns={"Time": "Test_Time(s)", "mA": "Current(A)", "mV": "Voltage(V)"},
        inplace=True,
    )
    desired_cols = ["Current(A)", "Voltage(V)", "Test_Time(s)"]
    df = df[desired_cols].dropna().reset_index(drop=True)
    df["Voltage(V)"] = df["Voltage(V)"] / 1000  # Convert mV to V
    df["Current(A)"] = df["Current(A)"] / 1000  # Convert mA to A
    df["Test_Time(s)"] = df["Test_Time(s)"] * 60.0  # Source files store time in minutes
    return df


def get_indices(df):

    I = df["Current(A)"]

    # Set threshold to avoid 0 current jitter
    thr = max(0.05 * np.nanmedian(np.abs(I[I != 0])), 1e-4)
    sign3 = np.zeros_like(I, dtype=int)
    sign3[I > thr] = 1
    sign3[I < -thr] = -1

    charge_indices, discharge_indices = [], []
    prev = 0
    for i, s in enumerate(sign3):
        if s == 0:
            continue
        if prev == 0:
            prev = s
            continue
        if s != prev:
            if s == 1:
                charge_indices.append(i)
            elif s == -1:
                discharge_indices.append(i)
            prev = s

    complexity, expected_order = check_indices(charge_indices, discharge_indices)
    if complexity == "High":
        # Skip this file
        raise ValueError("Indices do not alternate correctly.")
    else:
        if expected_order[0] == "discharge" and len(discharge_indices) > len(
            charge_indices
        ):
            discharge_indices = discharge_indices[:-1]
        elif expected_order[0] == "charge" and len(charge_indices) > len(
            discharge_indices
        ):
            charge_indices = charge_indices[:-1]

    assert len(charge_indices) == len(discharge_indices)
    return charge_indices, discharge_indices


def check_indices(charge_indices, discharge_indices):

    # Determine which starts first
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

    # Sort combined list by index
    combined.sort(key=lambda x: x[0])  # Sort by index

    # Check for alternating order
    for i, (_, label) in enumerate(combined):
        if label != expected_order[i % 2]:
            print(
                f"Error: Indices do not alternate correctly at position {i} ({combined[i - 1]} followed by {combined[i]})"
            )
            complexity = "High"
            return complexity, expected_order

    complexity = "Low"
    return complexity, expected_order


def scrub_and_tag(
    df,
    charge_indices,
    discharge_indices,
    cell_initial_capacity,
    vmax=None,
    vmin=None,
    tolerance=0.01,
):
    # Downsample to just between charge cycles
    df = df.iloc[charge_indices[0] : discharge_indices[-1] + 1].reset_index(drop=True)

    # Adjust charge_indices and discharge_indices to match the new DataFrame
    adjusted_charge_indices = [i - charge_indices[0] for i in charge_indices]
    adjusted_discharge_indices = [i - charge_indices[0] for i in discharge_indices]

    # Create a new column for tagging
    df["Cycle_Count"] = None

    # Process each cycle to filter out constant voltage holds
    filtered_cycles = []

    for i, charge_start in enumerate(adjusted_charge_indices, start=1):
        # Get the discharge start for this cycle
        if i <= len(adjusted_discharge_indices):
            discharge_start = adjusted_discharge_indices[i - 1]
        else:
            discharge_start = len(df)

        # Get the end of this cycle (start of next charge or end of dataframe)
        if i < len(adjusted_charge_indices):
            cycle_end = adjusted_charge_indices[i]
        else:
            cycle_end = len(df)

        # Extract cycle data (from charge start to next charge start)
        cycle_data = df.iloc[charge_start:cycle_end].copy()
        cycle_data["Cycle_Count"] = i

        # Filter out constant voltage holds if vmax and vmin are provided
        if vmax is not None and vmin is not None:
            cycle_data = filter_voltage_range(cycle_data, vmax, vmin, tolerance)

        if len(cycle_data) > 0:
            filtered_cycles.append(cycle_data)

    # Combine all filtered cycles
    if filtered_cycles:
        df = pd.concat(filtered_cycles, ignore_index=True)
    else:
        # Fallback to original method if no cycles were filtered
        for i, (start, end) in enumerate(
            zip(adjusted_charge_indices, adjusted_charge_indices[1:] + [len(df)]),
            start=1,
        ):
            df.loc[start : end - 1, "Cycle_Count"] = i

    # Coloumb count Ah throughput for each cycle
    df["Delta_Time(s)"] = df["Test_Time(s)"].diff().fillna(0)
    df["Delta_Ah"] = np.abs(df["Current(A)"]) * df["Delta_Time(s)"] / 3600
    df["Ah_throughput"] = df["Delta_Ah"].cumsum()

    # now calculate Equivalent Full Cycles (EFC) & Capacity Fade
    df["EFC"] = df["Ah_throughput"] / cell_initial_capacity
    return df


def filter_voltage_range(cycle_data, vmax, vmin, tolerance=0.01):
    """
    Filter cycle data to include only the voltage range between vmin and vmax,
    excluding constant voltage holds.
    """
    if len(cycle_data) == 0:
        return cycle_data

    # Find voltage range indices
    voltage = cycle_data["Voltage(V)"].values
    current = cycle_data["Current(A)"].values

    # Find discharge start (first negative current)
    discharge_start = None
    for i, c in enumerate(current):
        if c < -tolerance:
            discharge_start = i
            break

    # Find the first point where voltage reaches vmax (during charge)
    vmax_idx = None
    for i, v in enumerate(voltage):
        if v >= vmax - tolerance:
            vmax_idx = i
            break

    # Find the first point where voltage reaches vmin (during discharge, AFTER discharge starts)
    vmin_idx = None
    if discharge_start is not None:
        for i in range(discharge_start, len(voltage)):
            if voltage[i] <= vmin + tolerance:
                vmin_idx = i
                break

    # Filter data based on voltage range
    if vmax_idx is not None and vmin_idx is not None and discharge_start is not None:
        # Include charge portion up to vmax and discharge portion from start to vmin
        charge_portion = cycle_data.iloc[: vmax_idx + 1]
        discharge_portion = cycle_data.iloc[discharge_start : vmin_idx + 1]

        # Combine charge and discharge portions
        filtered_data = pd.concat(
            [charge_portion, discharge_portion], ignore_index=True
        )
        return filtered_data
    else:
        # If we can't find proper voltage bounds, return original data
        return cycle_data


def update_df(df, agg_df):
    if len(agg_df) == 0:
        return df
    else:
        max_cycle = agg_df["Cycle_Count"].max()
        df["Cycle_Count"] = df["Cycle_Count"].astype(int) + int(max_cycle)
        df["Ah_throughput"] = df["Ah_throughput"] + agg_df["Ah_throughput"].max()
        df["EFC"] = df["EFC"] + agg_df["EFC"].max()
        return df


def parse_file(
    file_path, cell_initial_capacity, cell_C_rate, method="excel", vmax=None, vmin=None
):
    if method == "excel":
        df = load_file(file_path)
    elif method == "text":
        df = load_from_text_file(file_path)

    charge_indices, discharge_indices = get_indices(df)
    df = scrub_and_tag(
        df, charge_indices, discharge_indices, cell_initial_capacity, vmax, vmin
    )
    df["C_rate"] = cell_C_rate
    return df


def compute_current_threshold(current_series: pd.Series) -> float:
    """Compute a threshold to separate charge and discharge modes."""
    non_zero = np.abs(current_series[current_series != 0])
    if len(non_zero) == 0:
        return 1e-4
    return max(0.05 * np.nanmedian(non_zero), 1e-4)


def resample_cycle_segment(
    segment_df: pd.DataFrame, sample_points: int
) -> pd.DataFrame:
    """Resample a cycle segment to a fixed number of points."""
    required_cols = ["Test_Time(s)", "Voltage(V)", "Current(A)"]
    if segment_df is None or segment_df.empty:
        return pd.DataFrame()

    segment = segment_df[required_cols].dropna()
    if segment.empty:
        return pd.DataFrame()

    segment = segment.drop_duplicates(subset=["Test_Time(s)"])
    segment = segment.sort_values("Test_Time(s)")
    if segment.empty:
        return pd.DataFrame()

    time_values = segment["Test_Time(s)"].to_numpy()
    relative_time = time_values - time_values[0]

    result = pd.DataFrame(
        {
            "Sample_Index": np.arange(sample_points, dtype=int),
            "Normalized_Time": np.linspace(0.0, 1.0, sample_points),
        }
    )

    if len(segment) == 1 or np.isclose(relative_time[-1], 0.0):
        result["Elapsed_Time(s)"] = np.zeros(sample_points)
        for column in ["Voltage(V)", "Current(A)"]:
            result[column] = np.full(sample_points, segment[column].iloc[0])
        return result

    normalized_time = relative_time / relative_time[-1]
    target_normalized = result["Normalized_Time"].to_numpy()

    result["Elapsed_Time(s)"] = np.interp(
        target_normalized, normalized_time, relative_time
    )
    for column in ["Voltage(V)", "Current(A)"]:
        result[column] = np.interp(
            target_normalized, normalized_time, segment[column].to_numpy()
        )

    return result


def split_cycle_segments(
    cycle_df: pd.DataFrame, tolerance: float = 1e-4
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split a cycle into charge and discharge segments based on current direction."""
    if cycle_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    current = cycle_df["Current(A)"].to_numpy()
    positive_indices = np.where(current > tolerance)[0]
    negative_indices = np.where(current < -tolerance)[0]

    charge_segment = pd.DataFrame()
    discharge_segment = pd.DataFrame()

    if positive_indices.size:
        charge_start = positive_indices[0]
        neg_after_charge = negative_indices[negative_indices > charge_start]
        charge_end = neg_after_charge[0] if neg_after_charge.size else len(cycle_df)
        charge_segment = cycle_df.iloc[charge_start:charge_end].copy()

    if negative_indices.size:
        discharge_start = negative_indices[0]
        pos_after_discharge = positive_indices[positive_indices > discharge_start]
        discharge_end = pos_after_discharge[0] if pos_after_discharge.size else len(cycle_df)
        discharge_segment = cycle_df.iloc[discharge_start:discharge_end].copy()

    return charge_segment, discharge_segment


def prepare_cycle_segment(
    segment_df: pd.DataFrame, min_samples: int
) -> Optional[pd.DataFrame]:
    """Sanitize a cycle segment and ensure it meets the minimum raw sample count."""
    required_cols = ["Test_Time(s)", "Voltage(V)", "Current(A)"]

    if segment_df is None or segment_df.empty:
        return None

    missing_cols = [col for col in required_cols if col not in segment_df.columns]
    if missing_cols:
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
    """Prepare resampled charge and discharge dataframes for export."""
    columns_order = [
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
        empty_df = pd.DataFrame(columns=columns_order)
        return empty_df.copy(), empty_df.copy()

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
        charge_segment_raw, discharge_segment_raw = split_cycle_segments(cycle_df)

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

        for label, resampled in (
            ("charge", charge_resampled),
            ("discharge", discharge_resampled),
        ):
            formatted = resampled.rename(
                columns={
                    "Sample_Index": "sample_index",
                    "Normalized_Time": "normalized_time",
                    "Elapsed_Time(s)": "elapsed_time_s",
                    "Voltage(V)": "voltage_v",
                    "Current(A)": "current_a",
                }
            ).copy()

            formatted["sample_index"] = formatted["sample_index"].astype(int)
            formatted["battery_id"] = battery_id
            formatted["chemistry"] = config.chemistry
            formatted["cycle_index"] = valid_cycle_count
            formatted["c_rate"] = cell_meta.c_rate
            formatted["temperature_k"] = cell_meta.temperature
            formatted = formatted[columns_order]

            if label == "charge":
                charge_segments.append(formatted)
            else:
                discharge_segments.append(formatted)

    empty_df = pd.DataFrame(columns=columns_order)

    if charge_segments:
        charge_df = pd.concat(charge_segments, ignore_index=True)
        charge_df = charge_df.sort_values(["cycle_index", "sample_index"])
        charge_df = charge_df.reset_index(drop=True)
    else:
        charge_df = empty_df.copy()

    if discharge_segments:
        discharge_df = pd.concat(discharge_segments, ignore_index=True)
        discharge_df = discharge_df.sort_values(["cycle_index", "sample_index"])
        discharge_df = discharge_df.reset_index(drop=True)
    else:
        discharge_df = empty_df.copy()

    return charge_df, discharge_df


def get_cs2_subfolders(base_path):
    """Get all CS2 subfolders from the base path."""
    subfolders = [
        f
        for f in os.listdir(base_path)
        if os.path.isdir(os.path.join(base_path, f)) and f.startswith("CS2_")
    ]
    return subfolders


def get_cell_metadata(meta_df: pd.DataFrame, cell_id: str) -> Optional[CellMetadata]:
    """Get cell metadata for a given battery ID."""
    cell_df = meta_df[meta_df["Battery_ID"].str.lower() == str.lower(cell_id)]

    if len(cell_df) == 0:
        print(f"Available Battery_IDs: {meta_df['Battery_ID'].dropna().unique()}")
        print(f"No metadata found for battery ID: {cell_id}")
        return None

    return CellMetadata(
        initial_capacity=cell_df["Initial_Capacity_Ah"].values[0],
        c_rate=cell_df["C_rate"].values[0],
        temperature=cell_df["Temperature (K)"].values[0],
        vmax=cell_df["Max_Voltage"].values[0],
        vmin=cell_df["Min_Voltage"].values[0],
    )


def process_files_in_subfolder(
    folder_path: str, sorted_files: List[str], cell_meta: CellMetadata
) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """Process all files in a subfolder and return aggregated data."""
    error_dict = {}
    agg_df = pd.DataFrame()

    # Extract subfolder name from folder_path
    subfolder = os.path.basename(folder_path)

    print(f"📁 Processing {len(sorted_files)} files for {subfolder}")

    for i_count, file_name in enumerate(sorted_files):
        try:
            file_path = os.path.join(folder_path, file_name)
            method = "text" if file_name.endswith(".txt") else "excel"

            df = parse_file(
                file_path,
                cell_meta.initial_capacity,
                cell_meta.c_rate,
                method,
                cell_meta.vmax,
                cell_meta.vmin,
            )
            df = update_df(df, agg_df)
            agg_df = pd.concat([agg_df, df], ignore_index=True)

        except Exception as e:
            error_dict[file_name] = str(e)

        if (i_count + 1) % 5 == 0 or i_count == len(
            sorted_files
        ) - 1:  # Show progress every 5 files
            print(
                f"📁 {subfolder}: Processed {i_count + 1}/{len(sorted_files)} files ({round((i_count+1)/len(sorted_files)*100,1)}%)"
            )

    return agg_df, error_dict


def save_processed_data(
    agg_df: pd.DataFrame,
    cell_id: str,
    cell_meta: CellMetadata,
    config: ProcessingConfig,
    output_dir: str,
) -> Tuple[str, str]:
    """Save resampled charge and discharge data to CSV files."""
    charge_df, discharge_df = prepare_resampled_outputs(
        agg_df, cell_meta, config, cell_id
    )

    os.makedirs(output_dir, exist_ok=True)

    charge_path = os.path.join(output_dir, f"{cell_id}_charge_aggregated_data.csv")
    discharge_path = os.path.join(
        output_dir, f"{cell_id}_discharge_aggregated_data.csv"
    )

    charge_df.to_csv(charge_path, index=False)
    discharge_df.to_csv(discharge_path, index=False)

    print(f"💾 Saved charge CSV: {charge_path}")
    print(f"💾 Saved discharge CSV: {discharge_path}")

    return charge_path, discharge_path


def save_error_log(error_dict: Dict[str, str], cell_id: str, output_dir: str) -> None:
    """Save error log for the subfolder."""
    if not error_dict:
        return

    os.makedirs(output_dir, exist_ok=True)

    error_df = pd.DataFrame(
        list(error_dict.items()), columns=["File_Name", "Error_Message"]
    )
    error_log_path = os.path.join(output_dir, f"error_log_{cell_id}.csv")
    error_df.to_csv(error_log_path, index=False)
    print(f"📝 Saved error log: {error_log_path}")


def process_single_subfolder(
    subfolder: str,
    raw_base_path: str,
    processed_base_path: str,
    meta_df: pd.DataFrame,
    config: ProcessingConfig,
) -> None:
    """Process a single subfolder completely."""
    folder_path = os.path.join(raw_base_path, subfolder)
    print(f"\nProcessing folder: {subfolder}")

    try:
        # Get file list and sort
        file_names = os.listdir(folder_path)
        sorted_files, _ = sort_files(file_names, orientation="last")
        sorted_files = sorted_files[::-1]

        # Get cell metadata
        cell_meta = get_cell_metadata(meta_df, subfolder)
        if cell_meta is None:
            return

        # Process all files
        agg_df, error_dict = process_files_in_subfolder(
            folder_path, sorted_files, cell_meta
        )

        if len(agg_df) == 0:
            print(f"No data processed for {subfolder}")
            return

        # Save processed data
        output_dir = os.path.join(processed_base_path, subfolder)
        save_processed_data(agg_df, subfolder, cell_meta, config, output_dir)

        # Save error log
        save_error_log(error_dict, subfolder, output_dir)

    except Exception as e:
        print(f"Error processing {subfolder}: {str(e)}")


def main(config: Optional[ProcessingConfig] = None) -> None:
    """Main function to process all CS2 subfolders with multi-threading."""
    if config is None:
        config = ProcessingConfig()

    # Start timing
    start_time = time.time()
    print("🚀 Starting CS2 battery data processing with 20 threads...")

    # Load metadata
    meta_df = load_meta_properties()

    # Setup paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.join(script_dir, "..", "..")
    raw_cs2_base_path = config.get_raw_data_path(project_root)
    processed_cs2_base_path = config.get_processed_dir(project_root)
    os.makedirs(processed_cs2_base_path, exist_ok=True)

    # Get subfolders
    cs2_subfolders = get_cs2_subfolders(raw_cs2_base_path)
    print(f"📂 Found {len(cs2_subfolders)} CS2 batteries: {cs2_subfolders}")

    # Process subfolders with 20 threads (increased from 10)
    with ThreadPoolExecutor(max_workers=20) as executor:
        # Submit all subfolder processing tasks
        future_to_subfolder = {
            executor.submit(
                process_single_subfolder,
                subfolder,
                raw_cs2_base_path,
                processed_cs2_base_path,
                meta_df,
                config,
            ): subfolder
            for subfolder in cs2_subfolders
        }

        # Wait for all subfolders to complete
        for future in as_completed(future_to_subfolder):
            subfolder = future_to_subfolder[future]
            try:
                future.result()
                print(f"✅ Completed processing battery: {subfolder}")
            except Exception as e:
                print(f"✗ Error processing subfolder {subfolder}: {str(e)}")

    # Calculate and display total processing time
    end_time = time.time()
    total_time = end_time - start_time

    # Format time display
    hours = int(total_time // 3600)
    minutes = int((total_time % 3600) // 60)
    seconds = int(total_time % 60)

    print(f"\n{'='*60}")
    print(f"🎉 All CS2 subfolders processed successfully!")
    print(f"⏱️  Total processing time: {hours:02d}:{minutes:02d}:{seconds:02d}")
    print(f"📊 Processed {len(cs2_subfolders)} subfolders with 20 threads")
    print(
        f"⚡ Average time per subfolder: {total_time/len(cs2_subfolders):.2f} seconds"
    )
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
# ============================================================
# 🎉 All CS2 subfolders processed successfully!
# ⏱️  Total processing time: 00:09:15
# 📊 Processed 10 subfolders with 20 threads
# ⚡ Average time per subfolder: 55.54 seconds
# ============================================================
