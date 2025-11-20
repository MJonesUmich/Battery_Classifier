import os
import sys
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.io import loadmat

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from utils.voltage_clamp import clamp_voltage_column


@dataclass
class ProcessingConfig:
    """Configuration for Oxford data processing."""

    raw_data_rel_path: str = os.path.join("assets", "raw", "Oxford")
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


def extract_voltage_limits_from_mat_full(file_path, cell_number=1, percentile=99):
    """
    Extract voltage limits (vmax and vmin) from Oxford_Battery_Degradation_Dataset_1.mat format

    Parameters:
    - file_path: path to .mat file
    - cell_number: which cell to analyze (1-8)
    - percentile: percentile to use for determining voltage limits

    Returns:
    - vmax: maximum charge voltage
    - vmin: minimum discharge voltage
    """
    mat_data = loadmat(file_path)

    # Get the cell data
    cell_key = f"Cell{cell_number}"
    if cell_key not in mat_data:
        raise ValueError(f"Cell {cell_number} not found in mat file")

    cell_data = mat_data[cell_key]
    dtype_names = cell_data.dtype.names

    if dtype_names is None:
        raise ValueError(f"No cycle data found for {cell_key}")

    cycle_names = [name for name in dtype_names if name.startswith("cyc")]

    all_charge_voltages = []
    all_discharge_voltages = []

    # Collect voltage data from all cycles
    for cycle_name in cycle_names[:5]:  # Use first 5 cycles to determine limits
        try:
            cycle_data = cell_data[cycle_name][0, 0]

            # C1 charge voltages
            c1ch_data = cycle_data["C1ch"][0, 0]
            v_ch = c1ch_data["v"][0, 0].flatten()
            all_charge_voltages.extend(v_ch)

            # C1 discharge voltages
            c1dc_data = cycle_data["C1dc"][0, 0]
            v_dc = c1dc_data["v"][0, 0].flatten()
            all_discharge_voltages.extend(v_dc)

        except Exception:
            continue

    if len(all_charge_voltages) == 0 or len(all_discharge_voltages) == 0:
        raise ValueError(f"Could not extract voltage data from {cell_key}")

    # Get voltage limits using percentile
    vmax = np.percentile(all_charge_voltages, percentile)
    vmin = np.percentile(all_discharge_voltages, 100 - percentile)

    return vmax, vmin


def load_from_mat_full(file_path, cell_number=1):
    """
    Load data from Oxford_Battery_Degradation_Dataset_1.mat format
    Structure: Cell[1-8] -> cyc#### -> C1ch/C1dc/OCVch/OCVdc -> t, v, q, T

    Parameters:
    - file_path: path to the .mat file
    - cell_number: which cell to extract (1-8)

    Returns combined DataFrame for all cycles of the specified cell
    """
    mat_data = loadmat(file_path)

    # Get the cell data
    cell_key = f"Cell{cell_number}"
    if cell_key not in mat_data:
        raise ValueError(
            f"Cell {cell_number} not found in mat file. Available keys: {list(mat_data.keys())}"
        )

    cell_data = mat_data[cell_key]

    # Get all cycle names (cyc0100, cyc0200, etc.)
    dtype_names = cell_data.dtype.names
    if dtype_names is None:
        raise ValueError(f"No cycle data found for {cell_key}")

    cycle_names = [name for name in dtype_names if name.startswith("cyc")]
    cycle_names.sort()  # Sort to get chronological order

    all_dfs = []
    cycle_offset = 0.0

    for cycle_name in cycle_names:
        cycle_num = int(cycle_name.replace("cyc", ""))
        cycle_data = cell_data[cycle_name][0, 0]

        # Extract C1 charge and discharge (1C rate tests)
        # For current, we need to infer from the test type since it's not in the data
        try:
            # C1 charge
            c1ch_data = cycle_data["C1ch"][0, 0]
            t_ch_days = c1ch_data["t"][0, 0].flatten()
            v_ch = c1ch_data["v"][0, 0].flatten()
            i_ch = np.ones_like(t_ch_days) * 0.74  # 740 mA = 0.74 A

            # C1 discharge
            c1dc_data = cycle_data["C1dc"][0, 0]
            t_dc_days = c1dc_data["t"][0, 0].flatten()
            v_dc = c1dc_data["v"][0, 0].flatten()
            i_dc = np.ones_like(t_dc_days) * (-0.74)

            # Combine and sort by timestamp to ensure chronological order
            time_days = np.concatenate([t_ch_days, t_dc_days])
            voltage = np.concatenate([v_ch, v_dc])
            current = np.concatenate([i_ch, i_dc])

            order = np.argsort(time_days)
            time_days = time_days[order]
            voltage = voltage[order]
            current = current[order]

            # Convert MATLAB datenum (days) to seconds relative to cycle start
            time_seconds = (time_days - time_days[0]) * 24.0 * 3600.0 + cycle_offset

            # Update offset so next cycle starts after this one
            if time_seconds.size > 0:
                cycle_offset = float(time_seconds[-1]) + 1.0

            # Create DataFrame for this cycle
            cycle_df = pd.DataFrame(
                {
                    "Test_Time(s)": time_seconds,
                    "Voltage(V)": voltage,
                    "Current(A)": current,
                    "Characterization_Cycle": cycle_num,
                }
            )

            all_dfs.append(cycle_df)

        except Exception as e:
            print(f"Warning: Could not process {cycle_name}: {str(e)}")
            continue

    if len(all_dfs) == 0:
        raise ValueError(f"No valid cycle data found for {cell_key}")

    # Combine all cycles
    df = pd.concat(all_dfs, ignore_index=True)

    return df.dropna().reset_index(drop=True)


def get_indices(df):
    """
    Get charge and discharge indices for Oxford data.
    For Oxford data, we use the existing Characterization_Cycle information
    and detect charge/discharge transitions within each cycle.
    """
    I = df["Current(A)"]

    # Set threshold to avoid 0 current jitter
    thr = max(0.05 * np.nanmedian(np.abs(I[I != 0])), 1e-4)
    sign3 = np.zeros_like(I, dtype=int)
    sign3[I > thr] = 1
    sign3[I < -thr] = -1

    charge_indices, discharge_indices = [], []

    # Get unique cycles from Characterization_Cycle column
    if "Characterization_Cycle" in df.columns:
        unique_cycles = sorted(df["Characterization_Cycle"].unique())
        print(f"Found {len(unique_cycles)} cycles in Characterization_Cycle column")

        # For each cycle, find charge and discharge transitions
        for cycle in unique_cycles:
            cycle_mask = df["Characterization_Cycle"] == cycle
            cycle_indices = df[cycle_mask].index.tolist()

            if len(cycle_indices) == 0:
                continue

            # Find charge start (first positive current in this cycle)
            charge_start = None
            for idx in cycle_indices:
                if sign3[idx] == 1:
                    charge_start = idx
                    break

            # Find discharge start (first negative current in this cycle)
            discharge_start = None
            for idx in cycle_indices:
                if sign3[idx] == -1:
                    discharge_start = idx
                    break

            if charge_start is not None:
                charge_indices.append(charge_start)
            if discharge_start is not None:
                discharge_indices.append(discharge_start)
    else:
        # Fallback to original method if no Characterization_Cycle column
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

    # Ensure we have the same number of charge and discharge indices
    min_count = min(len(charge_indices), len(discharge_indices))
    if len(charge_indices) > min_count:
        charge_indices = charge_indices[:min_count]
    if len(discharge_indices) > min_count:
        discharge_indices = discharge_indices[:min_count]

    print(
        f"Detected {len(charge_indices)} charge indices and {len(discharge_indices)} discharge indices"
    )
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
    print("Indices alternate correctly.")
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
    # Downsample to just between first charge and last discharge
    df = df.iloc[charge_indices[0] : discharge_indices[-1] + 1].reset_index(drop=True)

    # Adjust charge_indices and discharge_indices to match the new DataFrame
    adjusted_charge_indices = [i - charge_indices[0] for i in charge_indices]
    adjusted_discharge_indices = [i - charge_indices[0] for i in discharge_indices]

    # Create a new column for tagging
    df["Cycle_Count"] = None

    # Process each cycle to filter out constant voltage holds
    filtered_cycles = []

    # Ensure we have the same number of charge and discharge indices
    min_cycles = min(len(adjusted_charge_indices), len(adjusted_discharge_indices))

    for i in range(min_cycles):
        charge_start = adjusted_charge_indices[i]
        discharge_start = adjusted_discharge_indices[i]

        # Determine the end of this cycle (start of next charge or end of data)
        if i + 1 < len(adjusted_charge_indices):
            cycle_end = adjusted_charge_indices[i + 1]
        else:
            cycle_end = len(df)

        # Extract cycle data from charge start to cycle end
        cycle_data = df.iloc[charge_start:cycle_end].copy()
        cycle_data["Cycle_Count"] = i + 1

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
        for i in range(min_cycles):
            charge_start = adjusted_charge_indices[i]
            if i + 1 < len(adjusted_charge_indices):
                cycle_end = adjusted_charge_indices[i + 1]
            else:
                cycle_end = len(df)
            df.loc[charge_start : cycle_end - 1, "Cycle_Count"] = i + 1

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
    excluding constant voltage holds (CV phase).

    Strategy:
    - For charge: Keep data from start until voltage reaches vmax, but exclude CV hold
    - For discharge: Keep data from discharge start until voltage reaches vmin
    - CV hold detection: voltage stays near vmax/vmin while current decreases
    - Strict filtering: only keep data within vmin-vmax range
    """
    if len(cycle_data) == 0:
        return cycle_data

    voltage = cycle_data["Voltage(V)"].values
    current = cycle_data["Current(A)"].values

    # Find discharge start (first significant negative current)
    discharge_start = None
    current_threshold = (
        0.05 * np.abs(current[np.abs(current) > 1e-6]).max()
        if np.any(np.abs(current) > 1e-6)
        else 0.01
    )

    for i, c in enumerate(current):
        if c < -current_threshold:
            discharge_start = i
            break

    if discharge_start is None:
        # No discharge found, only charge data
        discharge_start = len(voltage)

    # Process charge phase (before discharge)
    charge_end_idx = None
    if discharge_start > 0:
        charge_voltage = voltage[:discharge_start]
        charge_current = current[:discharge_start]

        # Find where voltage first reaches vmax
        vmax_first = None
        for i, v in enumerate(charge_voltage):
            if v >= vmax - tolerance:
                vmax_first = i
                break

        if vmax_first is not None:
            # Check if there's a CV hold after reaching vmax
            # CV hold: voltage stays near vmax, current decreases significantly
            cv_hold_start = None
            for i in range(vmax_first, len(charge_voltage)):
                if i + 5 < len(charge_voltage):  # Need at least 5 points to detect CV
                    # Check if voltage is stable near vmax
                    voltage_stable = np.all(
                        np.abs(charge_voltage[i : i + 5] - vmax) < tolerance
                    )
                    # Check if current is decreasing (CV characteristic)
                    current_decreasing = (
                        charge_current[i] < 0.5 * charge_current[vmax_first]
                        if charge_current[vmax_first] > 0
                        else False
                    )

                    if voltage_stable and current_decreasing:
                        cv_hold_start = i
                        break

            # Set charge end: before CV hold starts, or at vmax if no clear CV hold
            charge_end_idx = (
                cv_hold_start if cv_hold_start is not None else vmax_first + 1
            )
        else:
            # vmax not reached, use all charge data
            charge_end_idx = discharge_start
    else:
        charge_end_idx = 0

    # Process discharge phase
    discharge_end_idx = None
    if discharge_start < len(voltage):
        discharge_voltage = voltage[discharge_start:]

        # Find where voltage first reaches vmin
        vmin_first = None
        for i, v in enumerate(discharge_voltage):
            if v <= vmin + tolerance:
                vmin_first = discharge_start + i
                break

        if vmin_first is not None:
            discharge_end_idx = vmin_first + 1
        else:
            # vmin not reached, use all discharge data
            discharge_end_idx = len(voltage)
    else:
        discharge_end_idx = len(voltage)

    # Combine charge and discharge portions
    filtered_indices = []

    # Add charge portion (up to CV hold or vmax)
    if charge_end_idx > 0:
        filtered_indices.extend(range(0, charge_end_idx))

    # Add discharge portion (from discharge start to vmin)
    if discharge_start < discharge_end_idx:
        filtered_indices.extend(range(discharge_start, discharge_end_idx))

    if len(filtered_indices) > 0:
        filtered_data = cycle_data.iloc[filtered_indices].reset_index(drop=True)

        # Apply strict voltage range filtering
        voltage_mask = (filtered_data["Voltage(V)"] >= vmin) & (
            filtered_data["Voltage(V)"] <= vmax
        )
        final_filtered = filtered_data[voltage_mask].reset_index(drop=True)

        return final_filtered
    else:
        # Fallback: return original data if filtering fails
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
    file_path, cell_initial_capacity, cell_C_rate, vmax=None, vmin=None, cell_number=1
):
    """
    Parse a single Oxford file and return processed DataFrame.

    Parameters:
    - file_path: path to the data file
    - cell_initial_capacity: initial capacity in Ah
    - cell_C_rate: C-rate value
    - vmax, vmin: voltage limits for filtering
    - cell_number: which cell to extract (1-8)
    """
    df = load_from_mat_full(file_path, cell_number)

    charge_indices, discharge_indices = get_indices(df)
    df = scrub_and_tag(
        df, charge_indices, discharge_indices, cell_initial_capacity, vmax, vmin
    )
    df["Cycle_Count"] = pd.to_numeric(df["Cycle_Count"], errors="coerce")
    df = df.dropna(subset=["Cycle_Count"]).reset_index(drop=True)
    df["Cycle_Count"] = df["Cycle_Count"].astype(int)
    df["C_rate"] = cell_C_rate
    return df


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
    """Split a cycle into charge and discharge segments by current sign."""
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
        discharge_end = (
            pos_after_discharge[0] if pos_after_discharge.size else len(cycle_df)
        )
        discharge_segment = cycle_df.iloc[discharge_start:discharge_end].copy()

    return charge_segment, discharge_segment


def prepare_cycle_segment(
    segment_df: pd.DataFrame, min_samples: int
) -> Optional[pd.DataFrame]:
    """Sanitize a cycle segment and ensure it meets minimum sample requirements."""

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

    if sanitized["Test_Time(s)"].iloc[-1] <= sanitized["Test_Time(s)"].iloc[0]:
        return None

    sanitized = sanitized.reset_index(drop=True)
    clamp_voltage_column(sanitized, column="Voltage(V)")
    return sanitized


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


def parse_mat_file(
    file_path, cell_initial_capacity, cell_C_rate, vmax=None, vmin=None, cell_number=1
):
    """
    Wrapper function specifically for parsing Oxford .mat files

    Parameters:
    - file_path: path to .mat file
    - cell_initial_capacity: initial capacity in Ah
    - cell_C_rate: C-rate value
    - vmax, vmin: voltage limits for filtering
    - cell_number: which cell to extract (1-8)
    """
    return parse_file(
        file_path, cell_initial_capacity, cell_C_rate, vmax, vmin, cell_number
    )


def get_cell_metadata(cell_number: int, vmax: float, vmin: float) -> CellMetadata:
    """Get cell metadata for Oxford cells."""
    # Oxford battery specifications from readme
    # Kokam CO LTD, SLPB533459H4, 740mAh = 0.74Ah
    return CellMetadata(
        initial_capacity=0.74,  # Ah
        c_rate=1.0,  # 1C rate (740mA)
        temperature=273.15 + 40,  # 40°C in Kelvin
        vmax=vmax,
        vmin=vmin,
    )


def process_single_cell(
    file_path: str,
    cell_number: int,
    cell_meta: CellMetadata,
) -> Tuple[pd.DataFrame, Optional[str]]:
    """Process a single Oxford cell."""
    error_msg = None

    try:
        print(f"\nProcessing Cell {cell_number}...")

        # Parse the mat file
        df = parse_mat_file(
            file_path,
            cell_meta.initial_capacity,
            cell_meta.c_rate,
            cell_meta.vmax,
            cell_meta.vmin,
            cell_number=cell_number,
        )

        return df, error_msg

    except Exception as e:
        error_msg = str(e)
        print(f"Error processing Cell {cell_number}: {error_msg}")
        return pd.DataFrame(), error_msg


def save_processed_data(
    agg_df: pd.DataFrame,
    battery_id: str,
    cell_meta: CellMetadata,
    config: ProcessingConfig,
    output_dir: str,
) -> Tuple[str, str]:
    """Save resampled charge and discharge data to CSV files."""
    charge_df, discharge_df = prepare_resampled_outputs(
        agg_df, cell_meta, config, battery_id
    )

    os.makedirs(output_dir, exist_ok=True)

    charge_path = os.path.join(output_dir, f"{battery_id}_charge_aggregated_data.csv")
    discharge_path = os.path.join(
        output_dir, f"{battery_id}_discharge_aggregated_data.csv"
    )

    charge_df.to_csv(charge_path, index=False)
    discharge_df.to_csv(discharge_path, index=False)

    print(f"[INFO] Saved charge CSV: {charge_path}")
    print(f"[INFO] Saved discharge CSV: {discharge_path}")

    return charge_path, discharge_path


def save_error_log(
    error_dict: Dict[str, str], battery_id: str, output_dir: str
) -> None:
    """Save error log for a specific battery."""
    if not error_dict:
        return

    os.makedirs(output_dir, exist_ok=True)

    error_df = pd.DataFrame(list(error_dict.items()), columns=["Context", "Error_Message"])
    error_log_path = os.path.join(output_dir, f"error_log_{battery_id}.csv")
    error_df.to_csv(error_log_path, index=False)
    print(f"[INFO] Saved error log: {error_log_path}")


def main(config: Optional[ProcessingConfig] = None) -> None:
    """Main function to process Oxford battery data."""
    if config is None:
        config = ProcessingConfig()

    # Start timing
    start_time = time.time()
    print("[INFO] Starting Oxford battery data processing...")

    # Setup paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.join(script_dir, "..", "..")
    raw_base_path = config.get_raw_data_path(project_root)
    processed_base_path = config.get_processed_dir(project_root)
    os.makedirs(processed_base_path, exist_ok=True)

    file_path = os.path.join(
        raw_base_path, "Oxford_Battery_Degradation_Dataset_1.mat"
    )

    print(f"[INFO] Processing file: {file_path}")
    print("[INFO] Dataset: Oxford Battery Degradation Dataset (LCO - Kokam 740mAh)")

    # Extract voltage limits from the data
    print("[INFO] Extracting voltage limits from data...")
    try:
        vmax, vmin = extract_voltage_limits_from_mat_full(
            file_path, cell_number=1, percentile=99
        )
        print(f"[INFO] Extracted voltage limits: vmax={vmax:.4f}V, vmin={vmin:.4f}V")
    except Exception as e:
        vmax, vmin = 4.2, 2.9
        print(f"[WARN] Using default voltage limits: vmax={vmax}V, vmin={vmin}V ({e})")

    # Process cells 1-8
    cells_to_process = range(1, 9)
    print(f"[INFO] Processing {len(cells_to_process)} cells")

    for cell_number in cells_to_process:
        battery_id = f"Oxford_Cell{cell_number}"
        output_dir = config.get_processed_dir(project_root, battery_id)
        cell_meta = get_cell_metadata(cell_number, vmax, vmin)

        df, error_msg = process_single_cell(file_path, cell_number, cell_meta)

        if error_msg:
            save_error_log({"parse": error_msg}, battery_id, output_dir)
            continue

        if df.empty:
            save_error_log({"parse": "No data extracted"}, battery_id, output_dir)
            continue

        save_processed_data(df, battery_id, cell_meta, config, output_dir)

    # Calculate and display total processing time
    end_time = time.time()
    total_time = end_time - start_time

    # Format time display
    hours = int(total_time // 3600)
    minutes = int((total_time % 3600) // 60)
    seconds = int(total_time % 60)

    print(f"\n{'='*60}")
    print("[INFO] Oxford cells processed!")
    print(f"[INFO] Total processing time: {hours:02d}:{minutes:02d}:{seconds:02d}")
    print(f"[INFO] Cells processed: {len(cells_to_process)}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
# ============================================================
# [INFO] Oxford cells processed!
# [INFO] Total processing time: 00:00:25
# [INFO] Cells processed: 8
# ============================================================