import os
import sys
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")  # Use non-interactive backend for multi-threading
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.io import loadmat

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from utils.help_function import load_meta_properties


@dataclass
class ProcessingConfig:
    """Configuration class for Oxford data processing."""

    base_data_path: str = "assets/raw_data/Oxford"
    output_data_path: str = "processed_datasets/LCO"
    output_images_path: str = "processed_images/LCO"
    required_columns: List[str] = None

    def __post_init__(self):
        if self.required_columns is None:
            self.required_columns = [
                "Current(A)",
                "Voltage(V)",
                "Test_Time(s)",
                "Cycle_Count",
                "Delta_Time(s)",
                "Delta_Ah",
                "Ah_throughput",
                "EFC",
                "C_rate",
            ]


@dataclass
class CellMetadata:
    """Cell metadata container."""

    initial_capacity: float
    c_rate: float
    temperature: float
    vmax: float
    vmin: float


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

    for cycle_name in cycle_names:
        cycle_num = int(cycle_name.replace("cyc", ""))
        cycle_data = cell_data[cycle_name][0, 0]

        # Extract C1 charge and discharge (1C rate tests)
        # For current, we need to infer from the test type since it's not in the data
        try:
            # C1 charge
            c1ch_data = cycle_data["C1ch"][0, 0]
            t_ch = c1ch_data["t"][0, 0].flatten()
            v_ch = c1ch_data["v"][0, 0].flatten()
            # For 1C charge, current = 740mA = 0.74A (based on readme)
            i_ch = np.ones_like(t_ch) * 0.74

            # C1 discharge
            c1dc_data = cycle_data["C1dc"][0, 0]
            t_dc = c1dc_data["t"][0, 0].flatten()
            v_dc = c1dc_data["v"][0, 0].flatten()
            # For 1C discharge, current = -740mA = -0.74A
            i_dc = np.ones_like(t_dc) * (-0.74)

            # Adjust discharge time to continue from charge
            if len(t_ch) > 0:
                t_dc = t_dc + t_ch[-1]

            # Combine charge and discharge
            time = np.concatenate([t_ch, t_dc])
            voltage = np.concatenate([v_ch, v_dc])
            current = np.concatenate([i_ch, i_dc])

            # Create DataFrame for this cycle
            cycle_df = pd.DataFrame(
                {
                    "Test_Time(s)": time,
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
    Parse Oxford .mat file and process battery data

    Parameters:
    - file_path: path to .mat file
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
    df["C_rate"] = cell_C_rate
    return df


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


def generate_figures(
    df,
    vmax,
    vmin,
    c_rate,
    temperature,
    battery_ID,
    output_dir,
    tolerance=0.01,
    one_fig_only=False,
):
    # Ensure output directory exists (directly use output_dir, no extra 'images' folder)
    os.makedirs(output_dir, exist_ok=True)

    unique_cycles = df["Cycle_Count"].unique()
    for i, cycle in enumerate(unique_cycles):
        cycle_df = df[df["Cycle_Count"] == cycle].reset_index(drop=True)

        # find where voltage first hits vmax and vmin, and where first discharge occurs
        try:
            vmax_idx = cycle_df[cycle_df["Voltage(V)"] >= vmax - tolerance].index[0]
        except IndexError:
            print(
                f"Warning: No voltage >= {vmax - tolerance} found in cycle {cycle}, using maximum voltage point..."
            )
            vmax_idx = cycle_df["Voltage(V)"].idxmax()

        try:
            vmin_idx = cycle_df[cycle_df["Voltage(V)"] <= vmin + tolerance].index[0]
        except IndexError:
            print(
                f"Warning: No voltage <= {vmin + tolerance} found in cycle {cycle}, using minimum voltage point..."
            )
            vmin_idx = cycle_df["Voltage(V)"].idxmin()

        try:
            disch_start = cycle_df[cycle_df["Current(A)"] < 0 - tolerance].index[0]
        except IndexError:
            print(f"Warning: No discharge current found in cycle {cycle}, skipping...")
            continue

        # clip data to initial until Vmax, then from discharge start to Vmin
        charge_cycle_df = cycle_df.loc[0:vmax_idx].copy()

        # Find vmin_idx after discharge starts (not before)
        discharge_portion = cycle_df.loc[disch_start:]
        vmin_reached = False
        try:
            vmin_idx_adjusted = discharge_portion[
                discharge_portion["Voltage(V)"] <= vmin + tolerance
            ].index[0]
            vmin_reached = True
        except IndexError:
            print(
                f"Warning: No vmin reached after discharge in cycle {cycle}, skipping discharge plot..."
            )
            vmin_reached = False

        # Only create discharge data if vmin was actually reached
        if vmin_reached:
            discharge_cycle_df = cycle_df.loc[disch_start:vmin_idx_adjusted].copy()
        else:
            discharge_cycle_df = pd.DataFrame()  # Empty dataframe

        # Check if we have valid charge data
        if len(charge_cycle_df) == 0:
            print(
                f"Warning: No charge data found in cycle {cycle}, skipping charge plot..."
            )
        else:
            charge_cycle_df["Charge_Time(s)"] = (
                charge_cycle_df["Test_Time(s)"]
                - charge_cycle_df["Test_Time(s)"].iloc[0]
            )

            # generate plot, clipped last datum in case current reset to rest
            plt.figure(figsize=(10, 6))
            plt.plot(
                charge_cycle_df["Charge_Time(s)"],
                charge_cycle_df["Voltage(V)"],
                color="blue",
            )
            plt.xlabel("Charge Time (s)")
            plt.ylabel("Voltage (V)", color="blue")
            plt.title(f"Cycle {cycle} Charge Profile")
            save_string = f"Cycle_{i+1}_charge_Crate_{c_rate}_tempK_{temperature}_batteryID_{battery_ID}.png"
            save_path = os.path.join(output_dir, save_string)
            plt.savefig(save_path, dpi=100, bbox_inches="tight")
            plt.close()

        # Check if we have valid discharge data and vmin was reached
        if len(discharge_cycle_df) == 0 or not vmin_reached:
            print(
                f"Warning: No valid discharge data found in cycle {cycle}, skipping discharge plot..."
            )
        else:
            # Additional validation: check if discharge curve has meaningful voltage range
            voltage_range = (
                discharge_cycle_df["Voltage(V)"].max()
                - discharge_cycle_df["Voltage(V)"].min()
            )
            if voltage_range < 0.1:  # Less than 0.1V range is not meaningful
                print(
                    f"Warning: Discharge voltage range too small ({voltage_range:.3f}V) in cycle {cycle}, skipping discharge plot..."
                )
            else:
                discharge_cycle_df["Discharge_Time(s)"] = (
                    discharge_cycle_df["Test_Time(s)"]
                    - discharge_cycle_df["Test_Time(s)"].iloc[0]
                )

                # plot current on secondary axis
                plt.figure(figsize=(10, 6))
                plt.plot(
                    discharge_cycle_df["Discharge_Time(s)"],
                    discharge_cycle_df["Voltage(V)"],
                    "r-",
                )  # remove last few points to avoid voltage recovery
                plt.ylabel("Voltage (V)", color="red")
                plt.title(f"Cycle {cycle} Discharge Profile")
                save_string = f"Cycle_{i+1}_discharge_Crate_{c_rate}_tempK_{temperature}_batteryID_{battery_ID}.png"
                save_path = os.path.join(output_dir, save_string)
                plt.savefig(save_path, dpi=100, bbox_inches="tight")
                plt.close()

        # Exit function after 1st run if one_fig_only is True
        if one_fig_only:
            break


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
    config: ProcessingConfig,
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
    df: pd.DataFrame,
    cell_number: int,
    cell_meta: CellMetadata,
    config: ProcessingConfig,
) -> str:
    """Save processed data to CSV file."""
    available_columns = [col for col in config.required_columns if col in df.columns]
    output_df = df[available_columns]

    csv_filename = (
        f"oxford_cell{cell_number}_{int(cell_meta.temperature)}k_aggregated_data.csv"
    )
    csv_path = os.path.join(config.output_data_path, csv_filename)

    os.makedirs(config.output_data_path, exist_ok=True)
    output_df.to_csv(csv_path, index=False)
    print(f"💾 Saved CSV file: {csv_path}")

    return csv_path


def generate_and_save_figures(
    df: pd.DataFrame,
    cell_number: int,
    cell_meta: CellMetadata,
    config: ProcessingConfig,
) -> None:
    """Generate and save figures for the processed data."""
    try:
        print(
            f"🖼️  Starting figure generation for Cell {cell_number} ({df.shape[0]} data points)"
        )

        # Create cell-specific subdirectory (directly under LCO, no extra 'images' folder)
        cell_id = f"Oxford_Cell{cell_number}"
        battery_images_path = os.path.join(config.output_images_path, cell_id)
        os.makedirs(battery_images_path, exist_ok=True)

        generate_figures(
            df,
            cell_meta.vmax,
            cell_meta.vmin,
            cell_meta.c_rate,
            cell_meta.temperature,
            battery_ID=cell_id,
            output_dir=battery_images_path,
            one_fig_only=False,
        )
        print(f"✅ Generated figures for Cell {cell_number}")

    except Exception as e:
        print(f"❌ Error generating figures for Cell {cell_number}: {str(e)}")


def save_error_log(error_dict: Dict[str, str], config: ProcessingConfig) -> None:
    """Save error log."""
    if not error_dict:
        return

    error_df = pd.DataFrame(list(error_dict.items()), columns=["Cell", "Error_Message"])
    error_log_path = os.path.join(config.output_data_path, "error_log_oxford.csv")
    error_df.to_csv(error_log_path, index=False)
    print(f"📝 Saved error log: {error_log_path}")


def main(config: Optional[ProcessingConfig] = None) -> None:
    """Main function to process Oxford battery data."""
    if config is None:
        config = ProcessingConfig()

    # Start timing
    start_time = time.time()
    print("🚀 Starting Oxford battery data processing...")

    # Setup paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.join(script_dir, "..", "..")
    file_path = os.path.join(
        project_root, config.base_data_path, "Oxford_Battery_Degradation_Dataset_1.mat"
    )

    print(f"📂 Processing: {file_path}")
    print(f"Dataset: Oxford Battery Degradation Dataset (LCO - Kokam 740mAh)")

    # Extract voltage limits from the data
    print("\nExtracting voltage limits from data...")
    try:
        vmax, vmin = extract_voltage_limits_from_mat_full(
            file_path, cell_number=1, percentile=99
        )
        print(f"✅ Extracted voltage limits: vmax={vmax:.4f}V, vmin={vmin:.4f}V")
    except Exception as e:
        vmax, vmin = 4.2, 2.9
        print(f"⚠️  Using default voltage limits: vmax={vmax}V, vmin={vmin}V")

    # Process cells 1-8
    cells_to_process = range(1, 9)
    print(f"📊 Processing {len(cells_to_process)} cells")

    error_dict = {}

    # Process each cell
    for cell_number in cells_to_process:
        cell_meta = get_cell_metadata(cell_number, vmax, vmin)

        # Process cell data
        df, error_msg = process_single_cell(file_path, cell_number, cell_meta, config)

        if error_msg:
            error_dict[f"Cell{cell_number}"] = error_msg
            continue

        if len(df) > 0:
            # Save processed data
            save_processed_data(df, cell_number, cell_meta, config)

            # Generate figures
            generate_and_save_figures(df, cell_number, cell_meta, config)

    # Save error log
    save_error_log(error_dict, config)

    # Calculate and display total processing time
    end_time = time.time()
    total_time = end_time - start_time

    # Format time display
    hours = int(total_time // 3600)
    minutes = int((total_time % 3600) // 60)
    seconds = int(total_time % 60)

    print(f"\n{'='*60}")
    print(f"🎉 All Oxford cells processed successfully!")
    print(f"⏱️  Total processing time: {hours:02d}:{minutes:02d}:{seconds:02d}")
    print(f"📊 Processed {len(cells_to_process)} cells")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
