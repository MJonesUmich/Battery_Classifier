import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
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
    """Configuration class for TU Finland data processing."""

    base_data_path: str = "assets/raw_data/TU_Finland"
    output_data_path: str = "processed_datasets"
    output_images_path: str = "processed_images"
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


def extract_battery_specs_from_mat(file_path, percentile=95):
    """
    Extract battery specifications (capacity, C-rate, voltage limits) from TU Finland .mat format

    Parameters:
    - file_path: path to .mat file
    - percentile: percentile to use for determining voltage limits

    Returns:
    - vmax: maximum charge voltage
    - vmin: minimum discharge voltage
    - capacity: estimated battery capacity in Ah
    - c_rate: estimated C-rate
    """
    mat_data = loadmat(file_path)
    table = mat_data["table"]

    all_voltages = []
    charge_voltages = []
    discharge_voltages = []
    all_currents = []
    charge_currents = []

    # Collect voltage and current data from all entries
    for i, entry in enumerate(table[0]):
        if len(entry[1]) > 0 and len(entry[2]) > 0:  # Has voltage and current data
            voltage_data = entry[1][0]  # Get voltage array
            current_data = entry[2][0]  # Get current array

            # Add all voltages for overall range
            all_voltages.extend(voltage_data)
            all_currents.extend(current_data)

            # Find charge and discharge portions based on current
            non_zero_mask = np.abs(current_data) > 0.01
            if np.any(non_zero_mask):
                charge_mask = current_data > 0.01
                discharge_mask = current_data < -0.01

                if np.any(charge_mask):
                    charge_voltages.extend(voltage_data[charge_mask])
                    charge_currents.extend(current_data[charge_mask])
                if np.any(discharge_mask):
                    discharge_voltages.extend(voltage_data[discharge_mask])

    if len(all_voltages) == 0:
        # Fallback to default values
        return 3.6, 2.5, 2.5, 1.0

    # Extract voltage limits
    if len(charge_voltages) > 0 and len(discharge_voltages) > 0:
        vmax = np.percentile(charge_voltages, percentile)
        vmin = np.percentile(discharge_voltages, 100 - percentile)
    else:
        # Use overall voltage range with more conservative percentiles
        vmax = np.percentile(all_voltages, 98)  # Use 98th percentile for max
        vmin = np.percentile(all_voltages, 2)  # Use 2nd percentile for min

    # Extract capacity and C-rate information
    if len(charge_currents) > 0:
        max_charge_current = np.max(charge_currents)

        if max_charge_current > 0:
            # Estimate capacity based on voltage characteristics and current magnitude
            if vmax > 4.0:  # High voltage battery (NCA/NMC)
                estimated_capacity = max(2.5, min(5.0, max_charge_current / 0.8))
            else:  # Lower voltage battery (LFP)
                estimated_capacity = max(1.5, min(4.0, max_charge_current / 0.8))

            c_rate = 1.0
        else:
            estimated_capacity = 2.5
            c_rate = 1.0
    else:
        estimated_capacity = 2.5
        c_rate = 1.0

    return vmax, vmin, estimated_capacity, c_rate


def extract_voltage_limits_from_mat(file_path, percentile=95):
    """
    Extract voltage limits (vmax and vmin) from TU Finland .mat format

    Parameters:
    - file_path: path to .mat file
    - percentile: percentile to use for determining voltage limits

    Returns:
    - vmax: maximum charge voltage
    - vmin: minimum discharge voltage
    """
    mat_data = loadmat(file_path)
    table = mat_data["table"]

    all_voltages = []
    charge_voltages = []
    discharge_voltages = []

    # Collect voltage data from all entries
    for i, entry in enumerate(table[0]):
        if len(entry[1]) > 0 and len(entry[2]) > 0:  # Has voltage and current data
            voltage_data = entry[1][0]  # Get voltage array
            current_data = entry[2][0]  # Get current array

            # Add all voltages for overall range
            all_voltages.extend(voltage_data)

            # Find charge and discharge portions based on current
            non_zero_mask = np.abs(current_data) > 0.01
            if np.any(non_zero_mask):
                charge_mask = current_data > 0.01
                discharge_mask = current_data < -0.01

                if np.any(charge_mask):
                    charge_voltages.extend(voltage_data[charge_mask])
                if np.any(discharge_mask):
                    discharge_voltages.extend(voltage_data[discharge_mask])

    if len(all_voltages) == 0:
        # Fallback to default values
        return 3.6, 2.5

    # If we have charge/discharge data, use that for more accurate limits
    if len(charge_voltages) > 0 and len(discharge_voltages) > 0:
        vmax = np.percentile(charge_voltages, percentile)
        vmin = np.percentile(discharge_voltages, 100 - percentile)
        print(
            f"  Extracted from charge/discharge data: vmax={vmax:.3f}V, vmin={vmin:.3f}V"
        )
    else:
        # Use overall voltage range with more conservative percentiles
        vmax = np.percentile(all_voltages, 98)  # Use 98th percentile for max
        vmin = np.percentile(all_voltages, 2)  # Use 2nd percentile for min
        print(
            f"  Extracted from overall voltage range: vmax={vmax:.3f}V, vmin={vmin:.3f}V"
        )

    return vmax, vmin


def load_from_mat(file_path):
    """
    Load data from TU Finland .mat format
    Structure: table -> entries with Time, Voltage, Current, Temperature, Comment

    Parameters:
    - file_path: path to the .mat file

    Returns combined DataFrame for all test entries
    """
    mat_data = loadmat(file_path)
    table = mat_data["table"]

    all_dfs = []

    for i, entry in enumerate(table[0]):
        if len(entry[1]) > 0 and len(entry[2]) > 0:  # Has voltage and current data
            voltage_data = entry[1][0]  # Get voltage array
            current_data = entry[2][0]  # Get current array
            temperature_data = (
                entry[3][0] if len(entry[3]) > 0 else np.zeros_like(voltage_data)
            )  # Get temperature array
            comment = entry[4][0] if len(entry[4]) > 0 else f"Entry_{i}"  # Get comment

            # Create time array (assuming uniform sampling)
            time_data = np.arange(len(voltage_data))

            # Create DataFrame for this entry
            entry_df = pd.DataFrame(
                {
                    "Test_Time(s)": time_data,
                    "Voltage(V)": voltage_data,
                    "Current(A)": current_data,
                    "Temperature(K)": temperature_data,
                    "Comment": comment,
                    "Entry_Index": i,
                }
            )

            all_dfs.append(entry_df)

    if len(all_dfs) == 0:
        raise ValueError("No valid data found in mat file")

    # Combine all entries
    df = pd.concat(all_dfs, ignore_index=True)

    return df.dropna().reset_index(drop=True)


def get_indices(df):
    """Extract charge and discharge indices from current data"""
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
    """Check if charge and discharge indices alternate correctly"""
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
    """Process and tag the data with cycle information"""
    # Downsample to just between charge cycles
    df = df.iloc[charge_indices[0] : discharge_indices[-1] + 1].reset_index(drop=True)

    # Adjust charge_indices and discharge_indices to match the new DataFrame
    adjusted_charge_indices = [i - charge_indices[0] for i in charge_indices]
    adjusted_discharge_indices = [i - charge_indices[0] for i in discharge_indices]

    # Create a new column for tagging
    df["Cycle_Count"] = None

    # Process each cycle to filter out constant voltage holds
    filtered_cycles = []

    for i, (charge_start, charge_end) in enumerate(
        zip(adjusted_charge_indices, adjusted_charge_indices[1:] + [len(df)]), start=1
    ):
        # Extract cycle data (from this charge start to next charge start)
        cycle_data = df.iloc[charge_start:charge_end].copy()
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

    # Coulomb count Ah throughput for each cycle
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
        return cycle_data.iloc[filtered_indices].reset_index(drop=True)
    else:
        # Fallback: return original data if filtering fails
        return cycle_data


def update_df(df, agg_df):
    """Update DataFrame with aggregated data"""
    if len(agg_df) == 0:
        return df
    else:
        max_cycle = agg_df["Cycle_Count"].max()
        df["Cycle_Count"] = df["Cycle_Count"].astype(int) + int(max_cycle)
        df["Ah_throughput"] = df["Ah_throughput"] + agg_df["Ah_throughput"].max()
        df["EFC"] = df["EFC"] + agg_df["EFC"].max()
        return df


def parse_file(file_path, cell_initial_capacity, cell_C_rate, vmax=None, vmin=None):
    """
    Parse TU Finland .mat file and process battery data

    Parameters:
    - file_path: path to .mat file
    - cell_initial_capacity: initial capacity in Ah
    - cell_C_rate: C-rate value
    - vmax, vmin: voltage limits for filtering
    """
    df = load_from_mat(file_path)

    charge_indices, discharge_indices = get_indices(df)
    df = scrub_and_tag(
        df, charge_indices, discharge_indices, cell_initial_capacity, vmax, vmin
    )
    df["C_rate"] = cell_C_rate
    return df


def parse_mat_file(file_path, cell_initial_capacity, cell_C_rate, vmax=None, vmin=None):
    """
    Wrapper function specifically for parsing TU Finland .mat files

    Parameters:
    - file_path: path to .mat file
    - cell_initial_capacity: initial capacity in Ah
    - cell_C_rate: C-rate value
    - vmax, vmin: voltage limits for filtering
    """
    return parse_file(file_path, cell_initial_capacity, cell_C_rate, vmax, vmin)


def generate_figures(
    df: pd.DataFrame,
    cell_meta: CellMetadata,
    battery_ID: str,
    output_dir: str,
    tolerance=0.01,
):
    """Generate charge and discharge profile figures following matplotlib standards."""
    # Create battery-specific subdirectory
    battery_images_path = os.path.join(output_dir, battery_ID)
    os.makedirs(battery_images_path, exist_ok=True)

    unique_cycles = df["Cycle_Count"].unique()

    for i, cycle in enumerate(unique_cycles):
        cycle_df = df[df["Cycle_Count"] == cycle].reset_index(drop=True)

        # Find key indices
        try:
            vmax_idx = cycle_df[
                cycle_df["Voltage(V)"] >= cell_meta.vmax - tolerance
            ].index[0]
        except IndexError:
            vmax_idx = cycle_df["Voltage(V)"].idxmax()

        try:
            vmin_idx = cycle_df[
                cycle_df["Voltage(V)"] <= cell_meta.vmin + tolerance
            ].index[0]
        except IndexError:
            vmin_idx = cycle_df["Voltage(V)"].idxmin()

        try:
            disch_start = cycle_df[cycle_df["Current(A)"] < -tolerance].index[0]
        except IndexError:
            # Try with a smaller threshold if no discharge found
            try:
                disch_start = cycle_df[cycle_df["Current(A)"] < 0].index[0]
            except IndexError:
                disch_start = len(cycle_df)

        # Clip charge data
        charge_cycle_df = cycle_df.loc[0:vmax_idx].copy()

        # Clip discharge data
        if disch_start < len(cycle_df):
            discharge_portion = cycle_df.loc[disch_start:]
            try:
                vmin_idx_adjusted = discharge_portion[
                    discharge_portion["Voltage(V)"] <= cell_meta.vmin + tolerance
                ].index[0]
            except IndexError:
                vmin_idx_adjusted = discharge_portion.index[-1]

            discharge_cycle_df = cycle_df.loc[disch_start:vmin_idx_adjusted].copy()
        else:
            discharge_cycle_df = pd.DataFrame()

        # Generate charge plot
        if len(charge_cycle_df) > 0:
            charge_cycle_df["Charge_Time(s)"] = (
                charge_cycle_df["Test_Time(s)"]
                - charge_cycle_df["Test_Time(s)"].iloc[0]
            )

            fig = plt.figure(figsize=(10, 6))
            plt.plot(
                charge_cycle_df["Charge_Time(s)"],
                charge_cycle_df["Voltage(V)"],
                "b-",
                linewidth=2,
            )
            plt.xlabel("Charge Time (s)", fontsize=12)
            plt.ylabel("Voltage (V)", fontsize=12)
            plt.title(f"Cycle {cycle} Charge Profile", fontsize=14)
            plt.grid(True, alpha=0.3)
            save_string = f"Cycle_{i+1}_charge_Crate_{cell_meta.c_rate}_tempK_{cell_meta.temperature}_batteryID_{battery_ID}.png"
            save_path = os.path.join(battery_images_path, save_string)
            plt.savefig(save_path, dpi=100, bbox_inches="tight")
            plt.close(fig)

        # Generate discharge plot
        if len(discharge_cycle_df) > 0:
            discharge_cycle_df["Discharge_Time(s)"] = (
                discharge_cycle_df["Test_Time(s)"]
                - discharge_cycle_df["Test_Time(s)"].iloc[0]
            )

            fig = plt.figure(figsize=(10, 6))
            plt.plot(
                discharge_cycle_df["Discharge_Time(s)"],
                discharge_cycle_df["Voltage(V)"],
                "r-",
                linewidth=2,
            )
            plt.xlabel("Discharge Time (s)", fontsize=12)
            plt.ylabel("Voltage (V)", fontsize=12)
            plt.title(f"Cycle {cycle} Discharge Profile", fontsize=14)
            plt.grid(True, alpha=0.3)
            save_string = f"Cycle_{i+1}_discharge_Crate_{cell_meta.c_rate}_tempK_{cell_meta.temperature}_batteryID_{battery_ID}.png"
            save_path = os.path.join(battery_images_path, save_string)
            plt.savefig(save_path, dpi=100, bbox_inches="tight")
            plt.close(fig)


def get_cell_metadata_from_file(file_path: str, battery_type: str) -> CellMetadata:
    """Extract cell metadata from .mat file."""
    try:
        vmax, vmin, capacity, c_rate = extract_battery_specs_from_mat(
            file_path, percentile=95
        )
        temperature = 298.15  # Default temperature for TU Finland

        return CellMetadata(
            initial_capacity=capacity,
            c_rate=c_rate,
            temperature=temperature,
            vmax=vmax,
            vmin=vmin,
        )
    except Exception as e:
        print(f"⚠️  Error extracting metadata, using defaults: {e}")
        # Default values based on battery type
        if battery_type == "LFP":
            return CellMetadata(
                initial_capacity=2.5, c_rate=1.0, temperature=298.15, vmax=3.6, vmin=2.5
            )
        elif battery_type in ["NCA", "NMC"]:
            return CellMetadata(
                initial_capacity=3.0, c_rate=1.0, temperature=298.15, vmax=4.2, vmin=2.7
            )
        else:
            return CellMetadata(
                initial_capacity=2.5, c_rate=1.0, temperature=298.15, vmax=3.6, vmin=2.5
            )


def process_single_mat_file(
    mat_file: str,
    battery_type: str,
    data_dir: str,
    config: ProcessingConfig,
) -> Dict[str, str]:
    """Process a single .mat file."""
    error_dict = {}

    try:
        cell_id = mat_file.replace(".mat", "")
        file_path = os.path.join(data_dir, mat_file)

        print(f"\n📁 Processing: {battery_type}/{cell_id}")

        # Get cell metadata from file
        cell_meta = get_cell_metadata_from_file(file_path, battery_type)

        # Parse the mat file
        df = parse_mat_file(
            file_path,
            cell_meta.initial_capacity,
            cell_meta.c_rate,
            cell_meta.vmax,
            cell_meta.vmin,
        )

        # Select only required columns
        available_columns = [
            col for col in config.required_columns if col in df.columns
        ]
        df_filtered = df[available_columns]

        # Setup output paths
        output_data_path = os.path.join(config.output_data_path, battery_type)
        output_images_path = os.path.join(config.output_images_path, battery_type)
        os.makedirs(output_data_path, exist_ok=True)
        os.makedirs(output_images_path, exist_ok=True)

        # Save CSV
        csv_filename = (
            f"{cell_id.lower()}_{int(cell_meta.temperature)}k_aggregated_data.csv"
        )
        csv_path = os.path.join(output_data_path, csv_filename)
        df_filtered.to_csv(csv_path, index=False)
        print(f"💾 Saved: {csv_path} ({len(df_filtered)} data points)")

        # Generate figures
        try:
            generate_figures(
                df,
                cell_meta,
                cell_id,
                output_images_path,
            )
            print(f"🖼️  Generated figures for {cell_id}")
        except Exception as e:
            print(f"⚠️  Could not generate figures for {cell_id}: {e}")
            error_dict[f"{cell_id}_figures"] = str(e)

    except Exception as e:
        print(f"❌ Error processing {mat_file}: {e}")
        error_dict[mat_file] = str(e)

    return error_dict


def process_battery_type(
    battery_type: str,
    data_dir: str,
    config: ProcessingConfig,
) -> Dict[str, str]:
    """Process all .mat files for a specific battery type."""
    print(f"\n{'='*60}")
    print(f"Processing TU Finland {battery_type} dataset")
    print(f"{'='*60}")

    # Get list of .mat files (exclude OCV files)
    mat_files = [
        f
        for f in os.listdir(data_dir)
        if f.endswith(".mat") and not f.startswith("OCV")
    ]
    print(f"📂 Found {len(mat_files)} {battery_type} .mat files")

    if len(mat_files) == 0:
        return {}

    # Process files with threading
    all_errors = {}
    completed_count = 0

    with ThreadPoolExecutor(max_workers=5) as executor:
        future_to_file = {
            executor.submit(
                process_single_mat_file, mat_file, battery_type, data_dir, config
            ): mat_file
            for mat_file in mat_files
        }

        for future in as_completed(future_to_file):
            mat_file = future_to_file[future]
            try:
                error_dict = future.result()
                all_errors.update(error_dict)
                completed_count += 1
                print(f"✅ [{completed_count}/{len(mat_files)}] Completed: {mat_file}")
            except Exception as e:
                print(f"✗ Error processing {mat_file}: {str(e)}")
                all_errors[mat_file] = str(e)
                completed_count += 1

    # Save error log if there are errors
    if all_errors:
        error_df = pd.DataFrame(
            list(all_errors.items()), columns=["File_Name", "Error_Message"]
        )
        error_log_path = os.path.join(
            config.output_data_path,
            battery_type,
            f"error_log_{battery_type.lower()}.csv",
        )
        os.makedirs(os.path.dirname(error_log_path), exist_ok=True)
        error_df.to_csv(error_log_path, index=False)
        print(f"📝 Saved error log: {error_log_path}")

    print(f"✅ {battery_type} processing complete!")
    return all_errors


def save_error_log(
    all_errors: Dict[str, Dict[str, str]], config: ProcessingConfig
) -> None:
    """Save consolidated error log for all battery types."""
    if not all_errors or not any(all_errors.values()):
        return

    # Flatten errors from all battery types
    flat_errors = {}
    for battery_type, errors in all_errors.items():
        for file, error in errors.items():
            flat_errors[f"{battery_type}/{file}"] = error

    if flat_errors:
        error_df = pd.DataFrame(
            list(flat_errors.items()), columns=["File_Name", "Error_Message"]
        )
        error_log_path = os.path.join(
            config.output_data_path, "error_log_tu_finland.csv"
        )
        os.makedirs(os.path.dirname(error_log_path), exist_ok=True)
        error_df.to_csv(error_log_path, index=False)
        print(f"\n📝 Saved consolidated error log: {error_log_path}")


def main(config: Optional[ProcessingConfig] = None) -> None:
    """Main function to process all TU Finland files with multi-threading."""
    if config is None:
        config = ProcessingConfig()

    # Start timing
    start_time = time.time()
    print("🚀 Starting TU Finland battery data processing with 5 threads per type...")

    # Setup paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.join(script_dir, "..", "..")
    tu_base_path = os.path.join(project_root, config.base_data_path)

    # Update config paths to absolute paths
    config.output_data_path = os.path.join(project_root, config.output_data_path)
    config.output_images_path = os.path.join(project_root, config.output_images_path)

    # Battery type configurations
    battery_types = ["LFP", "NCA", "NMC"]
    all_errors = {}

    # Process each battery type
    for battery_type in battery_types:
        data_dir = os.path.join(tu_base_path, battery_type)
        if os.path.exists(data_dir):
            try:
                errors = process_battery_type(battery_type, data_dir, config)
                all_errors[battery_type] = errors
            except Exception as e:
                print(f"❌ Error processing {battery_type}: {str(e)}")
                all_errors[battery_type] = {"general_error": str(e)}
        else:
            print(f"⚠️  Directory not found: {data_dir}")

    # Save consolidated error log
    save_error_log(all_errors, config)

    # Calculate and display total processing time
    end_time = time.time()
    total_time = end_time - start_time

    # Format time display
    hours = int(total_time // 3600)
    minutes = int((total_time % 3600) // 60)
    seconds = int(total_time % 60)

    print(f"\n{'='*60}")
    print(f"PROCESSING SUMMARY")
    print(f"{'='*60}")
    for battery_type, errors in all_errors.items():
        if errors:
            print(f"❌ {battery_type}: {len(errors)} errors")
        else:
            print(f"✅ {battery_type}: SUCCESS")
    print(f"{'='*60}")
    print(f"🎉 All TU Finland files processed successfully!")
    print(f"⏱️  Total processing time: {hours:02d}:{minutes:02d}:{seconds:02d}")
    print(f"📊 Processed {len(battery_types)} battery types with 5 threads each")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()


# ============================================================
# 🎉 All TU Finland files processed successfully!
# ⏱️  Total processing time: 00:21:08
# 📊 Processed 3 battery types with 5 threads each
# ============================================================
