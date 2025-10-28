import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")  # Use non-interactive backend for multi-threading
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from utils.help_function import load_meta_properties


@dataclass
class ProcessingConfig:
    """Configuration class for PL data processing."""

    base_data_path: str = "assets/raw_data/PL"
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


def extract_date(file_name):
    # Extract MM, DD, YR from the file name
    # Remove file extension first
    base_name = os.path.splitext(file_name)[0]
    parts = base_name.split("_")

    # Handle PL file name formats:
    # 06_02_2015_25C_3M_Final.xls
    # 2_24_2015_PLN_IC_25C_3M_1.xls
    # 04_08_2015_PLN_IC_25C_6.xls
    # PL11, PL13 (no date in filename)

    # Find the date parts by looking for numeric parts from the beginning
    # Date should be: month (1-12), day (1-31), year (typically 14-15)
    numeric_parts = []
    for i, part in enumerate(parts):
        try:
            val = int(part)
            # Only consider reasonable values for date components
            if 1 <= val <= 31:  # Day range
                numeric_parts.append(val)
            elif 1 <= val <= 12:  # Month range
                numeric_parts.append(val)
            elif 10 <= val <= 99:  # Year range (10-99, will be converted to 2010-2099)
                numeric_parts.append(val)
            elif 2000 <= val <= 2099:  # Full year range
                numeric_parts.append(val)

            if len(numeric_parts) == 3:
                break
        except ValueError:
            # Skip non-numeric parts
            continue

    if len(numeric_parts) < 3:
        # For files without date in filename (like PL11), use a default date
        print(f"No date found in filename: {file_name}, using default date")
        return datetime(2015, 1, 1).date()

    # The first 3 numeric parts should be: month, day, year
    month, day, year = numeric_parts[0], numeric_parts[1], numeric_parts[2]

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


def sort_files(file_names):

    file_dates = []

    # Extract dates and sort files
    for file_name in file_names:
        file_date = extract_date(file_name)
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


def load_from_csv_file(file_path):
    """Load data from CSV file with the new format"""
    try:
        df = pd.read_csv(file_path)
        df.columns = [str(c).strip() for c in df.columns]

        # Map the new CSV column names to the expected format
        column_mapping = {
            "Time_sec": "Test_Time(s)",
            "Current_Amp": "Current(A)",
            "Voltage_Volt": "Voltage(V)",
        }

        # Rename columns if they exist
        for old_col, new_col in column_mapping.items():
            if old_col in df.columns:
                df = df.rename(columns={old_col: new_col})

        # Get the desired columns
        quant_cols = ["Current(A)", "Voltage(V)", "Test_Time(s)"]

        # Force these columns to be float
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

    except Exception as e:
        print(f"Error loading CSV file {file_path}: {e}")
        return pd.DataFrame()


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

    # print(len(charge_indices), len(discharge_indices))
    # print(charge_indices[0], discharge_indices[0])
    # print(charge_indices[1], discharge_indices[1])
    # print(charge_indices[2], discharge_indices[2])
    # print(charge_indices[3], discharge_indices[3])

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
    # Determine the correct order and range
    all_indices = sorted(charge_indices + discharge_indices)
    start_idx = all_indices[0]
    end_idx = all_indices[-1]

    # Downsample to the range containing all cycles
    df = df.iloc[start_idx : end_idx + 1].reset_index(drop=True)

    # Adjust charge_indices and discharge_indices to match the new DataFrame
    adjusted_charge_indices = [i - start_idx for i in charge_indices]
    adjusted_discharge_indices = [i - start_idx for i in discharge_indices]

    # Create a new column for tagging
    df["Cycle_Count"] = None

    # Process each cycle to filter out constant voltage holds
    filtered_cycles = []

    print(
        f"Processing {len(adjusted_charge_indices)} cycles with vmax={vmax}, vmin={vmin}"
    )

    for i, (charge_start, charge_end) in enumerate(
        zip(adjusted_charge_indices, adjusted_charge_indices[1:] + [len(df)]), start=1
    ):
        # Get the discharge start for this cycle
        if i <= len(adjusted_discharge_indices):
            discharge_start = adjusted_discharge_indices[i - 1]
        else:
            discharge_start = len(df)

        # Extract cycle data
        cycle_data = df.iloc[charge_start:discharge_start].copy()
        cycle_data["Cycle_Count"] = i

        print(f"\nCycle {i}: {len(cycle_data)} points before filtering")

        # Filter out constant voltage holds if vmax and vmin are provided
        if vmax is not None and vmin is not None:
            print(f"  Calling filter_voltage_range...")
            cycle_data = filter_voltage_range(cycle_data, vmax, vmin, tolerance)
            print(
                f"  After filtering: {len(cycle_data)} points, has Phase column: {'Phase' in cycle_data.columns}"
            )
        else:
            print(f"  Skipping filter_voltage_range (vmax={vmax}, vmin={vmin})")

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
    excluding constant voltage holds. Also adds a Phase column to mark charge/discharge.
    """
    if len(cycle_data) == 0:
        return cycle_data

    # Find voltage range indices
    voltage = cycle_data["Voltage(V)"].values
    current = cycle_data["Current(A)"].values

    # Debug: Check current range
    print(f"  Current range in cycle: {current.min():.4f} to {current.max():.4f} A")
    print(
        f"  Positive current points: {np.sum(current > tolerance)}, Negative current points: {np.sum(current < -tolerance)}"
    )

    # Find the first point where voltage reaches vmax (during charge)
    vmax_reached = False
    vmax_idx = None
    for i, v in enumerate(voltage):
        if v >= vmax - tolerance and not vmax_reached:
            vmax_reached = True
            vmax_idx = i
            break

    # Find the first point where voltage reaches vmin (during discharge)
    vmin_reached = False
    vmin_idx = None
    for i, v in enumerate(voltage):
        if v <= vmin + tolerance and not vmin_reached:
            vmin_reached = True
            vmin_idx = i
            break

    # Find discharge start (first negative current)
    discharge_start = None
    for i, c in enumerate(current):
        if c < -tolerance:
            discharge_start = i
            break

    # Filter data based on voltage range
    if vmax_idx is not None and vmin_idx is not None and discharge_start is not None:
        # Include charge portion up to vmax and discharge portion from start to vmin
        charge_portion = cycle_data.iloc[: vmax_idx + 1].copy()
        discharge_portion = cycle_data.iloc[discharge_start : vmin_idx + 1].copy()

        # Mark the phase for each portion
        charge_portion["Phase"] = "Charge"
        discharge_portion["Phase"] = "Discharge"

        # Debug output
        print(f"  Charge portion: {len(charge_portion)} points (index 0 to {vmax_idx})")
        print(
            f"  Discharge portion: {len(discharge_portion)} points (index {discharge_start} to {vmin_idx})"
        )

        # Combine charge and discharge portions
        filtered_data = pd.concat(
            [charge_portion, discharge_portion], ignore_index=True
        )
        print(
            f"  Total filtered data: {len(filtered_data)} points (Charge: {np.sum(filtered_data['Phase']=='Charge')}, Discharge: {np.sum(filtered_data['Phase']=='Discharge')})"
        )
        return filtered_data
    else:
        # If we can't find proper voltage bounds, return original data with phase marking
        print(
            f"  Warning: Could not find vmax_idx={vmax_idx}, vmin_idx={vmin_idx}, discharge_start={discharge_start}"
        )
        print(f"  Falling back to current-based phase marking")
        cycle_data = cycle_data.copy()
        cycle_data["Phase"] = cycle_data["Current(A)"].apply(
            lambda x: (
                "Charge"
                if x > tolerance
                else ("Discharge" if x < -tolerance else "Rest")
            )
        )
        phase_counts = cycle_data["Phase"].value_counts()
        print(f"  Phase distribution: {phase_counts.to_dict()}")
        return cycle_data


def update_df(df, agg_df):
    if len(agg_df) == 0:
        return df
    else:
        max_cycle = agg_df["Cycle_Count"].max()
        # Handle None values in Cycle_Count
        df["Cycle_Count"] = df["Cycle_Count"].fillna(0).astype(int) + int(max_cycle)
        df["Ah_throughput"] = df["Ah_throughput"] + agg_df["Ah_throughput"].max()
        df["EFC"] = df["EFC"] + agg_df["EFC"].max()
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
    return df


def parse_file(
    file_path, cell_initial_capacity, cell_C_rate, method="excel", vmax=None, vmin=None
):
    if method == "excel":
        df = load_file(file_path)
    elif method == "text":
        df = load_from_text_file(file_path)
    elif method == "csv":
        df = load_from_csv_file(file_path)

    if len(df) == 0:
        print(f"No data loaded from {file_path}")
        return pd.DataFrame()

    # Debug info only for successful loads
    if len(df) > 0:
        print(f"Successfully loaded {len(df)} rows from {file_path}")
        try:
            print(
                f"Current range: {df['Current(A)'].min():.3f} to {df['Current(A)'].max():.3f} A"
            )
            print(
                f"Voltage range: {df['Voltage(V)'].min():.3f} to {df['Voltage(V)'].max():.3f} V"
            )
        except Exception as e:
            print(f"Data loaded but may have format issues: {e}")

    charge_indices, discharge_indices = get_indices(df)

    if len(charge_indices) == 0 or len(discharge_indices) == 0:
        print(f"No charge/discharge indices found in {file_path}")
        return pd.DataFrame()

    print(
        f"Found {len(charge_indices)} charge indices and {len(discharge_indices)} discharge indices"
    )

    df = scrub_and_tag(
        df, charge_indices, discharge_indices, cell_initial_capacity, vmax, vmin
    )
    df["C_rate"] = cell_C_rate
    return df


def generate_figures(
    df,
    vmax,
    vmin,
    c_rate,
    temperature,
    battery_ID,
    tolerance=0.01,
    one_fig_only=False,
    output_folder=None,
):
    unique_cycles = df["Cycle_Count"].dropna().unique()
    if len(unique_cycles) == 0:
        print("No valid cycles found for figure generation")
        return

    # Track generated plots
    generated_charge_plots = 0
    generated_discharge_plots = 0

    for i, cycle in enumerate(unique_cycles):
        cycle_df = df[df["Cycle_Count"] == cycle].copy()

        if len(cycle_df) == 0:
            print(f"Warning: No data found for cycle {cycle}")
            continue

        print(f"\n=== Processing Cycle {cycle} for figures ===")
        print(f"Total cycle data: {len(cycle_df)} points")

        # Separate charge and discharge based on Phase column (if available) or current
        if "Phase" in cycle_df.columns:
            print(f"Using Phase column to separate charge/discharge")
            phase_counts = cycle_df["Phase"].value_counts()
            print(f"Phase distribution: {phase_counts.to_dict()}")
            charge_cycle_df = cycle_df[cycle_df["Phase"] == "Charge"].copy()
            discharge_cycle_df = cycle_df[cycle_df["Phase"] == "Discharge"].copy()
        else:
            # Fallback to current-based separation
            print(f"Using current to separate charge/discharge (no Phase column)")
            charge_mask = cycle_df["Current(A)"] > tolerance
            discharge_mask = cycle_df["Current(A)"] < -tolerance
            charge_cycle_df = cycle_df[charge_mask].copy()
            discharge_cycle_df = cycle_df[discharge_mask].copy()
            print(
                f"Charge points: {np.sum(charge_mask)}, Discharge points: {np.sum(discharge_mask)}"
            )

        # Check charge data
        if len(charge_cycle_df) == 0:
            print(
                f"⚠️  Cycle {cycle}: No charge data found - charge plot will NOT be generated"
            )
        elif len(charge_cycle_df) < 10:
            print(
                f"⚠️  Cycle {cycle}: Insufficient charge data ({len(charge_cycle_df)} points) - charge plot will NOT be generated"
            )
            print(
                f"    Reason: This appears to be a discharge-only test (e.g., capacity measurement)"
            )
        else:
            print(
                f"✓  Cycle {cycle}: Sufficient charge data ({len(charge_cycle_df)} points) - charge plot will be generated"
            )

        # Check discharge data
        if len(discharge_cycle_df) == 0:
            print(
                f"⚠️  Cycle {cycle}: No discharge data found - discharge plot will NOT be generated"
            )
        elif len(discharge_cycle_df) < 10:
            print(
                f"⚠️  Cycle {cycle}: Insufficient discharge data ({len(discharge_cycle_df)} points) - discharge plot will NOT be generated"
            )
        else:
            print(
                f"✓  Cycle {cycle}: Sufficient discharge data ({len(discharge_cycle_df)} points) - discharge plot will be generated"
            )

        # Process charge data (only if we have enough points)
        if len(charge_cycle_df) >= 10:
            # Data is already in correct order from filter_voltage_range
            # Calculate relative time from the start
            charge_cycle_df = charge_cycle_df.reset_index(drop=True)
            if charge_cycle_df["Test_Time(s)"].iloc[0] is not None:
                charge_cycle_df["Charge_Time(s)"] = (
                    charge_cycle_df["Test_Time(s)"]
                    - charge_cycle_df["Test_Time(s)"].iloc[0]
                )
            else:
                charge_cycle_df["Charge_Time(s)"] = range(len(charge_cycle_df))

            # Debug output
            print(
                f"Cycle {cycle}: Charge data points: {len(charge_cycle_df)}, Voltage range: {charge_cycle_df['Voltage(V)'].min():.3f} - {charge_cycle_df['Voltage(V)'].max():.3f}V"
            )
            print(
                f"  Time range: {charge_cycle_df['Charge_Time(s)'].min():.1f} - {charge_cycle_df['Charge_Time(s)'].max():.1f}s"
            )

            # Generate charge plot
            plt.figure(figsize=(10, 6))
            plt.plot(
                charge_cycle_df["Charge_Time(s)"],
                charge_cycle_df["Voltage(V)"],
                "b-",
                linewidth=2,
            )
            plt.xlabel("Charge Time (s)", fontsize=12)
            plt.ylabel("Voltage (V)", fontsize=12)
            plt.title(
                f"Cycle {int(cycle)} Charge Profile (vmin-vmax range)", fontsize=14
            )
            plt.grid(True, alpha=0.3)
            save_string = f"Cycle_{i+1}_charge_Crate_{c_rate}_tempK_{temperature}_batteryID_{battery_ID}.png"
            if output_folder:
                save_path = os.path.join(output_folder, save_string)
            else:
                save_path = save_string
            plt.savefig(save_path, dpi=100, bbox_inches="tight")
            plt.close()
            generated_charge_plots += 1
            print(f"   ✓ Saved: {save_path}")

        # Process discharge data (only if we have enough points)
        if len(discharge_cycle_df) >= 10:
            # Data is already in correct order from filter_voltage_range
            # Calculate relative time from the start
            discharge_cycle_df = discharge_cycle_df.reset_index(drop=True)
            if discharge_cycle_df["Test_Time(s)"].iloc[0] is not None:
                discharge_cycle_df["Discharge_Time(s)"] = (
                    discharge_cycle_df["Test_Time(s)"]
                    - discharge_cycle_df["Test_Time(s)"].iloc[0]
                )
            else:
                discharge_cycle_df["Discharge_Time(s)"] = range(len(discharge_cycle_df))

            # Debug output
            print(
                f"Cycle {cycle}: Discharge data points: {len(discharge_cycle_df)}, Voltage range: {discharge_cycle_df['Voltage(V)'].min():.3f} - {discharge_cycle_df['Voltage(V)'].max():.3f}V"
            )
            print(
                f"  Time range: {discharge_cycle_df['Discharge_Time(s)'].min():.1f} - {discharge_cycle_df['Discharge_Time(s)'].max():.1f}s"
            )

            # Generate discharge plot
            plt.figure(figsize=(10, 6))
            plt.plot(
                discharge_cycle_df["Discharge_Time(s)"],
                discharge_cycle_df["Voltage(V)"],
                "r-",
                linewidth=2,
            )
            plt.xlabel("Discharge Time (s)", fontsize=12)
            plt.ylabel("Voltage (V)", fontsize=12)
            plt.title(
                f"Cycle {int(cycle)} Discharge Profile (vmin-vmax range)", fontsize=14
            )
            plt.grid(True, alpha=0.3)
            save_string = f"Cycle_{i+1}_discharge_Crate_{c_rate}_tempK_{temperature}_batteryID_{battery_ID}.png"
            if output_folder:
                save_path = os.path.join(output_folder, save_string)
            else:
                save_path = save_string
            plt.savefig(save_path, dpi=100, bbox_inches="tight")
            plt.close()
            generated_discharge_plots += 1
            print(f"   ✓ Saved: {save_path}")

        # Exit function after 1st cycle if one_fig_only is True
        if one_fig_only:
            break

    # Print summary
    print(f"\n{'='*70}")
    print(f"FIGURE GENERATION SUMMARY for {battery_ID}")
    print(f"{'='*70}")
    print(f"  Charge plots generated:    {generated_charge_plots}")
    print(f"  Discharge plots generated: {generated_discharge_plots}")
    print(
        f"  Total plots generated:     {generated_charge_plots + generated_discharge_plots}"
    )
    if generated_charge_plots == 0:
        print(
            f"\n  ℹ️  Note: No charge plots were generated because this dataset contains"
        )
        print(f"     discharge-only data (typical for capacity fade measurements).")
    print(f"{'='*70}\n")


def get_pl_subfolders(base_path):
    """Get all PL subfolders from the base path."""
    subfolders = [
        f
        for f in os.listdir(base_path)
        if os.path.isdir(os.path.join(base_path, f))
        and f.startswith(("Capacity_", "SOC_"))
    ]
    return subfolders


def get_cell_metadata(meta_df: pd.DataFrame, cell_id: str) -> Optional[CellMetadata]:
    """Get cell metadata for a given battery ID."""
    cell_df = meta_df[meta_df["Battery_ID"].str.lower() == str.lower(cell_id)]

    if len(cell_df) == 0:
        # Try with "PL_" prefix
        cell_id_with_prefix = f"PL_{cell_id}"
        cell_df = meta_df[
            meta_df["Battery_ID"].str.lower() == str.lower(cell_id_with_prefix)
        ]

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
    folder_path: str,
    sorted_files: List[str],
    cell_meta: CellMetadata,
    file_names: List[str],
) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """Process all files in a subfolder and return aggregated data."""
    error_dict = {}
    agg_df = pd.DataFrame()

    # Extract subfolder name from folder_path
    subfolder = os.path.basename(folder_path)

    print(f"📁 Processing {len(sorted_files)} files for {subfolder}")

    for i_count, file_name in enumerate(sorted_files):
        try:
            # Find the full path for this file from the original file list
            file_path = None
            for full_path in file_names:
                if os.path.basename(full_path) == file_name:
                    file_path = full_path
                    break

            if file_path is None:
                print(f"Could not find full path for {file_name}")
                continue

            if file_name.endswith(".txt"):
                method = "text"
            elif file_name.endswith(".xlsx") or file_name.endswith(".xls"):
                method = "excel"
            elif file_name.endswith(".csv"):
                method = "csv"
            else:
                continue

            df = parse_file(
                file_path,
                cell_meta.initial_capacity,
                cell_meta.c_rate,
                method,
                cell_meta.vmax,
                cell_meta.vmin,
            )
            if len(df) > 0:
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
    agg_df: pd.DataFrame, cell_id: str, config: ProcessingConfig
) -> str:
    """Save processed data to CSV file."""
    available_columns = [
        col for col in config.required_columns if col in agg_df.columns
    ]
    output_df = agg_df[available_columns]

    csv_filename = f"{cell_id.lower()}_aggregated_data.csv"
    csv_path = os.path.join(config.output_data_path, csv_filename)

    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    output_df.to_csv(csv_path, index=False)
    print(f"💾 Saved CSV file: {csv_path}")

    return csv_path


def generate_and_save_figures(
    agg_df: pd.DataFrame,
    cell_meta: CellMetadata,
    cell_id: str,
    error_dict: Dict[str, str],
    config: ProcessingConfig,
) -> None:
    """Generate and save figures for the processed data."""
    try:
        print(
            f"🖼️  Starting figure generation for {cell_id} ({agg_df.shape[0]} data points, {agg_df['Cycle_Count'].nunique()} cycles)"
        )

        # Create battery-specific subdirectory
        battery_images_path = os.path.join(config.output_images_path, cell_id)
        os.makedirs(battery_images_path, exist_ok=True)

        generate_figures(
            agg_df,
            cell_meta.vmax,
            cell_meta.vmin,
            cell_meta.c_rate,
            cell_meta.temperature,
            battery_ID=cell_id,
            one_fig_only=False,
            output_folder=battery_images_path,
        )
        print(f"✅ Generated figures for {cell_id}")

    except Exception as e:
        print(f"❌ Error generating figures for {cell_id}: {str(e)}")
        error_dict[f"figures_{cell_id}"] = str(e)


def save_error_log(
    error_dict: Dict[str, str], cell_id: str, config: ProcessingConfig
) -> None:
    """Save error log for the subfolder."""
    if not error_dict:
        return

    error_df = pd.DataFrame(
        list(error_dict.items()), columns=["File_Name", "Error_Message"]
    )
    error_log_path = os.path.join(config.output_data_path, f"error_log_{cell_id}.csv")
    error_df.to_csv(error_log_path, index=False)
    print(f"📝 Saved error log: {error_log_path}")


def process_single_subfolder(
    subfolder: str, pl_base_path: str, meta_df: pd.DataFrame, config: ProcessingConfig
) -> None:
    """Process a single subfolder completely."""
    folder_path = os.path.join(pl_base_path, subfolder)
    print(f"\nProcessing folder: {subfolder}")

    try:
        # Get file list and sort
        file_names = []
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                if file.endswith((".xls", ".xlsx", ".csv")):
                    file_path_full = os.path.join(root, file)
                    # skip files < 200kb since they don't have enough data
                    if os.path.getsize(file_path_full) > 200 * 1024:
                        file_names.append(file_path_full)

        # Extract just the filenames for sorting
        file_names_only = [os.path.basename(f) for f in file_names]
        sorted_files, _ = sort_files(file_names_only)
        sorted_files = sorted_files[::-1]

        # Get cell metadata
        cell_meta = get_cell_metadata(meta_df, subfolder)
        if cell_meta is None:
            # Use default values if no metadata found
            cell_meta = CellMetadata(
                initial_capacity=1.0,
                c_rate=1.0,
                temperature=298.0,
                vmax=4.2,
                vmin=3.0,
            )
            print("Using default values for PL data")

        # Process all files
        agg_df, error_dict = process_files_in_subfolder(
            folder_path, sorted_files, cell_meta, file_names
        )

        if len(agg_df) == 0:
            print(f"No data processed for {subfolder}")
            return

        # Save processed data
        save_processed_data(agg_df, subfolder, config)

        # Generate figures
        generate_and_save_figures(agg_df, cell_meta, subfolder, error_dict, config)

        # Save error log
        save_error_log(error_dict, subfolder, config)

    except Exception as e:
        print(f"Error processing {subfolder}: {str(e)}")


def main(config: Optional[ProcessingConfig] = None) -> None:
    """Main function to process all PL subfolders with multi-threading."""
    if config is None:
        config = ProcessingConfig()

    # Start timing
    start_time = time.time()
    print("🚀 Starting PL battery data processing with 20 threads...")

    # Load metadata
    meta_df = load_meta_properties(sheet_name="General_Infos")

    # Setup paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.join(script_dir, "..", "..")
    pl_base_path = os.path.join(project_root, config.base_data_path)

    # Get subfolders
    pl_subfolders = get_pl_subfolders(pl_base_path)
    print(f"📂 Found {len(pl_subfolders)} PL batteries: {pl_subfolders}")

    # Process subfolders with 20 threads
    with ThreadPoolExecutor(max_workers=20) as executor:
        # Submit all subfolder processing tasks
        future_to_subfolder = {
            executor.submit(
                process_single_subfolder, subfolder, pl_base_path, meta_df, config
            ): subfolder
            for subfolder in pl_subfolders
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
    print(f"🎉 All PL subfolders processed successfully!")
    print(f"⏱️  Total processing time: {hours:02d}:{minutes:02d}:{seconds:02d}")
    print(f"📊 Processed {len(pl_subfolders)} subfolders with 20 threads")
    print(f"⚡ Average time per subfolder: {total_time/len(pl_subfolders):.2f} seconds")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
# 🎉 All PL subfolders processed successfully!
# ⏱️  Total processing time: 00:11:20
# 📊 Processed 3 subfolders with 20 threads
# ⚡ Average time per subfolder: 226.88 seconds
