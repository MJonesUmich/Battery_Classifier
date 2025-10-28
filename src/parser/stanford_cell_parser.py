import glob
import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_meta_properties():
    """Load battery metadata from Excel file"""
    df = pd.read_excel(
        r"C:\Users\ShuS\Documents\github\Battery_Classifier\data\battery_data_mapper.xlsx",
        sheet_name="General_Infos",
    )
    return df


def extract_battery_info_from_filename(file_name):
    """Extract battery ID, C-rate, and temperature from Stanford filename"""
    # Example: LFP_k1_0_05C_25degC.xlsx
    base_name = os.path.splitext(file_name)[0]
    parts = base_name.split("_")

    # Extract battery ID (LFP_k1, LFP_k2, etc.)
    battery_id = f"{parts[0]}_{parts[1]}"

    # Extract C-rate (0_05C -> 0.05, 1C -> 1.0, etc.)
    if len(parts) >= 4 and parts[3].endswith("C") and not "deg" in parts[3]:
        # C-rate is split across parts[2] and parts[3] (e.g., "0" and "05C")
        c_rate_whole = parts[2]  # e.g., "0"
        c_rate_fraction = parts[3]  # e.g., "05C"
        c_rate_fraction = c_rate_fraction[:-1]  # Remove 'C'
        c_rate = float(c_rate_whole + "." + c_rate_fraction)
    elif len(parts) >= 3 and parts[2].endswith("C"):
        # Handle cases like "1C" where C-rate is in one part
        c_rate_str = parts[2][:-1]  # Remove 'C'
        c_rate = float(c_rate_str)
    else:
        c_rate = 1.0  # Default

    # Extract temperature (25degC -> 298.15K)
    # Temperature is in the last part, e.g., "25degC.xlsx"
    temp_str = parts[-1]  # e.g., "25degC.xlsx"
    if "degC" in temp_str:
        temp_c = float(temp_str.split("degC")[0])
        temp_k = temp_c + 273.15
    else:
        temp_k = 298.15  # Default to 25°C

    return battery_id, c_rate, temp_k


def find_mapper_entry(meta_df, battery_id, temp_k):
    """Find the correct mapper entry by matching battery ID and temperature"""
    # Convert temperature to Celsius for matching
    temp_c = temp_k - 273.15

    # Create the expected mapper ID format: LFP_k1_05degC, LFP_k1_25degC, etc.
    if temp_c == 5.0:
        mapper_id = f"{battery_id}_05degC"
    elif temp_c == 25.0:
        mapper_id = f"{battery_id}_25degC"
    elif temp_c == 35.0:
        mapper_id = f"{battery_id}_35degC"
    else:
        # For other temperatures, try to match with the closest
        mapper_id = f"{battery_id}_{int(temp_c)}degC"

    # Normalize for comparison
    meta_df["_Battery_ID_norm"] = (
        meta_df["Battery_ID"].astype(str).str.strip().str.lower()
    )
    mapper_id_norm = mapper_id.strip().lower()

    # Try exact match first
    cell_df = meta_df[meta_df["_Battery_ID_norm"] == mapper_id_norm]

    if cell_df.empty:
        # Try partial match (in case of slight variations)
        cell_df = meta_df[
            meta_df["_Battery_ID_norm"].str.contains(battery_id.lower(), na=False)
        ]

    return cell_df, mapper_id


def load_stanford_file(file_path):
    """Load data from Stanford Excel file"""
    # Read Excel file
    df = pd.read_excel(file_path, sheet_name="Sheet1")
    df.columns = [str(c).strip() for c in df.columns]

    # Get the desired columns
    desired_cols = ["Current(A)", "Voltage(V)", "Test_Time(s)", "Date_Time"]
    available_cols = [col for col in desired_cols if col in df.columns]
    df = df[available_cols]

    # Convert numeric columns to proper data types
    numeric_cols = ["Current(A)", "Voltage(V)", "Test_Time(s)"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Convert Date_Time to datetime if it exists
    if "Date_Time" in df.columns:
        df["Date_Time"] = pd.to_datetime(df["Date_Time"], errors="coerce")

    # Remove rows where numeric conversion failed
    essential_cols = ["Current(A)", "Voltage(V)", "Test_Time(s)"]
    essential_available = [col for col in essential_cols if col in df.columns]
    df = df.dropna(subset=essential_available).reset_index(drop=True)

    return df


def get_charge_discharge_indices(df, voltage_tolerance=0.01):
    """Identify charge and discharge cycles based on current and voltage patterns"""
    I = df["Current(A)"]
    V = df["Voltage(V)"]

    # Set threshold to avoid 0 current jitter
    non_zero_currents = I[I != 0]
    if len(non_zero_currents) > 0:
        thr = max(0.05 * np.nanmedian(np.abs(non_zero_currents)), 1e-4)
    else:
        thr = 1e-4  # Default threshold when all currents are zero

    # Create current sign array
    sign3 = np.zeros_like(I, dtype=int)
    sign3[I > thr] = 1  # Charge
    sign3[I < -thr] = -1  # Discharge

    # Find charge/discharge transitions
    charge_indices, discharge_indices = [], []
    prev = 0

    for i, s in enumerate(sign3):
        if s == 0:
            continue
        if prev == 0:
            # First non-zero current - this is the start of charge
            if s == 1:
                charge_indices.append(i)
            prev = s
            continue
        if s != prev:
            if s == 1:  # Charge
                charge_indices.append(i)
            elif s == -1:  # Discharge
                discharge_indices.append(i)
            prev = s

    # For Stanford data, we often have single charge-discharge cycles
    # Check if this is a single cycle pattern
    if len(charge_indices) == 1 and len(discharge_indices) == 1:
        # print("Detected single charge-discharge cycle pattern (typical for Stanford data)")
        return charge_indices, discharge_indices

    # Validate and clean up indices for multi-cycle patterns
    complexity, expected_order = check_indices(charge_indices, discharge_indices)
    if complexity == "High":
        # print("Warning: Complex cycling pattern detected, using simplified processing")
        # For complex patterns, try to extract at least one cycle
        if charge_indices and discharge_indices:
            # Take the first charge and first discharge
            return [charge_indices[0]], [discharge_indices[0]]
        else:
            return [], []
    else:
        if expected_order[0] == "discharge" and len(discharge_indices) > len(
            charge_indices
        ):
            discharge_indices = discharge_indices[:-1]
        elif expected_order[0] == "charge" and len(charge_indices) > len(
            discharge_indices
        ):
            charge_indices = charge_indices[:-1]

    return charge_indices, discharge_indices


def check_indices(charge_indices, discharge_indices):
    """Check if charge/discharge indices alternate correctly"""
    if not charge_indices or not discharge_indices:
        return "High", []

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
    combined.sort(key=lambda x: x[0])

    # Check for alternating order
    for i, (_, label) in enumerate(combined):
        if label != expected_order[i % 2]:
            print(f"Error: Indices do not alternate correctly at position {i}")
            complexity = "High"
            return complexity, expected_order

    complexity = "Low"
    # print("Indices alternate correctly.")
    return complexity, expected_order


def filter_voltage_range(df, vmin, vmax, tolerance=0.01):
    """Filter data to only include voltage range between vmin and vmax"""
    # Find indices where voltage is within the target range
    voltage_mask = (df["Voltage(V)"] >= vmin - tolerance) & (
        df["Voltage(V)"] <= vmax + tolerance
    )

    # Find continuous segments within voltage range
    voltage_segments = []
    in_segment = False
    segment_start = None

    for i, is_in_range in enumerate(voltage_mask):
        if is_in_range and not in_segment:
            # Start of new segment
            segment_start = i
            in_segment = True
        elif not is_in_range and in_segment:
            # End of segment
            voltage_segments.append((segment_start, i - 1))
            in_segment = False

    # Handle case where segment continues to end of data
    if in_segment:
        voltage_segments.append((segment_start, len(df) - 1))

    # Keep only the largest segment (main charge/discharge cycle)
    if voltage_segments:
        largest_segment = max(voltage_segments, key=lambda x: x[1] - x[0])
        return df.iloc[largest_segment[0] : largest_segment[1] + 1].reset_index(
            drop=True
        )
    else:
        return df


def remove_voltage_plateaus(df, voltage_threshold=0.01, min_duration=50):
    """Remove constant voltage holds (plateaus) to keep only the informative voltage curves"""
    if len(df) == 0:
        return df

    voltage = df["Voltage(V)"].values
    keep_mask = np.ones(len(df), dtype=bool)

    i = 0
    while i < len(voltage) - min_duration:
        # Check if we have a plateau starting at position i
        plateau_voltage = voltage[i]
        plateau_end = i

        # Find the end of the plateau
        for j in range(i + 1, len(voltage)):
            if abs(voltage[j] - plateau_voltage) > voltage_threshold:
                plateau_end = j - 1
                break
            plateau_end = j

        # If plateau is long enough, mark it for removal
        if plateau_end - i >= min_duration:
            # Keep only a few points at the beginning and end of the plateau
            keep_start = min(5, plateau_end - i + 1)  # Keep first 5 points
            keep_end = min(5, plateau_end - i + 1)  # Keep last 5 points

            # Mark middle points for removal
            if plateau_end - i > keep_start + keep_end:
                keep_mask[i + keep_start : plateau_end - keep_end + 1] = False

            i = plateau_end + 1
        else:
            i += 1

    # Return filtered dataframe
    filtered_df = df[keep_mask].reset_index(drop=True)
    # print(f"Removed voltage plateaus: {len(df)} -> {len(filtered_df)} points ({len(df) - len(filtered_df)} points removed)")

    return filtered_df


def scrub_and_tag(df, charge_indices, discharge_indices, cell_initial_capacity):
    """Process and tag the data with cycle information"""
    # Downsample to just between charge cycles
    if charge_indices and discharge_indices:
        df = df.iloc[charge_indices[0] : discharge_indices[-1] + 1].reset_index(
            drop=True
        )

        # Adjust charge_indices and discharge_indices to match the new DataFrame
        adjusted_charge_indices = [i - charge_indices[0] for i in charge_indices]
        adjusted_discharge_indices = [i - charge_indices[0] for i in discharge_indices]
    else:
        adjusted_charge_indices = []
        adjusted_discharge_indices = []

    # Create a new column for tagging
    df["Cycle_Count"] = None

    # Assign cycle tags if we have indices
    if adjusted_charge_indices and adjusted_discharge_indices:
        for i, (start, end) in enumerate(
            zip(adjusted_charge_indices, adjusted_charge_indices[1:] + [len(df)]),
            start=1,
        ):
            df.loc[start : end - 1, "Cycle_Count"] = i
    else:
        # If no clear cycles, assign all data to cycle 1
        df["Cycle_Count"] = 1

    # Calculate Ah throughput for each cycle
    df["Delta_Time(s)"] = df["Test_Time(s)"].diff().fillna(0)
    df["Delta_Ah"] = np.abs(df["Current(A)"]) * df["Delta_Time(s)"] / 3600
    df["Ah_throughput"] = df["Delta_Ah"].cumsum()

    # Calculate Equivalent Full Cycles (EFC) & Capacity Fade
    df["EFC"] = df["Ah_throughput"] / cell_initial_capacity

    return df


def update_df(df, agg_df):
    """Update dataframe with aggregated data from previous files"""
    if len(agg_df) == 0:
        return df
    else:
        max_cycle = agg_df["Cycle_Count"].max()
        df["Cycle_Count"] = df["Cycle_Count"] + max_cycle
        df["Ah_throughput"] = df["Ah_throughput"] + agg_df["Ah_throughput"].max()
        df["EFC"] = df["EFC"] + agg_df["EFC"].max()
        return df


def parse_file(file_path, cell_initial_capacity, cell_C_rate, vmin, vmax):
    """Parse a single Stanford file"""
    df = load_stanford_file(file_path)

    # Filter to voltage range of interest (vmin to vmax)
    df = filter_voltage_range(df, vmin, vmax)

    # Remove constant voltage holds (plateaus) to keep only informative curves
    df = remove_voltage_plateaus(df, voltage_threshold=0.01, min_duration=50)

    try:
        charge_indices, discharge_indices = get_charge_discharge_indices(df)
        df = scrub_and_tag(df, charge_indices, discharge_indices, cell_initial_capacity)
    except ValueError as e:
        print(f"Warning: {e}. Using simplified processing.")
        df = scrub_and_tag(df, [], [], cell_initial_capacity)

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
    images_folder=".",
):
    """Generate charge/discharge profile plots based on available data"""
    unique_cycles = df["Cycle_Count"].unique()
    print(f"Processing {len(unique_cycles)} cycles for battery {battery_ID}")

    for i, cycle in enumerate(unique_cycles):
        cycle_df = df[df["Cycle_Count"] == cycle]

        if len(cycle_df) == 0:
            continue

        # Check for charge and discharge data availability
        has_charge_data = False
        has_discharge_data = False

        # Check for positive current (charge)
        positive_current = cycle_df[cycle_df["Current(A)"] > tolerance]
        if (
            len(positive_current) > 10
        ):  # Require at least 10 data points for meaningful charge data
            has_charge_data = True

        # Check for negative current (discharge) with sufficient data points
        negative_current = cycle_df[cycle_df["Current(A)"] < -tolerance]
        if (
            len(negative_current) > 50
        ):  # Require at least 50 data points for meaningful discharge data
            has_discharge_data = True
            print(
                f"Found {len(negative_current)} discharge data points for cycle {cycle}"
            )
        else:
            print(
                f"Skipping discharge plot for cycle {cycle} - only {len(negative_current)} discharge data points (need at least 50)"
            )

        # Generate charge plot if charge data exists
        if has_charge_data:
            try:
                # Find voltage maximum for charge data
                vmax_idx = cycle_df[cycle_df["Voltage(V)"] >= vmax - tolerance].index[0]
            except IndexError:
                vmax_idx = cycle_df["Voltage(V)"].idxmax()

            # Get charge data (from start to voltage maximum)
            charge_cycle_df = cycle_df.loc[0:vmax_idx].copy()

            if len(charge_cycle_df) > 0:
                charge_cycle_df.loc[:, "Charge_Time(s)"] = (
                    charge_cycle_df["Test_Time(s)"]
                    - charge_cycle_df["Test_Time(s)"].iloc[0]
                )

                # Generate charge plot
                plt.figure(figsize=(10, 6))
                plt.plot(
                    charge_cycle_df["Charge_Time(s)"],
                    charge_cycle_df["Voltage(V)"],
                    color="blue",
                )
                plt.xlabel("Charge Time (s)")
                plt.ylabel("Voltage (V)", color="blue")
                plt.title(f"Cycle {cycle} Charge Profile - Stanford LFP")
                save_string = f"Cycle_{i+1}_charge_Crate_{c_rate}_tempK_{temperature}_batteryID_Stanford_{battery_ID}.png"
                save_path = os.path.join(images_folder, save_string)
                plt.savefig(save_path)
                plt.close()

        # Generate discharge plot if discharge data exists
        if has_discharge_data:
            try:
                # Find discharge start (first negative current)
                disch_start = cycle_df[cycle_df["Current(A)"] < -tolerance].index[0]

                # Find voltage minimum for discharge data
                vmin_idx = cycle_df[cycle_df["Voltage(V)"] <= vmin + tolerance].index[0]
            except IndexError:
                # If no voltage minimum found, use the end of discharge data
                disch_start = cycle_df[cycle_df["Current(A)"] < -tolerance].index[0]
                vmin_idx = len(cycle_df) - 1

            # Get discharge data
            discharge_cycle_df = cycle_df.loc[disch_start:vmin_idx].copy()

            if len(discharge_cycle_df) > 0:
                discharge_cycle_df.loc[:, "Discharge_Time(s)"] = (
                    discharge_cycle_df["Test_Time(s)"]
                    - discharge_cycle_df["Test_Time(s)"].iloc[0]
                )

                # Generate discharge plot
                plt.figure(figsize=(10, 6))
                plt.plot(
                    discharge_cycle_df["Discharge_Time(s)"],
                    discharge_cycle_df["Voltage(V)"],
                    "r-",
                )
                plt.xlabel("Discharge Time (s)")
                plt.ylabel("Voltage (V)", color="red")
                plt.title(f"Cycle {cycle} Discharge Profile - Stanford LFP")
                save_string = f"Cycle_{i+1}_discharge_Crate_{c_rate}_tempK_{temperature}_batteryID_Stanford_{battery_ID}.png"
                save_path = os.path.join(images_folder, save_string)
                plt.savefig(save_path)
                plt.close()

        # Exit function after 1st run if one_fig_only is True
        if one_fig_only:
            break


def main():
    """Main processing function"""
    meta_df = load_meta_properties()

    # Get all Excel files from Stanford battery folder
    stanford_folder = (
        r"C:\Users\ShuS\Documents\github\Battery_Classifier\data\raw\Stanford\LFP"
    )
    # stanford_folder = r'C:\Users\ShuS\Documents\github\Battery_Classifier\data\raw\Stanford\NCA'
    # stanford_folder = r'C:\Users\ShuS\Documents\github\Battery_Classifier\data\raw\Stanford\NMC'

    # Extract battery type from folder path
    battery_type = os.path.basename(stanford_folder).upper()  # LFP, NCA, or NMC

    # Get all xlsx files from all temperature subdirectories
    file_pattern = os.path.join(stanford_folder, "**", "*.xlsx")
    file_paths = glob.glob(file_pattern, recursive=True)

    print(f"Found {len(file_paths)} Stanford {battery_type} files")

    # Group files by battery ID and temperature
    file_groups = {}
    for file_path in file_paths:
        file_name = os.path.basename(file_path)
        battery_id, c_rate, temp_k = extract_battery_info_from_filename(file_name)

        # Create a key for grouping
        group_key = f"{battery_id}_{temp_k:.0f}K"

        if group_key not in file_groups:
            file_groups[group_key] = []

        file_groups[group_key].append(
            {
                "path": file_path,
                "name": file_name,
                "battery_id": battery_id,
                "c_rate": c_rate,
                "temp_k": temp_k,
            }
        )

    print(f"Found {len(file_groups)} battery groups")

    # Create output folder for this battery type
    output_folder = f"stanford_{battery_type}"
    images_folder = os.path.join(output_folder, "images")
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"Created output folder: {output_folder}")
    if not os.path.exists(images_folder):
        os.makedirs(images_folder)
        print(f"Created images folder: {images_folder}")

    # Process each group
    error_dict = {}

    for group_key, files in file_groups.items():
        print(f"Processing group: {group_key}")

        # Get battery metadata
        battery_id = files[0]["battery_id"]
        temp_k = files[0]["temp_k"]

        # Find mapper entry using the new matching function
        cell_df, mapper_id = find_mapper_entry(meta_df, battery_id, temp_k)

        if cell_df.empty:
            cell_initial_capacity = 2.5  # Default for LFP
            cell_vmax = 3.6  # Default for LFP
            cell_vmin = 2.0  # Default for LFP
        else:
            cell_initial_capacity = cell_df["Initial_Capacity_Ah"].values[0]
            cell_vmax = (
                cell_df["Max_Voltage"].values[0]
                if not pd.isna(cell_df["Max_Voltage"].values[0])
                else 3.6
            )
            cell_vmin = cell_df["Min_Voltage"].values[0]

        # Sort files by C-rate for processing order
        files.sort(key=lambda x: x["c_rate"])

        agg_df = pd.DataFrame()

        for i, file_info in enumerate(files):
            try:
                df = parse_file(
                    file_info["path"],
                    cell_initial_capacity,
                    file_info["c_rate"],
                    cell_vmin,
                    cell_vmax,
                )
                df = update_df(df, agg_df)
                agg_df = pd.concat([agg_df, df], ignore_index=True)

            except Exception as e:
                error_dict[file_info["name"]] = str(e)
                print(f"Error processing {file_info['name']}: {e}")

        # Add battery metadata to aggregated data
        if len(agg_df) > 0:
            agg_df["Battery_ID"] = battery_id
            agg_df["Temperature_K"] = temp_k

            # Export individual CSV for this battery with only specified fields
            try:
                # Select only the required fields and rename Voltage(V) to Voltage(y)
                export_df = agg_df[
                    [
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
                ].copy()
                export_df = export_df.rename(columns={"Voltage(V)": "Voltage(y)"})

                # Create filename for individual battery export (corrected format)
                individual_filename = f"{battery_id}_{temp_k:.0f}K.csv"
                individual_filepath = os.path.join(output_folder, individual_filename)
                export_df.to_csv(individual_filepath, index=False)
                print(
                    f"Exported individual battery data: {len(export_df)} rows to {individual_filepath}"
                )

            except Exception as e:
                print(f"Error exporting individual CSV for {battery_id}: {e}")

            # Generate figures for this battery
            try:
                generate_figures(
                    agg_df,
                    cell_vmax,
                    cell_vmin,
                    files[0]["c_rate"],
                    temp_k,
                    battery_id,
                    one_fig_only=True,
                    images_folder=images_folder,
                )
            except Exception as e:
                print(f"Error generating figures for {battery_id}: {e}")

    print(f"Processing completed for {battery_type} battery type.")

    # Save error log
    if error_dict:
        error_df = pd.DataFrame(
            list(error_dict.items()), columns=["File_Name", "Error_Message"]
        )
        error_log_filename = f"stanford_{battery_type}_error_log.csv"
        error_df.to_csv(error_log_filename, index=False)
        print(f"Saved error log: {len(error_dict)} errors to {error_log_filename}")


if __name__ == "__main__":
    main()
