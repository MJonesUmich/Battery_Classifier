import os
import re
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils.help_function import check_file_string, load_meta_properties


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
    unique_cycles = df["Cycle_Count"].unique()

    # If vmax or vmin are None or invalid, calculate from data
    if vmax is None or vmin is None or vmax <= 0 or vmin <= 0:
        data_vmax = df["Voltage(V)"].max()
        data_vmin = df["Voltage(V)"].min()
        vmax = data_vmax * 0.95  # Use 95% of max voltage
        vmin = data_vmin * 1.05  # Use 105% of min voltage
        print(f"Using calculated voltage limits: vmax={vmax:.3f}V, vmin={vmin:.3f}V")

    for i, cycle in enumerate(unique_cycles):
        cycle_df = df[df["Cycle_Count"] == cycle]

        # find where voltage first hits vmax and vmin, and where first discharge occurs
        vmax_matches = cycle_df[cycle_df["Voltage(V)"] >= vmax - tolerance]
        vmin_matches = cycle_df[cycle_df["Voltage(V)"] <= vmin + tolerance]
        disch_matches = cycle_df[cycle_df["Current(A)"] < 0 - tolerance]

        if len(vmax_matches) == 0 or len(vmin_matches) == 0 or len(disch_matches) == 0:
            print(f"Skipping cycle {cycle} - missing required voltage/current data")
            continue

        vmax_idx = vmax_matches.index[0]
        vmin_idx = vmin_matches.index[0]
        disch_start = disch_matches.index[0]

        # clip data to initial until Vmax, then from discharge start to Vmin
        charge_cycle_df = cycle_df.loc[0:vmax_idx].copy()
        discharge_cycle_df = cycle_df.loc[disch_start:vmin_idx].copy()

        # Check if we have valid data for plotting
        if len(charge_cycle_df) < 2 or len(discharge_cycle_df) < 2:
            print(f"Skipping cycle {cycle} - insufficient data points after filtering")
            continue

        # Calculate relative time for charge and discharge cycles
        if len(charge_cycle_df) > 0:
            charge_cycle_df["Charge_Time(s)"] = (
                charge_cycle_df["Test_Time(s)"]
                - charge_cycle_df["Test_Time(s)"].iloc[0]
            )
        if len(discharge_cycle_df) > 0:
            discharge_cycle_df["Discharge_Time(s)"] = (
                discharge_cycle_df["Test_Time(s)"]
                - discharge_cycle_df["Test_Time(s)"].iloc[0]
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
        save_string = os.path.join(
            output_dir,
            f"Cycle_{i+1}_charge_Crate_{c_rate}_tempK_{temperature}_batteryID_{battery_ID}.png",
        )
        plt.savefig(save_string)
        plt.close()

        # plot current on secondary axis
        plt.figure(figsize=(10, 6))
        plt.plot(
            discharge_cycle_df["Discharge_Time(s)"],
            discharge_cycle_df["Voltage(V)"],
            "r-",
        )  # remove last few points to avoid voltage recovery
        plt.ylabel("Voltage (V)", color="red")
        plt.title(f"Cycle {cycle} Discharge Profile")
        save_string = os.path.join(
            output_dir,
            f"Cycle_{i+1}_discharge_Crate_{c_rate}_tempK_{temperature}_batteryID_{battery_ID}.png",
        )
        plt.savefig(save_string)
        plt.close()

        # Exit function after 1st run if one_fig_only is True
        if one_fig_only:
            break


def main():
    # Process all CS2 subfolders
    meta_df = load_meta_properties()

    # Use relative path from the script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.join(script_dir, "..", "..")
    cs2_base_path = os.path.join(project_root, "assets", "raw_data", "CS2")

    # Get all subfolders in CS2 directory
    cs2_subfolders = [
        f
        for f in os.listdir(cs2_base_path)
        if os.path.isdir(os.path.join(cs2_base_path, f)) and f.startswith("CS2_")
    ]

    print(f"Found {len(cs2_subfolders)} CS2 subfolders: {cs2_subfolders}")

    # Process each subfolder
    for subfolder in cs2_subfolders:
        folder_path = os.path.join(cs2_base_path, subfolder)
        print(f"\nProcessing folder: {subfolder}")

        try:
            file_names = [file for file in os.listdir(folder_path)]

            sorted_files, file_dates = sort_files(file_names, orientation="last")
            sorted_files = sorted_files[::-1]
            file_dates = file_dates[::-1]

            print(f"Processing {len(sorted_files)} files in {subfolder}")
            error_dict = {}
            agg_df = pd.DataFrame()

            cell_id = subfolder
            print(f"Looking for battery ID: {cell_id}")
            cell_df = meta_df[meta_df["Battery_ID"].str.lower() == str.lower(cell_id)]
            print(f"Found {len(cell_df)} matching records")

            if len(cell_df) == 0:
                print(
                    f"Available Battery_IDs: {meta_df['Battery_ID'].dropna().unique()}"
                )
                print(f"No metadata found for battery ID: {cell_id}, skipping...")
                continue

            cell_initial_capacity = cell_df["Initial_Capacity_Ah"].values[0]
            cell_C_rate = cell_df["C_rate"].values[0]
            cell_temperature = cell_df["Temperature (K)"].values[0]
            cell_vmax = cell_df["Max_Voltage"].values[0]
            cell_vmin = cell_df["Min_Voltage"].values[0]

            # Set output directories
            output_base_dir = "processed_datasets"
            images_dir = os.path.join("processed_images", "LCO")

            for i_count, file_name in enumerate(sorted_files):
                try:
                    file_path = os.path.join(folder_path, file_name)
                    if file_name.endswith(".txt"):
                        method = "text"
                    elif file_name.endswith(".xlsx") or file_name.endswith(".xls"):
                        method = "excel"

                    df = parse_file(
                        file_path,
                        cell_initial_capacity,
                        cell_C_rate,
                        method,
                        cell_vmax,
                        cell_vmin,
                    )
                    df = update_df(df, agg_df)
                    agg_df = pd.concat([agg_df, df], ignore_index=True)

                # except add failed files to dictionary with error message
                except Exception as e:
                    error_dict[file_name] = str(e)

                print(
                    f"{round(i_count/len(sorted_files)*100,1)}% Complete for {subfolder}"
                )

            # Create CSV with only required columns
            required_columns = [
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
            available_columns = [
                col for col in required_columns if col in agg_df.columns
            ]
            output_df = agg_df[available_columns]

            # Generate output filename
            temperature_str = f"{int(cell_temperature)}K"
            csv_filename = f"{cell_id.lower()}_aggregated_data.csv"
            csv_path = os.path.join("processed_datasets", "LCO", csv_filename)

            # Create LCO subdirectory if it doesn't exist
            os.makedirs(os.path.dirname(csv_path), exist_ok=True)

            # Save CSV file
            output_df.to_csv(csv_path, index=False)
            print(f"Saved CSV file: {csv_path}")

            # Generate figures and save to images directory
            try:
                print(f"Starting figure generation for {cell_id}")
                print(f"Data shape: {agg_df.shape}")
                print(f"Cycles found: {agg_df['Cycle_Count'].nunique()}")
                generate_figures(
                    agg_df,
                    cell_vmax,
                    cell_vmin,
                    cell_C_rate,
                    cell_temperature,
                    battery_ID=cell_id,
                    output_dir=images_dir,
                    one_fig_only=False,
                )
                print(f"Generated figures for {cell_id}")
            except Exception as e:
                print(f"Error generating figures for {cell_id}: {str(e)}")
                error_dict[f"figures_{cell_id}"] = str(e)

            # Save error log for this subfolder
            if error_dict:
                error_df = pd.DataFrame(
                    list(error_dict.items()), columns=["File_Name", "Error_Message"]
                )
                error_log_path = os.path.join(
                    "processed_datasets", "LCO", "error_log.csv"
                )
                error_df.to_csv(error_log_path, index=False)
                print(f"Saved error log: {error_log_path}")

        except Exception as e:
            print(f"Error processing {subfolder}: {str(e)}")
            continue

    print("\nAll CS2 subfolders processed!")


if __name__ == "__main__":
    main()
