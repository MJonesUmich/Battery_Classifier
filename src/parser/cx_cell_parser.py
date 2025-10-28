import os
import re
import sys
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from utils.help_function import check_file_string, load_meta_properties


def extract_date(file_name, orientation="last"):
    # Extract MM, DD, YR from the file name
    name, extension = os.path.splitext(file_name)
    parts = name.split("_")
    if orientation == "last":
        month, day, year = int(parts[-3]), int(parts[-2]), int(parts[-1])
    elif orientation == "first":
        # find which part is that last on to have a path in it "\\":
        last_path = [i for i in range(len(parts)) if "\\" in parts[i]]
        # Extract the month from the last valid path part
        month = int(parts[last_path[-1]].split("\\")[-1])
        day, year = int(last_path[-1] + 1), int(last_path[-1] + 2)
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
    for s in xls.sheet_names:
        cols = set(pd.read_excel(file_path, sheet_name=s, nrows=1).columns.astype(str))
        if {"Current(A)", "Voltage(V)"} <= cols:
            chosen = s
            break
    if chosen is None:
        chosen = xls.sheet_names[0]

    df = pd.read_excel(file_path, sheet_name=chosen)
    df.columns = [str(c).strip() for c in df.columns]

    # Get the desired columns out:
    quant_cols = ["Current(A)", "Voltage(V)", "Test_Time(s)"]
    # In the dataframe force these columns to be float
    for col in quant_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    desired_cols = ["Current(A)", "Voltage(V)", "Test_Time(s)"]
    df = df[desired_cols].dropna().reset_index(drop=True)
    return df


def load_from_text_file(file_path):
    # Load data from a text file
    df = pd.read_csv(file_path, delimiter="\t")
    # rename columns:
    df.rename(
        columns={
            "Time": "Test_Time(s)",
            "mA": "Current(A)",
            "mV": "Voltage(V)",
            "Duration (sec)": "Test_Time(s)",
        },
        inplace=True,
    )

    if "Duration (sec)" in df.columns:
        df.rename(columns={"Duration (sec)": "Test_Time(s)"}, inplace=True)

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
    # print("Indices alternate correctly.")
    return complexity, expected_order


def scrub_and_tag(df, charge_indices, discharge_indices, cell_initial_capacity):
    # Downsample to just between charge cycles
    df = df.iloc[charge_indices[0] : discharge_indices[-1] + 1].reset_index(drop=True)

    # Adjust charge_indices and discharge_indices to match the new DataFrame
    adjusted_charge_indices = [i - charge_indices[0] for i in charge_indices]
    adjusted_discharge_indices = [i - charge_indices[0] for i in discharge_indices]

    # Create a new column for tagging
    df["Cycle_Count"] = None

    # Assign "Charge {i}" tags
    for i, (start, end) in enumerate(
        zip(adjusted_charge_indices, adjusted_charge_indices[1:] + [len(df)]), start=1
    ):
        df.loc[start : end - 1, "Cycle_Count"] = i

    # Coloumb count Ah throughput for each cycle
    df["Delta_Time(s)"] = df["Test_Time(s)"].diff().fillna(0)
    df["Delta_Ah"] = np.abs(df["Current(A)"]) * df["Delta_Time(s)"] / 3600
    df["Ah_throughput"] = df["Delta_Ah"].cumsum()

    # now calculate Equivalent Full Cycles (EFC) & Capacity Fade
    df["EFC"] = df["Ah_throughput"] / cell_initial_capacity
    return df


def update_df(df, agg_df):
    if len(agg_df) == 0:
        return df
    else:
        max_cycle = agg_df["Cycle_Count"].max()
        df["Cycle_Count"] = df["Cycle_Count"].astype(int) + int(max_cycle)
        df["Ah_throughput"] = df["Ah_throughput"] + agg_df["Ah_throughput"].max()
        df["EFC"] = df["EFC"] + agg_df["EFC"].max()
        return df


def parse_file(file_path, cell_initial_capacity, cell_C_rate, method="excel"):
    if method == "excel":
        df = load_file(file_path)
    elif method == "text":
        df = load_from_text_file(file_path)

    charge_indices, discharge_indices = get_indices(df)
    df = scrub_and_tag(df, charge_indices, discharge_indices, cell_initial_capacity)
    df["C_rate"] = cell_C_rate
    return df


def interpolate_df(input_df, n_points=100):
    return input_df
    direction = input_df.direction.iloc[0]
    input_df.drop(columns=["direction"], inplace=True)
    n_points = 100  # target number of samples
    old_index = np.linspace(0, 1, len(input_df))
    new_index = np.linspace(0, 1, n_points)

    # Interpolate linearly for all columns
    df_interp = pd.DataFrame(
        {
            col: np.interp(new_index, old_index, input_df[col].to_numpy(dtype=float))
            for col in input_df.columns
        }
    )
    df_interp["direction"] = direction
    return df_interp


def split_charge_discharge(
    vmax_idx, vmin_idx, iteration_num, cycle_df, vmax, vmin, tolerance=0.01
):
    # clip data to initial until Vmax, then from discharge start to Vmin
    disch_start = cycle_df[cycle_df["Current(A)"] < 0 - tolerance].index[0]
    charge_cycle_df = cycle_df.loc[0:vmax_idx].copy(deep=True)
    discharge_cycle_df = cycle_df.loc[disch_start:vmin_idx].copy(deep=True)
    if len(charge_cycle_df) > 3 and len(discharge_cycle_df) > 3:
        output = "Valid"
    else:
        output = "Invalid"

    if output == "Valid":
        charge_cycle_df["Charge_Time(s)"] = (
            charge_cycle_df["Test_Time(s)"] - charge_cycle_df["Test_Time(s)"].iloc[0]
        )
        discharge_cycle_df["Discharge_Time(s)"] = (
            discharge_cycle_df["Test_Time(s)"]
            - discharge_cycle_df["Test_Time(s)"].iloc[0]
        )
        charge_cycle_df["direction"] = "charge"
        discharge_cycle_df["direction"] = "discharge"

        # interpolate charge_cyle & discharge cycle
        interp_charge_cycle_df = interpolate_df(charge_cycle_df)
        interp_discharge_cycle_df = interpolate_df(discharge_cycle_df)

        # Stitch together charge and discharge
        end_charge_time = interp_charge_cycle_df["Test_Time(s)"].iloc[-1]
        interp_discharge_cycle_df["Test_Time(s)"] = (
            interp_discharge_cycle_df["Test_Time(s)"] + end_charge_time
        )

        # concat to the cycle df file
        cycle_df = pd.concat(
            [
                interp_charge_cycle_df.drop(columns=["Charge_Time(s)"]),
                interp_discharge_cycle_df.drop(columns=["Discharge_Time(s)"]),
            ],
            axis=0,
        )
        cycle_df["Cycle_Count"] = iteration_num + 1

    else:
        interp_charge_cycle_df = None
        interp_discharge_cycle_df = None
        cycle_df = None

    validity = output

    return interp_charge_cycle_df, interp_discharge_cycle_df, cycle_df, validity


def generate_figures(
    out_folder,
    charge_cycle_df,
    discharge_cycle_df,
    c_rate,
    temperature,
    battery_ID,
    cycle,
):
    # Set plot directory
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)

    # generate plot, clipped last datum in case current reset to rest
    fig = plt.figure(figsize=(10, 6))
    plt.plot(
        charge_cycle_df["Charge_Time(s)"], charge_cycle_df["Voltage(V)"], color="blue"
    )
    plt.xlabel("Charge Time (s)")
    plt.ylabel("Voltage (V)", color="blue")
    plt.title(f"Cycle {cycle} Charge Profile")
    save_string = f"{out_folder}\Cycle_{cycle}_charge_Crate_{c_rate}_tempK_{temperature}_batteryID_{battery_ID}.png"
    fig.savefig(save_string)
    plt.close(fig)

    # plot current on secondary axis
    fig = plt.figure(figsize=(10, 6))
    plt.plot(
        discharge_cycle_df["Discharge_Time(s)"], discharge_cycle_df["Voltage(V)"], "r-"
    )  # remove last few points to avoid voltage recovery
    plt.ylabel("Voltage (V)", color="red")
    plt.title(f"Cycle {cycle} Discharge Profile")
    save_string = f"{out_folder}\Cycle_{cycle}_discharge_Crate_{c_rate}_tempK_{temperature}_batteryID_{battery_ID}.png"
    fig.savefig(save_string)
    plt.close(fig)


def process_CX_datafiles(input_folder_paths, output_folder_paths, tolerance=0.001):
    meta_df = load_meta_properties()
    agg_folder_df = pd.DataFrame()
    agg_error_df = pd.DataFrame()
    out_img_folder, out_data_folder = output_folder_paths

    for folder_path in folder_paths:
        file_names = [file for file in os.listdir(folder_path)]

        # skip files < 200kb and sort by date of test:
        file_names = [
            file
            for file in file_names
            if os.path.getsize(os.path.join(folder_path, file)) > 200 * 1024
            and check_file_string(file) != "bad"
        ]
        sorted_files, file_dates = sort_files(file_names, orientation="last")

        error_dict = {}
        agg_df = pd.DataFrame()

        cell_id = folder_path.split("\\")[-1]
        cell_df = meta_df[meta_df["Battery_ID"].str.lower() == str.lower(cell_id)]
        cell_initial_capacity = cell_df["Initial_Capacity_Ah"].values[0]
        cell_C_rate = cell_df["C_rate"].values[0]
        cell_temperature = cell_df["Temperature (K)"].values[0]
        cell_vmax = cell_df["Max_Voltage"].values[0]
        cell_vmin = cell_df["Min_Voltage"].values[0]

        agg_file_df = pd.DataFrame()
        cycle_counter = 1
        for i_count, file_name in enumerate(file_names):
            # try:
            print("this is the file name: ", file_name)
            file_path = os.path.join(folder_path, file_name)
            if file_name.endswith(".txt"):
                method = "text"
            elif file_name.endswith(".xlsx") or file_name.endswith(".xls"):
                method = "excel"

            df = parse_file(file_path, cell_initial_capacity, cell_C_rate, method)
            unique_cycles = df["Cycle_Count"].unique()
            agg_cycle_df = pd.DataFrame()
            for iteration_num, cycle in enumerate(unique_cycles):
                print("This is the cycle num: ", cycle)
                cycle_df = df[df["Cycle_Count"] == cycle]
                vmax_candidates = cycle_df[
                    cycle_df["Voltage(V)"] >= cell_vmax - tolerance
                ]
                vmin_candidates = cycle_df[
                    cycle_df["Voltage(V)"] <= cell_vmin + tolerance
                ]

                if vmax_candidates.empty or vmin_candidates.empty:
                    continue
                else:
                    vmax_idx = vmax_candidates.index[0]
                    vmin_idx = vmin_candidates.index[0]
                    if len(cycle_df) > 5:
                        (
                            interp_charge_cycle_df,
                            interp_discharge_cycle_df,
                            inter_cycle_df,
                            validity,
                        ) = split_charge_discharge(
                            vmax_idx,
                            vmin_idx,
                            iteration_num,
                            cycle_df,
                            cell_vmax,
                            cell_vmin,
                            tolerance=0.01,
                        )
                        if validity != "Valid":
                            continue
                        else:
                            generate_figures(
                                out_img_folder,
                                interp_charge_cycle_df,
                                interp_discharge_cycle_df,
                                cell_C_rate,
                                cell_temperature,
                                cell_id,
                                cycle_counter,
                            )
                            agg_cycle_df = pd.concat(
                                [agg_cycle_df, inter_cycle_df],
                                axis=0,
                                ignore_index=True,
                            )
                            cycle_counter += 1
            agg_file_df = pd.concat(
                [agg_file_df, agg_cycle_df], axis=0, ignore_index=True
            )
            # except add failed files to dictionary with error message
            # except Exception as e:
            #     error_dict[file_name] = str(e)

            print(f"{round(i_count/len(file_names)*100,1)}% Complete")

        # send to df and output:
        error_df = pd.DataFrame(
            list(error_dict.items()), columns=["File_Name", "Error_Message"]
        )
        agg_error_df = pd.concat([agg_error_df, error_df], axis=0, ignore_index=True)

        # Export this battery's aggregated data
        battery_csv_name = f"{cell_id}_aggregated_data.csv"
        battery_csv_path = os.path.join(out_data_folder, battery_csv_name)
        agg_file_df.to_csv(battery_csv_path, index=False)
        print(f"battery_csv file exported to: {battery_csv_path}")

    # Export Datafiles
    if not os.path.exists(out_data_folder):
        os.makedirs(out_data_folder)
    agg_error_df.to_csv(os.path.join(out_data_folder, "error_log.csv"), index=False)


if __name__ == "__main__":
    # Example run through on 1 file
    folder_paths = [
        r"../../assets/raw_data/CX2/cx2_16",
        r"../../assets/raw_data/CX2/cx2_8",
        r"../../assets/raw_data/CX2/cx2_33",
        r"../../assets/raw_data/CX2/cx2_34",
        r"../../assets/raw_data/CX2/cx2_35",
        r"../../assets/raw_data/CX2/cx2_36",
        r"../../assets/raw_data/CX2/cx2_37",
        r"../../assets/raw_data/CX2/cx2_38",
    ]

    process_CX_datafiles(
        folder_paths,
        output_folder_paths=[
            r"../../processed_images/LCO",
            r"../../processed_datasets/LCO",
        ],
        tolerance=0.001,
    )


# Need to Export: Current(A), Voltage(V), Test_Time(s), Cycle_Count, Delta Time(s), Delta_Ah, Ah_throughput, EFC, C-rate
