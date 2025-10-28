import os

import cx_cell_parser as cx
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def inr_parser(input_data_folder, output_folder_paths):
    output_data_folder, output_image_folder = output_folder_paths
    desired_folders = [
        folder
        for folder in os.listdir(input_data_folder)
        if os.path.isdir(os.path.join(input_data_folder, folder))
    ]
    counter = 0
    for folder in desired_folders:
        folder_path = os.path.join(input_data_folder, folder)
        cell_id = folder
        meta_df = cx.load_meta_properties()
        cell_df = meta_df[meta_df["Battery_ID"].str.lower() == str.lower(cell_id)]
        cell_initial_capacity = cell_df["Initial_Capacity_Ah"].values[0]
        cell_C_rate = cell_df["C_rate"].values[0]
        cell_temperature = cell_df["Temperature (K)"].values[0]
        cell_vmax = cell_df["Max_Voltage"].values[0]

        files = os.listdir(folder_path)
        agg_df = pd.DataFrame()
        for file in files:
            excel_file = pd.ExcelFile(os.path.join(folder_path, file))
            df = excel_file.parse(excel_file.sheet_names[-1])

            if "mV" in df.columns:
                df["mA"] = df["mA"] / 1000
                df["mV"] = df["mV"] / 1000

            df = df.rename(
                columns={
                    "mV": "Voltage(V)",
                    "mA": "Current(A)",
                    "Duration (sec)": "Test_Time(s)",
                }
            )

            # Separate data into charge and discharge & count the number of cycles
            discharge_start, charge_start, discharge_stop, charge_stop = [], [], [], []
            for i in range(len(df) - 1):  # Avoid accessing i+1 out of bounds
                # Detect the end of a charge cycle
                if df["Current(A)"].iloc[i] > 0 and df["Current(A)"].iloc[i + 1] <= 0:
                    charge_stop.append(i)

                # Detect the start of a charge cycle
                if df["Current(A)"].iloc[i] <= 0 and df["Current(A)"].iloc[i + 1] > 0:
                    charge_start.append(i)

                # Detect the end of a discharge cycle
                if df["Current(A)"].iloc[i] < 0 and df["Current(A)"].iloc[i + 1] >= 0:
                    discharge_stop.append(i)

                # Detect the start of a discharge cycle
                if df["Current(A)"].iloc[i] >= 0 and df["Current(A)"].iloc[i + 1] < 0:
                    discharge_start.append(i)

            if counter == 0:
                print("mismatch, downsampling!")
                charge_start = [charge_start[-1]]
                charge_stop = [charge_stop[-1]]
                print(charge_stop[-1])
                discharge_start = [discharge_start[0]]
                discharge_stop = [discharge_stop[0]]

                charge_df = df[charge_start[0] : charge_stop[0]].reset_index(drop=True)
                discharge_df = df[discharge_start[0] : discharge_stop[0]].reset_index(
                    drop=True
                )

            elif counter == 1:
                print("mismatch, downsampling!")
                charge_start = [charge_start[1]]
                charge_stop = [charge_stop[1]]
                discharge_start = [discharge_start[1]]
                discharge_stop = [discharge_stop[1]]
                charge_df = df[charge_start[0] : charge_stop[0]].reset_index(drop=True)

                # clip the moment we get to 4.2V
                clip_idx = np.where(charge_df["Voltage(V)"] >= cell_vmax)[0][0]
                charge_df = charge_df[0 : clip_idx + 1]
                discharge_df = df[discharge_start[0] : discharge_stop[0]].reset_index(
                    drop=True
                )

            charge_df["Charge_Time(s)"] = (
                charge_df["Test_Time(s)"] - charge_df["Test_Time(s)"].iloc[0]
            )
            discharge_df["Discharge_Time(s)"] = (
                discharge_df["Test_Time(s)"] - discharge_df["Test_Time(s)"].iloc[0]
            )

            # Add the direction feature
            charge_df["direction"] = "charge"
            discharge_df["direction"] = "discharge"
            out_df = pd.concat([discharge_df, charge_df]).reset_index(drop=True)
            out_df["Cycle_Count"] = counter + 1

            # Now coloumb colount:
            out_df["Delta_Time(s)"] = out_df["Test_Time(s)"].diff()

            Ah_list = []
            for i in range(len(out_df)):
                Ah_iter = (
                    np.abs(out_df["Current(A)"].iloc[i])
                    * out_df["Delta_Time(s)"].iloc[i]
                )
                Ah_list.append(Ah_iter)

            out_df["Delta_Ah"] = Ah_list
            out_df["Ah_throughput"] = out_df["Delta_Ah"].sum()
            out_df["EFC"] = out_df["Ah_throughput"] / cell_initial_capacity
            out_df["C_rate"] = cell_C_rate
            out_df = out_df[
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
                    "direction",
                ]
            ]

        agg_df = pd.concat([agg_df, out_df])

        # send to df and output:
        if not os.path.exists(output_data_folder):
            os.makedirs(output_data_folder)
        output_file_name = f"{cell_id}_aggregated_data.csv"
        agg_df.to_csv(os.path.join(output_data_folder, output_file_name), index=False)
        print(f"successfully exported {output_file_name} to {output_data_folder}")

        # Set plot directory
        if not os.path.exists(output_image_folder):
            os.makedirs(output_image_folder)

        # generate plot, clipped last datum in case current reset to rest
        plt.figure(figsize=(10, 6))
        plt.plot(charge_df["Charge_Time(s)"], charge_df["Voltage(V)"], color="blue")
        plt.xlabel("Charge Time (s)")
        plt.ylabel("Voltage (V)", color="blue")
        plt.title(f"Cycle {1} Charge Profile")
        save_string = f"{output_image_folder}\Cycle_{1}_charge_Crate_{cell_C_rate}_tempK_{cell_temperature}_batteryID_{cell_id}.png"
        plt.savefig(save_string)
        print(f"successfully exported {save_string} to {output_image_folder}")

        # plot current on secondary axis
        plt.figure(figsize=(10, 6))
        plt.plot(
            discharge_df["Discharge_Time(s)"], discharge_df["Voltage(V)"], "r-"
        )  # remove last few points to avoid voltage recovery
        plt.ylabel("Voltage (V)", color="red")
        plt.title(f"Cycle {1} Discharge Profile")
        save_string = f"{output_image_folder}\Cycle_{1}_discharge_Crate_{cell_C_rate}_tempK_{cell_temperature}_batteryID_{cell_id}.png"
        plt.savefig(save_string)
        counter += 1

        print(f"successfully exported {save_string} to {output_image_folder}")


if __name__ == "__main__":
    input_data_folder = r"C:\Users\MJone\Downloads\INR"
    output_data_folder = r"processed_datasets\NMC"
    output_image_folder = r"processed_images\NMC"
    output_foler_paths = (output_data_folder, output_image_folder)
    inr_parser(input_data_folder, output_foler_paths)
