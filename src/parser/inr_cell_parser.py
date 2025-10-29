import os
import sys
import time
from dataclasses import dataclass
from typing import Dict, List, Optional

import matplotlib

matplotlib.use("Agg")  # Use non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from utils.help_function import load_meta_properties


@dataclass
class ProcessingConfig:
    """Configuration class for INR data processing."""

    base_data_path: str = "assets/raw_data/INR"
    output_data_path: str = "processed_datasets/NMC"
    output_images_path: str = "processed_images/NMC"
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
                "direction",
            ]


@dataclass
class CellMetadata:
    """Cell metadata container."""

    initial_capacity: float
    c_rate: float
    temperature: float
    vmax: float
    vmin: float


def get_cell_metadata(meta_df: pd.DataFrame, cell_id: str) -> Optional[CellMetadata]:
    """Get cell metadata for a given battery ID."""
    cell_df = meta_df[meta_df["Battery_ID"].str.lower() == str.lower(cell_id)]

    if len(cell_df) == 0:
        print(f"No metadata found for battery ID: {cell_id}")
        return None

    return CellMetadata(
        initial_capacity=cell_df["Initial_Capacity_Ah"].values[0],
        c_rate=cell_df["C_rate"].values[0],
        temperature=cell_df["Temperature (K)"].values[0],
        vmax=cell_df["Max_Voltage"].values[0],
        vmin=cell_df["Min_Voltage"].values[0],
    )


def process_single_file(
    file_path: str,
    cell_meta: CellMetadata,
    counter: int,
) -> tuple:
    """Process a single INR file."""
    excel_file = pd.ExcelFile(file_path)
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
    for i in range(len(df) - 1):
        if df["Current(A)"].iloc[i] > 0 and df["Current(A)"].iloc[i + 1] <= 0:
            charge_stop.append(i)
        if df["Current(A)"].iloc[i] <= 0 and df["Current(A)"].iloc[i + 1] > 0:
            charge_start.append(i)
        if df["Current(A)"].iloc[i] < 0 and df["Current(A)"].iloc[i + 1] >= 0:
            discharge_stop.append(i)
        if df["Current(A)"].iloc[i] >= 0 and df["Current(A)"].iloc[i + 1] < 0:
            discharge_start.append(i)

    if counter == 0:
        charge_start = [charge_start[-1]]
        charge_stop = [charge_stop[-1]]
        discharge_start = [discharge_start[0]]
        discharge_stop = [discharge_stop[0]]
        charge_df = df[charge_start[0] : charge_stop[0]].reset_index(drop=True)
        discharge_df = df[discharge_start[0] : discharge_stop[0]].reset_index(drop=True)
    elif counter == 1:
        charge_start = [charge_start[1]]
        charge_stop = [charge_stop[1]]
        discharge_start = [discharge_start[1]]
        discharge_stop = [discharge_stop[1]]
        charge_df = df[charge_start[0] : charge_stop[0]].reset_index(drop=True)
        clip_idx = np.where(charge_df["Voltage(V)"] >= cell_meta.vmax)[0][0]
        charge_df = charge_df[0 : clip_idx + 1]
        discharge_df = df[discharge_start[0] : discharge_stop[0]].reset_index(drop=True)

    charge_df["Charge_Time(s)"] = (
        charge_df["Test_Time(s)"] - charge_df["Test_Time(s)"].iloc[0]
    )
    discharge_df["Discharge_Time(s)"] = (
        discharge_df["Test_Time(s)"] - discharge_df["Test_Time(s)"].iloc[0]
    )

    charge_df["direction"] = "charge"
    discharge_df["direction"] = "discharge"
    out_df = pd.concat([discharge_df, charge_df]).reset_index(drop=True)
    out_df["Cycle_Count"] = counter + 1

    out_df["Delta_Time(s)"] = out_df["Test_Time(s)"].diff()
    Ah_list = []
    for i in range(len(out_df)):
        Ah_iter = np.abs(out_df["Current(A)"].iloc[i]) * out_df["Delta_Time(s)"].iloc[i]
        Ah_list.append(Ah_iter)

    out_df["Delta_Ah"] = Ah_list
    out_df["Ah_throughput"] = out_df["Delta_Ah"].sum()
    out_df["EFC"] = out_df["Ah_throughput"] / cell_meta.initial_capacity
    out_df["C_rate"] = cell_meta.c_rate

    return out_df, charge_df, discharge_df


def generate_figures(
    charge_df: pd.DataFrame,
    discharge_df: pd.DataFrame,
    cell_meta: CellMetadata,
    cell_id: str,
    cycle: int,
    output_dir: str,
):
    """Generate charge and discharge figures."""
    os.makedirs(output_dir, exist_ok=True)

    # Generate charge plot
    plt.figure(figsize=(10, 6))
    plt.plot(
        charge_df["Charge_Time(s)"],
        charge_df["Voltage(V)"],
        "b-",
        linewidth=2,
    )
    plt.xlabel("Charge Time (s)", fontsize=12)
    plt.ylabel("Voltage (V)", fontsize=12)
    plt.title(f"Cycle {cycle} Charge Profile", fontsize=14)
    plt.grid(True, alpha=0.3)
    save_string = f"Cycle_{cycle}_charge_Crate_{cell_meta.c_rate}_tempK_{cell_meta.temperature}_batteryID_{cell_id}.png"
    save_path = os.path.join(output_dir, save_string)
    plt.savefig(save_path, dpi=100, bbox_inches="tight")
    plt.close()

    # Generate discharge plot
    plt.figure(figsize=(10, 6))
    plt.plot(
        discharge_df["Discharge_Time(s)"],
        discharge_df["Voltage(V)"],
        "r-",
        linewidth=2,
    )
    plt.xlabel("Discharge Time (s)", fontsize=12)
    plt.ylabel("Voltage (V)", fontsize=12)
    plt.title(f"Cycle {cycle} Discharge Profile", fontsize=14)
    plt.grid(True, alpha=0.3)
    save_string = f"Cycle_{cycle}_discharge_Crate_{cell_meta.c_rate}_tempK_{cell_meta.temperature}_batteryID_{cell_id}.png"
    save_path = os.path.join(output_dir, save_string)
    plt.savefig(save_path, dpi=100, bbox_inches="tight")
    plt.close()


def process_inr_files(
    folder_path: str,
    meta_df: pd.DataFrame,
    config: ProcessingConfig,
) -> Dict[str, str]:
    """Process all INR files in the folder (grouped by battery ID from metadata)."""
    error_dict = {}

    # Get all Excel files
    all_files = [
        f for f in os.listdir(folder_path) if f.endswith(".xls") or f.endswith(".xlsx")
    ]

    if len(all_files) == 0:
        print(f"⚠️  No Excel files found in {folder_path}")
        return error_dict

    print(f"📂 Found {len(all_files)} INR files")

    # Get all INR battery IDs from metadata (those starting with SP)
    inr_batteries = meta_df[
        meta_df["Battery_ID"].str.contains("SP.*_LC_OCV", na=False, regex=True)
    ]

    print(f"📊 Found {len(inr_batteries)} INR batteries in metadata:")
    for battery_id in inr_batteries["Battery_ID"].values:
        print(f"  - {battery_id}")

    # Process each battery
    for _, row in inr_batteries.iterrows():
        cell_id = row["Battery_ID"]
        print(f"\n📁 Processing battery: {cell_id}")

        try:
            # Get cell metadata
            cell_meta = get_cell_metadata(meta_df, cell_id)
            if cell_meta is None:
                continue

            agg_df = pd.DataFrame()
            counter = 0

            # Process all files for this battery
            for file in all_files:
                try:
                    file_path = os.path.join(folder_path, file)
                    out_df, charge_df, discharge_df = process_single_file(
                        file_path, cell_meta, counter
                    )

                    # Select required columns
                    out_df = out_df[config.required_columns]
                    agg_df = pd.concat([agg_df, out_df])

                    # Generate figures for this cycle
                    battery_images_path = os.path.join(
                        config.output_images_path, cell_id
                    )
                    generate_figures(
                        charge_df,
                        discharge_df,
                        cell_meta,
                        cell_id,
                        counter + 1,
                        battery_images_path,
                    )

                    counter += 1
                    print(f"  ✓ Processed file: {file}")

                except Exception as e:
                    error_dict[f"{cell_id}_{file}"] = str(e)
                    print(f"  ❌ Error processing {file}: {e}")

            # Save aggregated data
            if len(agg_df) > 0:
                csv_filename = f"{cell_id.lower()}_aggregated_data.csv"
                csv_path = os.path.join(config.output_data_path, csv_filename)
                os.makedirs(os.path.dirname(csv_path), exist_ok=True)
                agg_df.to_csv(csv_path, index=False)
                print(f"💾 Saved CSV file: {csv_path}")

        except Exception as e:
            print(f"❌ Error processing {cell_id}: {str(e)}")
            error_dict[cell_id] = str(e)

    return error_dict


def save_error_log(error_dict: Dict[str, str], config: ProcessingConfig) -> None:
    """Save error log for all batteries."""
    if not error_dict:
        return

    error_df = pd.DataFrame(
        list(error_dict.items()), columns=["File_Name", "Error_Message"]
    )
    error_log_path = os.path.join(config.output_data_path, "error_log_inr.csv")
    os.makedirs(os.path.dirname(error_log_path), exist_ok=True)
    error_df.to_csv(error_log_path, index=False)
    print(f"📝 Saved error log: {error_log_path}")


def main(config: Optional[ProcessingConfig] = None) -> None:
    """Main function to process all INR files."""
    if config is None:
        config = ProcessingConfig()

    # Start timing
    start_time = time.time()
    print("🚀 Starting INR battery data processing...")

    # Load metadata
    meta_df = load_meta_properties()

    # Setup paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.join(script_dir, "..", "..")
    inr_base_path = os.path.join(project_root, config.base_data_path)

    # Update config paths to absolute paths
    config.output_data_path = os.path.join(project_root, config.output_data_path)
    config.output_images_path = os.path.join(project_root, config.output_images_path)

    # Process all INR files in the base directory
    all_errors = process_inr_files(inr_base_path, meta_df, config)

    # Save error log
    save_error_log(all_errors, config)

    # Calculate and display total processing time
    end_time = time.time()
    total_time = end_time - start_time

    # Format time display
    hours = int(total_time // 3600)
    minutes = int((total_time % 3600) // 60)
    seconds = int(total_time % 60)

    print(f"\n{'='*60}")
    print(f"🎉 All INR files processed successfully!")
    print(f"⏱️  Total processing time: {hours:02d}:{minutes:02d}:{seconds:02d}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
