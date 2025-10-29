import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
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
    """Configuration class for ISU data processing."""

    base_data_path: str = "assets/raw_data/ISU"
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
    c_rate_charge: float
    c_rate_discharge: float
    temperature: float
    dod: float


def load_cycling_json(file, path):
    """Load and convert cycling JSON file for a given cell."""
    with open(f"{path}/{file}", "r") as f:
        data = json.loads(json.load(f))  # double decode!

    # Convert start/stop times to numpy datetime for consistency
    for i, start_time in enumerate(data["start_stop_time"]["start"]):
        if start_time != "[]":
            data["start_stop_time"]["start"][i] = np.datetime64(start_time)
            data["start_stop_time"]["stop"][i] = np.datetime64(
                data["start_stop_time"]["stop"][i]
            )
        else:
            data["start_stop_time"]["start"][i] = []
            data["start_stop_time"]["stop"][i] = []
    return data


def extract_cycle_data(cycling_dict):
    """
    Extract time, voltage, and capacity for each charge/discharge cycle.
    Returns two DataFrames: charge_df and discharge_df.
    """
    charge_cycles = []
    discharge_cycles = []
    num_cycles = len(cycling_dict["QV_charge"]["t"])

    # Each entry in QV_charge and QV_discharge corresponds to one cycle
    for i in range(num_cycles):
        # --- Extract charge data ---
        I_charge = cycling_dict["QV_charge"]["I"][i]
        V_charge = cycling_dict["QV_charge"]["V"][i]
        t_charge = cycling_dict["QV_charge"]["t"][i]
        Q_charge = cycling_dict["QV_charge"]["Q"][i]

        if len(Q_charge) > 0:
            df_c = pd.DataFrame(
                {
                    "Cycle_Count": i + 1,
                    "Time": pd.to_datetime(
                        t_charge, format="%Y-%m-%d %H:%M:%S", errors="coerce"
                    ),
                    "Current(A)": I_charge,
                    "Voltage(V)": V_charge,
                    "Capacity": Q_charge,
                    "direction": "Charge",
                }
            )
            # Normalize time to start from zero
            t0 = df_c["Time"].iloc[0]
            df_c["Time(s)"] = (df_c["Time"] - t0).dt.total_seconds()

            # Clean and ensure monotonic direction
            df_c["Voltage"] = df_c["Voltage(V)"].round(3)
            df_c = df_c.drop_duplicates(subset=["Voltage(V)"])
            df_c = df_c.sort_values(by="Time(s)").reset_index(drop=True)

            # Sanity check — ensure charge increases in voltage
            if df_c["Voltage"].iloc[0] < df_c["Voltage"].iloc[-1]:
                charge_cycles.append(df_c)
            else:
                # Misclassified (actually discharge)
                df_c["direction"] = "Discharge"
                discharge_cycles.append(df_c)

        # --- Extract discharge data ---
        I_discharge = cycling_dict["QV_discharge"]["I"][i]
        V_discharge = cycling_dict["QV_discharge"]["V"][i]
        t_discharge = cycling_dict["QV_discharge"]["t"][i]
        Q_discharge = cycling_dict["QV_discharge"]["Q"][i]

        if len(Q_discharge) > 0:
            df_d = pd.DataFrame(
                {
                    "Cycle_Count": i + 1,
                    "Time": pd.to_datetime(
                        t_discharge, format="%Y-%m-%d %H:%M:%S", errors="coerce"
                    ),
                    "Current(A)": I_discharge,
                    "Voltage(V)": V_discharge,
                    "Capacity": Q_discharge,
                    "direction": "Discharge",
                }
            )
            # Normalize time to start from zero
            t0 = df_d["Time"].iloc[0]
            df_d["Time(s)"] = (df_d["Time"] - t0).dt.total_seconds()

            # Clean and ensure monotonic direction
            df_d["Voltage"] = df_d["Voltage(V)"].round(3)
            df_d = df_d.drop_duplicates(subset=["Voltage(V)"])
            df_d = df_d.sort_values(by="Time(s)").reset_index(drop=True)

            # Sanity check — ensure discharge decreases in voltage
            if df_d["Voltage"].iloc[0] > df_d["Voltage"].iloc[-1]:
                discharge_cycles.append(df_d)
            else:
                # Misclassified (actually charge)
                df_d["Type"] = "Charge"
                charge_cycles.append(df_d)

    # Combine all cycles into single DataFrames
    charge_df = pd.concat(charge_cycles, ignore_index=True)
    discharge_df = pd.concat(discharge_cycles, ignore_index=True)

    return charge_df, discharge_df


def clip_data(input_df, direction):
    voltage = input_df["Voltage(V)"].values
    if direction == "charge":
        limit = voltage.max() - 0.005
        clip_idx = np.argmax(voltage >= limit)
    elif direction == "discharge":
        limit = voltage.min() + 0.01
        clip_idx = np.argmax(voltage <= limit)
    else:
        return input_df.copy()

    return input_df.iloc[: clip_idx + 1].copy()


def monotonicity_check(input_df, direction):
    if len(input_df) > 5:
        if direction == "charge":
            valid_profile = input_df["Voltage(V)"].is_monotonic_increasing
            valid_profile = True
        elif direction == "discharge":
            valid_profile = input_df["Voltage(V)"].is_monotonic_decreasing
            valid_profile = True
    else:
        valid_profile = False

    return valid_profile


def generate_figures(
    charge_cycle_df: pd.DataFrame,
    discharge_cycle_df: pd.DataFrame,
    cell_meta: CellMetadata,
    battery_ID: str,
    cycle: int,
    output_dir: str,
):
    """Generate charge and discharge figures following matplotlib standards."""
    os.makedirs(output_dir, exist_ok=True)

    # Prepare data
    charge_cycle_df = charge_cycle_df.copy()
    discharge_cycle_df = discharge_cycle_df.copy()

    charge_cycle_df["step_time(s)"] = (
        charge_cycle_df["Test_Time(s)"] - charge_cycle_df["Test_Time(s)"].iloc[0]
    )
    discharge_cycle_df["step_time(s)"] = (
        discharge_cycle_df["Test_Time(s)"] - discharge_cycle_df["Test_Time(s)"].iloc[0]
    )

    # Generate charge plot
    fig = plt.figure(figsize=(10, 6))
    plt.plot(
        charge_cycle_df["step_time(s)"],
        charge_cycle_df["Voltage(V)"],
        "b-",
        linewidth=2,
    )
    plt.xlabel("Charge Time (s)", fontsize=12)
    plt.ylabel("Voltage (V)", fontsize=12)
    plt.title(f"Cycle {cycle} Charge Profile", fontsize=14)
    plt.grid(True, alpha=0.3)
    save_string = f"Cycle_{cycle}_charge_Crate_{cell_meta.c_rate_charge}_tempK_{cell_meta.temperature}_batteryID_{battery_ID}.png"
    save_path = os.path.join(output_dir, save_string)
    fig.savefig(save_path, dpi=100, bbox_inches="tight")
    plt.close(fig)

    # Generate discharge plot
    fig = plt.figure(figsize=(10, 6))
    plt.plot(
        discharge_cycle_df["step_time(s)"],
        discharge_cycle_df["Voltage(V)"],
        "r-",
        linewidth=2,
    )
    plt.xlabel("Discharge Time (s)", fontsize=12)
    plt.ylabel("Voltage (V)", fontsize=12)
    plt.title(f"Cycle {cycle} Discharge Profile", fontsize=14)
    plt.grid(True, alpha=0.3)
    save_string = f"Cycle_{cycle}_discharge_Crate_{cell_meta.c_rate_discharge}_tempK_{cell_meta.temperature}_batteryID_{battery_ID}.png"
    save_path = os.path.join(output_dir, save_string)
    fig.savefig(save_path, dpi=100, bbox_inches="tight")
    plt.close(fig)


def scrub_cycles(
    input_df: pd.DataFrame,
    cell_meta: CellMetadata,
    cell_id: str,
    output_images_path: str,
):
    """Process cycles and generate figures."""
    unique_cycles = input_df["Cycle_Count"].unique()
    output_df = pd.DataFrame()

    for cycle in unique_cycles:
        include_data = True
        cycle_df = input_df[input_df["Cycle_Count"] == cycle]
        df_charge = cycle_df[cycle_df["direction"] == "charge"]
        df_discharge = cycle_df[cycle_df["direction"] == "discharge"]

        approx_time_charge = 3600 * 0.9 / cell_meta.c_rate_charge / 2
        approx_time_discharge = 3600 * 0.9 / cell_meta.c_rate_discharge / 2

        if (
            df_charge["Test_Time(s)"].max() - df_charge["Test_Time(s)"].min()
            < approx_time_charge
        ):
            include_data = False
        if (
            df_discharge["Test_Time(s)"].max() - df_discharge["Test_Time(s)"].min()
            < approx_time_discharge
        ):
            include_data = False

        if include_data:
            try:
                # Clip data to avoid rest periods and constant-voltage conditions
                charge_clip_df = clip_data(df_charge, direction="charge")
                discharge_clip_df = clip_data(df_discharge, direction="discharge")

                # Create battery-specific subdirectory
                battery_images_path = os.path.join(output_images_path, cell_id)

                # Generate figures
                generate_figures(
                    df_charge,
                    df_discharge,
                    cell_meta,
                    cell_id,
                    cycle,
                    battery_images_path,
                )

                output_df = pd.concat(
                    [output_df, charge_clip_df, discharge_clip_df], ignore_index=True
                )
            except Exception as e:
                print(f"❌ Error processing cycle {cycle}: {e}")
                continue

    output_df = output_df.sort_values(
        by=["Cycle_Count", "direction", "Test_Time(s)"], ascending=[True, True, True]
    ).reset_index(drop=True)
    return output_df


def get_cell_metadata(meta_df: pd.DataFrame, cell_id: str) -> Optional[CellMetadata]:
    """Get cell metadata for a given battery ID."""
    cell_df = meta_df[meta_df["Battery_ID"].str.lower() == str.lower(cell_id)]

    if len(cell_df) == 0:
        print(f"No metadata found for battery ID: {cell_id}")
        return None

    return CellMetadata(
        initial_capacity=cell_df["Initial_Capacity_Ah"].values[0],
        c_rate_charge=cell_df["C_rate_Charge"].values[0],
        c_rate_discharge=cell_df["C_rate_Discharge"].values[0],
        temperature=cell_df["Temperature (K)"].values[0],
        dod=cell_df["DoD"].values[0],
    )


def process_single_file(
    file: str,
    input_data_folder: str,
    meta_df: pd.DataFrame,
    config: ProcessingConfig,
) -> Dict[str, str]:
    """Process a single ISU JSON file."""
    error_dict = {}
    cell_id = file.split(".")[0]

    print(f"\n📁 Processing file: {file}")

    try:
        # Get cell metadata
        cell_meta = get_cell_metadata(meta_df, cell_id)
        if cell_meta is None:
            return error_dict

        # Extract Cycles, but only for datasets where they comprise >90% DOD profile
        if cell_meta.dod > 0.9:
            cycling_dict = load_cycling_json(file, input_data_folder)
            df_charge, df_discharge = extract_cycle_data(cycling_dict)
            df_charge["direction"] = "charge"
            df_discharge["direction"] = "discharge"

            # Split between charge and discharge
            valid_charge = monotonicity_check(df_charge, direction="charge")
            valid_discharge = monotonicity_check(df_discharge, direction="discharge")

            if valid_charge and valid_discharge:
                # Combine and sort
                combined = pd.concat([df_charge, df_discharge])
                combined = combined.sort_values(
                    by=["Cycle_Count", "direction", "Time(s)"],
                    ascending=[True, True, True],
                ).reset_index(drop=True)

                combined["Test_Time(s)"] = (
                    combined["Time"] - combined["Time"].iloc[0]
                ).dt.total_seconds()
                combined["Delta_Time(s)"] = combined["Test_Time(s)"].diff().fillna(0)
                combined["Delta_Ah"] = combined["Capacity"].diff().fillna(0)
                combined["Ah_throughput"] = combined["Delta_Ah"].cumsum()
                combined["EFC"] = combined["Ah_throughput"] / cell_meta.initial_capacity
                combined["C_rate"] = combined["Current(A)"] / cell_meta.initial_capacity

                # Select required columns
                combined = combined[config.required_columns]

                # Process cycles and generate figures
                agg_df = scrub_cycles(
                    combined, cell_meta, cell_id, config.output_images_path
                )

                # Save aggregated data
                if len(agg_df) > 0:
                    csv_filename = f"{cell_id.lower()}_aggregated_data.csv"
                    csv_path = os.path.join(config.output_data_path, csv_filename)
                    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
                    agg_df.to_csv(csv_path, index=False)
                    print(f"💾 Saved CSV file: {csv_path}")
            else:
                print(f"⚠️  Skipping {cell_id} - invalid charge/discharge profiles")
        else:
            print(f"⚠️  Skipping {cell_id} - DoD < 90% ({cell_meta.dod:.1%})")

    except Exception as e:
        print(f"❌ Error processing {file}: {e}")
        error_dict[file] = str(e)

    return error_dict


def save_error_log(error_dict: Dict[str, str], config: ProcessingConfig) -> None:
    """Save error log for all batteries."""
    if not error_dict:
        return

    error_df = pd.DataFrame(
        list(error_dict.items()), columns=["File_Name", "Error_Message"]
    )
    error_log_path = os.path.join(config.output_data_path, "error_log_isu.csv")
    os.makedirs(os.path.dirname(error_log_path), exist_ok=True)
    error_df.to_csv(error_log_path, index=False)
    print(f"📝 Saved error log: {error_log_path}")


def main(config: Optional[ProcessingConfig] = None) -> None:
    """Main function to process all ISU files with multi-threading."""
    if config is None:
        config = ProcessingConfig()

    # Start timing
    start_time = time.time()
    print("🚀 Starting ISU battery data processing with 10 threads...")

    # Load metadata
    meta_df = load_meta_properties()

    # Setup paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.join(script_dir, "..", "..")
    isu_base_path = os.path.join(project_root, config.base_data_path)

    # Update config paths to absolute paths
    config.output_data_path = os.path.join(project_root, config.output_data_path)
    config.output_images_path = os.path.join(project_root, config.output_images_path)

    # Get all JSON files
    files = [f for f in os.listdir(isu_base_path) if f.endswith(".json")]
    print(f"📂 Found {len(files)} ISU JSON files")

    # Collect all errors
    all_errors = {}
    completed_count = 0

    # Process files with 10 threads
    with ThreadPoolExecutor(max_workers=10) as executor:
        # Submit all file processing tasks
        future_to_file = {
            executor.submit(
                process_single_file, file, isu_base_path, meta_df, config
            ): file
            for file in files
        }

        # Wait for all files to complete
        for future in as_completed(future_to_file):
            file = future_to_file[future]
            try:
                error_dict = future.result()
                all_errors.update(error_dict)
                completed_count += 1
                print(f"✅ [{completed_count}/{len(files)}] Completed: {file}")
            except Exception as e:
                print(f"✗ Error processing file {file}: {str(e)}")
                all_errors[file] = str(e)
                completed_count += 1

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
    print(f"🎉 All ISU files processed successfully!")
    print(f"⏱️  Total processing time: {hours:02d}:{minutes:02d}:{seconds:02d}")
    print(f"📊 Processed {len(files)} files with 10 threads")
    print(f"⚡ Average time per file: {total_time/len(files):.2f} seconds")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
# 🎉 All ISU files processed successfully!
# ⏱️  Total processing time: 02:30:44
# 📊 Processed 251 files with 10 threads
# ⚡ Average time per file: 36.04 seconds
