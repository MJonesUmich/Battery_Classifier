import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import h5py
import matplotlib

matplotlib.use("Agg")  # Use non-interactive backend for multi-threading
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from utils.help_function import load_meta_properties


@dataclass
class ProcessingConfig:
    """Configuration class for MIT data processing."""

    base_data_path: str = "assets/raw_data/MIT"
    output_data_path: str = "processed_datasets/LFP"  # For aggregated data
    output_images_path: str = "processed_images/LFP"
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


def gen_bat_df(input_path, input_file, output_path):
    """Generate battery dataframe from MIT .mat files with reduced precision"""
    input_filepath = os.path.join(input_path, input_file)
    bol_cap = 1.1

    # Extract short batch name from filename (e.g., "2017-05-12" -> "MIT_20170512")
    batch_date = input_file.split("_")[0]  # Get date part
    batch_name = f"MIT_{batch_date.replace('-', '')}"

    try:
        with h5py.File(input_filepath, "r") as f:
            batch = f.get("batch")
            if batch is None or "cycles" not in batch:
                print(f"Warning: Required data not found in {input_file}")
                return

            cell_qty = batch["summary"].shape[0]
            for cell in range(cell_qty):
                all_cycles_data = []
                cycles_ref = batch["cycles"][cell, 0]
                cycles = f[cycles_ref]

                if "I" not in cycles:
                    continue

                n_cycles = cycles["I"].shape[0]
                for cycle in range(n_cycles):
                    # Read cycle data
                    def get_cycle_data(key):
                        try:
                            ref = cycles[key][cycle, 0]
                            return np.array(f[ref]).squeeze()
                        except Exception:
                            return np.array([])

                    current = get_cycle_data("I")
                    voltage = get_cycle_data("V")
                    temperature = get_cycle_data("T")
                    time = get_cycle_data("t")

                    # Skip empty cycles
                    if len(current) == 0 or len(voltage) == 0:
                        continue

                    # Create cycle DataFrame with reduced precision
                    cycle_data = pd.DataFrame(
                        {
                            "Cell_ID": np.full(len(current), cell, dtype=np.int16),
                            "Cycle_Count": np.full(len(current), cycle, dtype=np.int16),
                            "Current(A)": current.astype(np.float32),
                            "Voltage(V)": voltage.astype(np.float32),
                            "Test_Time(s)": time.astype(np.float32),
                            "direction": np.where(current > 0, "charge", "discharge"),
                        }
                    )

                    cycle_data["Test_Time(s)"] = (
                        cycle_data["Test_Time(s)"] * 60
                    ).astype(np.float32)
                    cycle_data["Delta_Time(s)"] = (
                        cycle_data["Test_Time(s)"].diff().fillna(0).astype(np.float32)
                    )
                    cycle_data["Delta_Ah"] = (
                        cycle_data["Current(A)"] * cycle_data["Delta_Time(s)"] / 3600
                    ).astype(np.float32)
                    cycle_data["C_rate"] = (cycle_data["Current(A)"] / bol_cap).astype(
                        np.float16
                    )  # BOL 1.1Ah capacity

                    all_cycles_data.append(cycle_data)

                if all_cycles_data:
                    # Combine all cycles and save with reduced memory footprint
                    cell_df = pd.concat(all_cycles_data, ignore_index=True)
                    cell_df["Ah_throughput"] = (
                        np.abs(cell_df["Delta_Ah"]).cumsum().astype(np.float32)
                    )
                    cell_df["EFC"] = (cell_df["Ah_throughput"] / bol_cap).astype(
                        np.float32
                    )  # BOL 1.1Ah capacity

                    output_df = cell_df.copy()
                    output_df = output_df[
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

                    # Export with shorter filename
                    out_name = f"{batch_name}_cell_{cell}_processed.csv"
                    output_filepath = os.path.join(output_path, out_name)
                    os.makedirs(os.path.dirname(output_filepath), exist_ok=True)
                    output_df.to_csv(
                        output_filepath,
                        index=False,
                        float_format="%.4f",  # Limit decimal places in output
                    )

                    print(
                        f"Processed cell {cell}: {len(all_cycles_data)} cycles, {len(cell_df)} total points"
                    )

    except Exception as e:
        print(f"Error processing {input_file}: {str(e)}")
        return None


def generate_figures(
    df_charge: pd.DataFrame,
    df_discharge: pd.DataFrame,
    cell_meta: CellMetadata,
    battery_id: str,
    cycle: int,
    output_dir: str,
):
    """Generate charge and discharge figures following matplotlib standards."""
    os.makedirs(output_dir, exist_ok=True)

    # Prepare data - normalize time to start from zero
    df_charge = df_charge.copy()
    df_discharge = df_discharge.copy()

    df_charge["step_time(s)"] = (
        df_charge["Test_Time(s)"] - df_charge["Test_Time(s)"].iloc[0]
    )
    df_discharge["step_time(s)"] = (
        df_discharge["Test_Time(s)"] - df_discharge["Test_Time(s)"].iloc[0]
    )

    # Generate charge plot
    fig = plt.figure(figsize=(10, 6))
    plt.plot(
        df_charge["step_time(s)"],
        df_charge["Voltage(V)"],
        "b-",
        linewidth=2,
    )
    plt.xlabel("Charge Time (s)", fontsize=12)
    plt.ylabel("Voltage (V)", fontsize=12)
    plt.title(f"Cycle {cycle} Charge Profile", fontsize=14)
    plt.grid(True, alpha=0.3)
    save_string = f"Cycle_{cycle}_charge_Crate_{cell_meta.c_rate_charge}_tempK_{cell_meta.temperature}_batteryID_{battery_id}.png"
    save_path = os.path.join(output_dir, save_string)
    fig.savefig(save_path, dpi=100, bbox_inches="tight")
    plt.close(fig)

    # Generate discharge plot
    fig = plt.figure(figsize=(10, 6))
    plt.plot(
        df_discharge["step_time(s)"],
        df_discharge["Voltage(V)"],
        "r-",
        linewidth=2,
    )
    plt.xlabel("Discharge Time (s)", fontsize=12)
    plt.ylabel("Voltage (V)", fontsize=12)
    plt.title(f"Cycle {cycle} Discharge Profile", fontsize=14)
    plt.grid(True, alpha=0.3)
    save_string = f"Cycle_{cycle}_discharge_Crate_{cell_meta.c_rate_discharge}_tempK_{cell_meta.temperature}_batteryID_{battery_id}.png"
    save_path = os.path.join(output_dir, save_string)
    fig.savefig(save_path, dpi=100, bbox_inches="tight")
    plt.close(fig)


def generate_figures_task(image_task):
    """Process a single image generation task - designed for multiprocessing."""
    try:
        df_charge, df_discharge, cell_meta, battery_id, cycle, output_dir = image_task

        # Import matplotlib here to avoid issues with multiprocessing
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        os.makedirs(output_dir, exist_ok=True)

        # Prepare data - normalize time to start from zero
        df_charge = df_charge.copy()
        df_discharge = df_discharge.copy()

        df_charge["step_time(s)"] = (
            df_charge["Test_Time(s)"] - df_charge["Test_Time(s)"].iloc[0]
        )
        df_discharge["step_time(s)"] = (
            df_discharge["Test_Time(s)"] - df_discharge["Test_Time(s)"].iloc[0]
        )

        # Generate charge plot
        fig = plt.figure(figsize=(10, 6))
        plt.plot(
            df_charge["step_time(s)"],
            df_charge["Voltage(V)"],
            "b-",
            linewidth=2,
        )
        plt.xlabel("Charge Time (s)", fontsize=12)
        plt.ylabel("Voltage (V)", fontsize=12)
        plt.title(f"Cycle {cycle} Charge Profile", fontsize=14)
        plt.grid(True, alpha=0.3)
        save_string = f"Cycle_{cycle}_charge_Crate_{cell_meta.c_rate_charge}_tempK_{cell_meta.temperature}_batteryID_{battery_id}.png"
        save_path = os.path.join(output_dir, save_string)
        fig.savefig(save_path, dpi=100, bbox_inches="tight")
        plt.close(fig)

        # Generate discharge plot
        fig = plt.figure(figsize=(10, 6))
        plt.plot(
            df_discharge["step_time(s)"],
            df_discharge["Voltage(V)"],
            "r-",
            linewidth=2,
        )
        plt.xlabel("Discharge Time (s)", fontsize=12)
        plt.ylabel("Voltage (V)", fontsize=12)
        plt.title(f"Cycle {cycle} Discharge Profile", fontsize=14)
        plt.grid(True, alpha=0.3)
        save_string = f"Cycle_{cycle}_discharge_Crate_{cell_meta.c_rate_discharge}_tempK_{cell_meta.temperature}_batteryID_{battery_id}.png"
        save_path = os.path.join(output_dir, save_string)
        fig.savefig(save_path, dpi=100, bbox_inches="tight")
        plt.close(fig)

        return f"Generated images for {battery_id} cycle {cycle}"

    except Exception as e:
        return f"Error generating images for {battery_id} cycle {cycle}: {str(e)}"


def clip_cv(input_df, direction):
    """Vectorized clipping: find first index where delta-current > 0.5 AND voltage crosses threshold.
    If no trigger found, return the full input_df unchanged.
    """
    if input_df.shape[0] < 2:
        return input_df.copy()

    curr = input_df["Current(A)"].to_numpy(dtype=float)
    volt = input_df["Voltage(V)"].to_numpy(dtype=float)

    # delta_current at position i corresponds to abs(curr[i] - curr[i-1]); set first element to 0
    delta_curr = np.abs(np.concatenate(([0.0], np.diff(curr))))

    # previous-voltage array (volt[i-1]) with first element = volt[0]
    volt_prev = np.concatenate(([volt[0]], volt[:-1]))

    if direction == "discharge":
        # trigger when delta_current > 0.5 and either current or previous voltage <= 2.5
        mask = (delta_curr > 0.5) & ((volt <= 2.5) | (volt_prev <= 2.5))
    elif direction == "charge":
        # trigger when delta_current > 0.5 and either current or previous voltage >= 3.5
        mask = (delta_curr > 0.5) & ((volt >= 3.5) | (volt_prev >= 3.5))
    else:
        return input_df.copy()

    idxs = np.where(mask)[0]
    if idxs.size == 0:
        # no trigger found -> return full dataframe
        return input_df.copy()

    stop_index = int(idxs[0])
    return input_df.iloc[: stop_index + 1].reset_index(drop=True)


def get_cell_metadata(meta_df: pd.DataFrame, battery_id: str) -> Optional[CellMetadata]:
    """Get cell metadata for a given battery ID."""
    cell_df = meta_df[meta_df["Battery_ID"].str.lower() == str.lower(battery_id)]

    if len(cell_df) == 0:
        print(f"⚠️  No metadata found for {battery_id}, using defaults")
        # Default values for MIT batteries (LFP)
        return CellMetadata(
            initial_capacity=1.1,  # 1.1 Ah
            c_rate_charge="fast",
            c_rate_discharge=4.0,
            temperature=303.0,  # 30°C = 303K
        )

    return CellMetadata(
        initial_capacity=cell_df["Initial_Capacity_Ah"].values[0],
        c_rate_charge=cell_df["C_rate_Charge"].values[0],
        c_rate_discharge=cell_df["C_rate_Discharge"].values[0],
        temperature=cell_df["Temperature (K)"].values[0],
    )


def scrub_data(
    input_df: pd.DataFrame,
    cell_meta: CellMetadata,
    battery_id: str,
    output_images_path: str,
) -> Tuple[pd.DataFrame, List]:
    """Process cycles, clip CV holds, and collect image generation tasks."""
    # Create battery-specific subdirectory
    battery_images_path = os.path.join(output_images_path, battery_id)
    os.makedirs(battery_images_path, exist_ok=True)

    agg_df = pd.DataFrame()
    image_tasks = []  # Collect image generation tasks
    cycles = input_df.Cycle_Count.unique()

    for cycle in cycles:
        cycle_df = input_df[input_df["Cycle_Count"] == cycle]
        df_charge_cycle = cycle_df[cycle_df["Current(A)"] > 0]
        df_discharge_cycle = cycle_df[cycle_df["Current(A)"] < 0]

        if len(df_charge_cycle) == 0 or len(df_discharge_cycle) == 0:
            continue

        # Clip CV holds
        df_charge_cycle = df_charge_cycle.reset_index(drop=True)
        df_discharge_cycle = df_discharge_cycle.reset_index(drop=True)
        df_charge_cycle = clip_cv(df_charge_cycle, "charge")
        df_discharge_cycle = clip_cv(df_discharge_cycle, "discharge")

        if df_charge_cycle is not None and df_discharge_cycle is not None:
            if len(df_charge_cycle) > 0 and len(df_discharge_cycle) > 0:
                try:
                    # Collect image generation task instead of generating immediately
                    image_task = (
                        df_charge_cycle,
                        df_discharge_cycle,
                        cell_meta,
                        battery_id,
                        cycle,
                        battery_images_path,
                    )
                    image_tasks.append(image_task)

                    # Append data to dataframe
                    cycle_df = pd.concat(
                        [df_charge_cycle, df_discharge_cycle], ignore_index=True
                    )
                    agg_df = pd.concat([agg_df, cycle_df], ignore_index=True)
                except Exception as e:
                    print(f"❌ Error processing cycle {cycle}: {e}")
                    continue

    return agg_df, image_tasks


def process_single_mat_file(
    mat_file: str,
    input_folder: str,
    meta_df: pd.DataFrame,
    config: ProcessingConfig,
) -> Tuple[List[str], Dict[str, str]]:
    """Process a single .mat file and return list of generated CSV files."""
    generated_csvs = []
    error_dict = {}

    print(f"\n📁 Processing MAT file: {mat_file}")

    try:
        # Generate cell CSVs from .mat file - save to same directory as .mat files
        gen_bat_df(input_folder, mat_file, input_folder)

        # Extract batch name (e.g., "MIT_20170512")
        batch_date = mat_file.split("_")[0]
        batch_name = f"MIT_{batch_date.replace('-', '')}"

        # Find generated CSV files for this .mat file (in the same directory as .mat files)
        csv_files = [
            f
            for f in os.listdir(input_folder)
            if f.startswith(batch_name) and f.endswith("_processed.csv")
        ]

        generated_csvs.extend(csv_files)
        print(f"✅ Generated {len(csv_files)} cell files from {mat_file}")

    except Exception as e:
        print(f"❌ Error processing {mat_file}: {e}")
        error_dict[mat_file] = str(e)

    return generated_csvs, error_dict


def process_single_cell_csv(
    csv_file: str,
    output_folder: str,
    meta_df: pd.DataFrame,
    config: ProcessingConfig,
) -> Dict[str, str]:
    """Process a single cell CSV file."""
    error_dict = {}

    try:
        # Read processed CSV from the same directory as .mat files
        cell_filepath = os.path.join(config.base_data_path, csv_file)
        cell_df = pd.read_csv(cell_filepath)

        # Extract battery ID from filename (remove "_processed.csv")
        battery_id = csv_file.replace("_processed.csv", "")

        print(f"\n🔋 Processing cell: {battery_id}")

        # Get cell metadata
        cell_meta = get_cell_metadata(meta_df, battery_id)
        if cell_meta is None:
            return error_dict

        # Process data and collect image tasks
        agg_df, image_tasks = scrub_data(
            cell_df,
            cell_meta,
            battery_id,
            config.output_images_path,
        )

        if len(agg_df) > 0:
            # Save aggregated data
            csv_filename = f"{battery_id.lower()}_aggregated_data.csv"
            csv_path = os.path.join(config.output_data_path, csv_filename)
            os.makedirs(os.path.dirname(csv_path), exist_ok=True)
            agg_df.to_csv(csv_path, index=False)
            print(f"💾 Saved: {csv_path}")

            # Return image tasks for later processing
            return error_dict, image_tasks
        else:
            print(f"⚠️  No valid data for {battery_id}")
            return error_dict, []

    except Exception as e:
        print(f"❌ Error processing {csv_file}: {e}")
        error_dict[csv_file] = str(e)

    return error_dict


def save_error_log(error_dict: Dict[str, str], config: ProcessingConfig) -> None:
    """Save error log for all batteries."""
    if not error_dict:
        return

    error_df = pd.DataFrame(
        list(error_dict.items()), columns=["File_Name", "Error_Message"]
    )
    error_log_path = os.path.join(config.output_data_path, "error_log_mit.csv")
    os.makedirs(os.path.dirname(error_log_path), exist_ok=True)
    error_df.to_csv(error_log_path, index=False)
    print(f"📝 Saved error log: {error_log_path}")


def check_existing_csv_files(config: ProcessingConfig) -> List[str]:
    """Check if processed CSV files already exist and return list of existing files."""
    # Check in the same directory as .mat files (base_data_path)
    if not os.path.exists(config.base_data_path):
        return []

    # Look for all processed CSV files in the .mat files directory
    existing_csvs = [
        f for f in os.listdir(config.base_data_path) if f.endswith("_processed.csv")
    ]

    print(
        f"🔍 Found {len(existing_csvs)} existing processed CSV files in {config.base_data_path}"
    )
    return existing_csvs


def main(config: Optional[ProcessingConfig] = None) -> None:
    """Main function to process all MIT files with multi-threading."""
    if config is None:
        config = ProcessingConfig()

    # Start timing
    start_time = time.time()
    print("🚀 Starting MIT battery data processing with 10 threads...")

    # Load metadata
    meta_df = load_meta_properties()

    # Setup paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.join(script_dir, "..", "..")
    mit_base_path = os.path.join(project_root, config.base_data_path)

    # Update config paths to absolute paths
    config.output_data_path = os.path.join(project_root, config.output_data_path)
    config.output_images_path = os.path.join(project_root, config.output_images_path)

    # Check for existing processed CSV files
    existing_csvs = check_existing_csv_files(config)

    # Get all .mat files
    mat_files = [f for f in os.listdir(mit_base_path) if f.endswith(".mat")]
    print(f"📂 Found {len(mat_files)} MIT .mat files")

    # Collect all errors
    all_errors = {}
    all_csv_files = []

    # Step 1: Process .mat files to generate CSVs (only if needed)
    if existing_csvs:
        print("\n" + "=" * 60)
        print("Step 1: Using existing processed CSV files")
        print("=" * 60)
        all_csv_files = existing_csvs
        print(
            f"✅ Skipping .mat conversion, using {len(existing_csvs)} existing CSV files"
        )
    else:
        print("\n" + "=" * 60)
        print("Step 1: Converting .mat files to CSVs")
        print("=" * 60)

        with ThreadPoolExecutor(max_workers=5) as executor:
            future_to_mat = {
                executor.submit(
                    process_single_mat_file, mat_file, mit_base_path, meta_df, config
                ): mat_file
                for mat_file in mat_files
            }

            for future in as_completed(future_to_mat):
                mat_file = future_to_mat[future]
                try:
                    csv_files, error_dict = future.result()
                    all_csv_files.extend(csv_files)
                    all_errors.update(error_dict)
                except Exception as e:
                    print(f"✗ Error processing {mat_file}: {str(e)}")
                    all_errors[mat_file] = str(e)

    # Step 2: Process generated CSV files (multithreaded)
    print(f"\n{'='*60}")
    print(f"Step 2: Processing {len(all_csv_files)} cell CSV files")
    print("=" * 60)

    completed_count = 0
    all_image_tasks = []  # Collect all image generation tasks

    with ThreadPoolExecutor(max_workers=10) as executor:
        future_to_csv = {
            executor.submit(
                process_single_cell_csv,
                csv_file,
                mit_base_path,  # Use mit_base_path since processed.csv files are now in the same directory as .mat files
                meta_df,
                config,
            ): csv_file
            for csv_file in all_csv_files
        }

        for future in as_completed(future_to_csv):
            csv_file = future_to_csv[future]
            try:
                error_dict, image_tasks = future.result()
                all_errors.update(error_dict)
                all_image_tasks.extend(image_tasks)  # Collect image tasks
                completed_count += 1
                print(
                    f"✅ [{completed_count}/{len(all_csv_files)}] Completed: {csv_file}"
                )
            except Exception as e:
                print(f"✗ Error processing {csv_file}: {str(e)}")
                all_errors[csv_file] = str(e)
                completed_count += 1

    # Step 3: Generate images using process pool (CPU-intensive task)
    if all_image_tasks:
        print(f"\n{'='*60}")
        print(f"Step 3: Generating {len(all_image_tasks)} images using process pool")
        print("=" * 60)

        # Use process pool for image generation (CPU-intensive)
        max_processes = min(os.cpu_count() or 1, 8)  # Limit to 8 processes max
        print(f"🖼️  Using {max_processes} processes for image generation")

        with ProcessPoolExecutor(max_workers=max_processes) as executor:
            future_to_image = {
                executor.submit(generate_figures_task, task): task
                for task in all_image_tasks
            }

            image_completed = 0
            for future in as_completed(future_to_image):
                task = future_to_image[future]
                try:
                    result = future.result()
                    image_completed += 1
                    if image_completed % 10 == 0:  # Print progress every 10 images
                        print(
                            f"🖼️  [{image_completed}/{len(all_image_tasks)}] Images generated"
                        )
                except Exception as e:
                    print(f"✗ Error generating images for task: {str(e)}")

        print(f"✅ Generated {len(all_image_tasks)} images successfully")
    else:
        print("ℹ️  No images to generate")

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
    print(f"🎉 All MIT files processed successfully!")
    print(f"⏱️  Total processing time: {hours:02d}:{minutes:02d}:{seconds:02d}")
    print(
        f"📊 Processed {len(mat_files)} .mat files → {len(all_csv_files)} cells with 10 threads"
    )
    if len(all_csv_files) > 0:
        print(f"⚡ Average time per cell: {total_time/len(all_csv_files):.2f} seconds")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
# ============================================================
# 🎉 All MIT files processed successfully!
# ⏱️  Total processing time: 01:20:03
# 📊 Processed 4 .mat files → 140 cells with 10 threads
# ⚡ Average time per cell: 34.31 seconds
# ============================================================
