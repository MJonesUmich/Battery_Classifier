import time
import os
import random

import numpy as np
import pandas as pd


def extract_cycles(
    input_df, max_cycles, file_name, output_folder, downsample_ratio=0.20, seed=42
):
    np.random.seed(seed)
    # detect cycle index column (support multiple naming conventions)
    cols_map = {c.lower(): c for c in input_df.columns}
    cycle_col = None
    for cand in ["cycle_index", "cycle_count", "cycle", "cycle_number", "cycle id", "cycle index"]:
        if cand in cols_map:
            cycle_col = cols_map[cand]
            break
    if cycle_col is None:
        print(f"Warning: no cycle index column found in {file_name}; looked for cycle_index/Cycle_Count/etc.")
        return pd.DataFrame(columns=input_df.columns)

    unique_cycles = input_df[cycle_col].unique()
    interval_step = max(1, int(1.0 / downsample_ratio))

    # If fewer cycles than max_cycles, take all
    if len(unique_cycles) <= max_cycles:
        selected_cycles = unique_cycles
    else:
        selected_cycles = np.random.choice(
            unique_cycles, size=max_cycles, replace=False
        )

    # Collect downsampled data
    all_cycles = []
    for cycle in selected_cycles:
        sub_df = input_df[input_df["cycle_index"] == cycle].reset_index(drop=True)
        downsampled_df = sub_df.iloc[::interval_step, :].reset_index(drop=True)
        all_cycles.append(downsampled_df)

    # Combine into one DataFrame
    if not all_cycles:
        # no cycles selected or no matching cycle indices found
        print(f"Warning: no cycles found in {file_name} (returning empty DataFrame)")
        # return empty DataFrame with same columns as input (if available)
        try:
            return pd.DataFrame(columns=input_df.columns)
        except Exception:
            return pd.DataFrame()

    combined_df = pd.concat(all_cycles, ignore_index=True)
    return combined_df


def create_folder_structure(
    processed_base_dir, model_folders, exclusion_list, output_parent_dir
):
    """
    Create continuous_model_prep/<train|val|test>/<chemistry> folders.
    - processed_base_dir: path to processed_datasets (we read chemistry folders from here)
    - output_parent_dir: parent directory where continuous_model_prep will be created
    """
    error_substr = "error_log"
    # Only consider directories inside processed_datasets (skip files and excluded names)
    base_entries = os.listdir(processed_base_dir)
    chemistry_folders = [
        folder
        for folder in base_entries
        if folder not in exclusion_list
        and error_substr not in folder.lower()
        and os.path.isdir(os.path.join(processed_base_dir, folder))
    ]

    # Create model_prep under the provided output_parent_dir
    for model_folder in model_folders:
        folder_path = os.path.join(output_parent_dir, "model_prep", model_folder)
        for chemistry_folder in chemistry_folders:
            chemistry_path = os.path.join(folder_path, chemistry_folder)
            os.makedirs(chemistry_path, exist_ok=True)


def transfer_files(
    train_val_test_dict,
    base_input_folder,
    base_output_folderpath,
    chemistry,
    downsample_ratio=0.20,
    max_cycles=100,
):
    """
    base_input_folder should be the processed_datasets path.
    base_output_folderpath should be the path to continuous_model_prep (not its parent).
    """
    error_substr = "error_log"
    for split, file_list in train_val_test_dict.items():
        dest_dir = os.path.join(base_output_folderpath, split, chemistry)
        os.makedirs(dest_dir, exist_ok=True)
        copied = 0
        for cell_id in file_list:
            # cell_id is expected to be a directory name under base_input_folder/chemistry
            cell_dir = os.path.join(base_input_folder, chemistry, cell_id)
            if not os.path.exists(cell_dir):
                # maybe file_list contained file names; try treating cell_id as filename directly
                candidate = os.path.join(base_input_folder, chemistry, cell_id)
                if os.path.exists(candidate) and candidate.lower().endswith(".csv"):
                    csv_files = [cell_id]
                    cell_dir = os.path.join(base_input_folder, chemistry)
                else:
                    print(f"Cell entry not found, skipping: {cell_dir}")
                    continue
            else:
                # list csv files inside the cell directory
                try:
                    entries = os.listdir(cell_dir)
                except Exception:
                    print(f"Unable to list directory, skipping: {cell_dir}")
                    continue
                csv_files = [f for f in entries if f.lower().endswith(".csv")]

            if not csv_files:
                print(f"No CSV files found for cell {cell_id} under {chemistry}")
                continue

            for file in csv_files:
                # Skip any files that contain "error_log" (case-insensitive)
                if error_substr in file.lower():
                    continue

                src_path = os.path.join(cell_dir, file)
                # protect against non-existent source (extra check)
                if not os.path.exists(src_path):
                    print(f"Source missing, skipping: {src_path}")
                    continue

                try:
                    read_df = pd.read_csv(src_path)
                except Exception as e:
                    print(f"Failed to read {src_path}: {e}")
                    continue

                out_name = os.path.basename(file)
                out_path = os.path.join(dest_dir, out_name)
                out_path = (
                    out_path[0:-4]
                    + f"_{int(max_cycles)}_cycles_downsampled_to_{int(downsample_ratio*100)}_pct.csv"
                )

                #print(list(read_df.columns))
                combined_df = extract_cycles(
                    read_df,
                    max_cycles,
                    file,
                    out_path,
                    downsample_ratio=downsample_ratio,
                    seed=42,
                )

                #Specify Charge/Discharge Designation 
                if "discharge" in file: 
                    combined_df["direciton"] = 0
                else: 
                    combined_df["direction"] = 1

                #Remove Unwanted Features: 
                drop_cols = ["battery_id", "sample_index", "elapsed_time_s", "current_a", "temperature_k", "chemistry"]
                output_combined_df = combined_df.drop(columns=drop_cols).copy()

                if os.path.abspath(src_path) == os.path.abspath(out_path):
                    print(f"Skipping same-file: {src_path}")
                    continue
                if os.path.exists(out_path):
                    print(f"Destination exists, skipping: {out_path}")
                    continue
                
                if len(output_combined_df.dropna())>0 and "variable" not in output_combined_df["c_rate"].values: 
                    output_combined_df.to_csv(out_path, index=False)
                    copied += 1
        print(f"Copied {copied} files to {split}/{chemistry}")


def prep_train_test_split(input_folderpath, random_seed=42):
    random.seed(random_seed)
    unique_cell_ids = os.listdir(input_folderpath)
    # remove known non-data files and anything containing "error_log"
    unique_cell_ids = [
        cell for cell in unique_cell_ids if "error_log" not in cell.lower()
    ]
    unique_cell_ids = [
        cell
        for cell in unique_cell_ids
        if os.path.isfile(os.path.join(input_folderpath, cell))
        or os.path.isdir(os.path.join(input_folderpath, cell))
    ]
    unique_cell_ids.sort()  # stable order before shuffle
    random.shuffle(unique_cell_ids)
    total_ids = len(unique_cell_ids)
    train_end = int(total_ids * 0.8)
    val_end = train_end + int(total_ids * 0.1)
    train_ids = unique_cell_ids[:train_end]
    val_ids = unique_cell_ids[train_end:val_end]
    test_ids = unique_cell_ids[val_end:]
    return {"train": train_ids, "val": val_ids, "test": test_ids}


def main(base_dir, chemistries):
    """
    base_dir should point to the processed_datasets directory.
    This function will create continuous_model_prep as a sibling of processed_datasets.
    """
    model_folders = ["train", "val", "test"]
    exclusion_list = [
        "all_data",
        "continuous_model_prep",
        "stored_models",
        "error_log.csv",
    ]

    # place model_prep inside this package directory (ts_classification/model_prep)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_output_folderpath = os.path.join(script_dir, "model_prep")

    # Create folder structure under script_dir/model_prep
    create_folder_structure(base_dir, model_folders, exclusion_list, script_dir)

    # Now transfer files: base_input_folder is processed_datasets (base_dir)
    for chemistry in chemistries:
        input_folderpath = os.path.join(base_dir, chemistry)
        if not os.path.isdir(input_folderpath):
            print(f"Chemistry folder not found, skipping: {input_folderpath}")
            continue

        train_val_test_dict = prep_train_test_split(input_folderpath, random_seed=42)
        transfer_files(
            train_val_test_dict,
            base_dir,  # processed_datasets
            base_output_folderpath,
            chemistry,
            downsample_ratio=0.25,
            max_cycles=100,
        )


if __name__ == "__main__":
    start = time.time()
    # Default base_dir: repo-root/processed_datasets
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.abspath(os.path.join(script_dir, "..", "..", "..", ".."))
    base_dir = os.path.join(repo_root, "BATTERY_CLASSIFIER", "assets", "processed")
    chemistries = ["NMC", "NCA", "LFP", "LCO"]

    if not os.path.isdir(base_dir):
        print(f"Processed datasets folder not found at {base_dir}. Please check path or call main() with the correct base_dir.")
    else:
        print(f"Using processed datasets folder: {base_dir}")
        main(base_dir, chemistries)
    end = time.time() 
    time_delta = np.round((end - start)/60,1)
    print(f'File Copying Complete in {time_delta} minutes')


# ============================================================
# 🎉 Training,Validation, and Test File Structure Completed!
# ⏱️  Total processing time: 00:01:54
# ============================================================
