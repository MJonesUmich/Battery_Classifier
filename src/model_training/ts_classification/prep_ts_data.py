import os
import random

import numpy as np
import pandas as pd


def extract_cycles(
    input_df, max_cycles, file_name, output_folder, downsample_ratio=0.20, seed=42
):
    np.random.seed(seed)
    unique_cycles = input_df["Cycle_Count"].unique()
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
        sub_df = input_df[input_df["Cycle_Count"] == cycle].reset_index(drop=True)
        downsampled_df = sub_df.iloc[::interval_step, :].reset_index(drop=True)
        all_cycles.append(downsampled_df)

    # Combine into one DataFrame
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

    for model_folder in model_folders:
        folder_path = os.path.join(
            output_parent_dir, "continuous_model_prep", model_folder
        )
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
        for file in file_list:
            # Skip any files that contain "error_log" (case-insensitive)
            if error_substr in file.lower():
                continue

            # Only process csv files
            if not file.lower().endswith(".csv"):
                continue

            src_path = os.path.join(base_input_folder, chemistry, file)
            # protect against non-existent source
            if not os.path.exists(src_path):
                print(f"Source missing, skipping: {src_path}")
                continue

            try:
                read_df = pd.read_csv(src_path)
            except Exception as e:
                print(f"Failed to read {src_path}: {e}")
                continue

            out_path = os.path.join(dest_dir, file)
            out_path = (
                out_path[0:-4]
                + f"_{int(max_cycles)}_cycles_downsampled_to_{int(downsample_ratio*100)}_pct.csv"
            )

            combined_df = extract_cycles(
                read_df,
                max_cycles,
                file,
                out_path,
                downsample_ratio=downsample_ratio,
                seed=42,
            )

            if os.path.abspath(src_path) == os.path.abspath(out_path):
                print(f"Skipping same-file: {src_path}")
                continue
            if os.path.exists(out_path):
                print(f"Destination exists, skipping: {out_path}")
                continue

            combined_df.to_csv(out_path, index=False)
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

    # parent dir of processed_datasets
    parent_dir = os.path.dirname(base_dir)
    # full path to the continuous_model_prep folder (we pass this to transfer_files)
    base_output_folderpath = os.path.join(parent_dir, "continuous_model_prep")

    # Create folder structure based on chemistry folders inside processed_datasets only
    create_folder_structure(base_dir, model_folders, exclusion_list, parent_dir)

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
    # Base dir points to processed_datasets
    base_dir = os.path.join("..", "..", "processed_datasets")
    chemistries = ["NMC", "NCA", "LFP", "LCO"]
    main(base_dir, chemistries)
