import os 
import random
import pandas as pd
import numpy as np
 

def extract_cycles(input_df, max_cycles, file_name, output_folder, downsample_ratio=0.20, seed=42):
    np.random.seed(seed)
    unique_cycles = input_df['Cycle_Count'].unique()
    interval_step = max(1, int(1.0 / downsample_ratio))

    # If fewer cycles than max_cycles, take all
    if len(unique_cycles) <= max_cycles:
        selected_cycles = unique_cycles
    else:
        selected_cycles = np.random.choice(unique_cycles, size=max_cycles, replace=False)
    
    # Collect downsampled data
    all_cycles = []
    for cycle in selected_cycles:
        sub_df = input_df[input_df['Cycle_Count'] == cycle].reset_index(drop=True)
        downsampled_df = sub_df.iloc[::interval_step, :].reset_index(drop=True)
        all_cycles.append(downsampled_df)

    # Combine into one DataFrame
    combined_df = pd.concat(all_cycles, ignore_index=True)
    return combined_df


def create_folder_structure(base_dir, model_folders, exclusion_list):
    base_folders = os.listdir(base_dir)
    chemistry_folders = [folder for folder in base_folders if folder not in exclusion_list]
    for model_folder in model_folders: 
        folder_path = os.path.join(base_dir, 'model_prep', model_folder)
        for chemistry_folder in chemistry_folders: 
            chemistry_path = os.path.join(folder_path, chemistry_folder)
            os.makedirs(chemistry_path, exist_ok=True)


def transfer_files(train_val_test_dict, base_input_folder, 
                   base_output_folderpath, chemistry, downsample_ratio=0.20, max_cycles=100):
    # protect downsample_ratio and compute integer step

    for split, file_list in train_val_test_dict.items():
        dest_dir = os.path.join(base_output_folderpath, split, chemistry)
        os.makedirs(dest_dir, exist_ok=True)
        copied = 0
        for file in file_list:
            src_path = os.path.join(base_input_folder, chemistry, file)
            read_df = pd.read_csv(src_path)
            out_path = os.path.join(dest_dir, file)
            out_path = out_path[0:-4] + f"_{int(max_cycles)}_cycles_downsampled_to_{int(downsample_ratio*100)}_pct.csv"
            combined_df = extract_cycles(read_df, max_cycles, file, out_path, downsample_ratio=0.20, seed=42)
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
    unique_cell_ids = [cell for cell in unique_cell_ids if cell != "error_log.csv"]
    unique_cell_ids.sort()          # stable order before shuffle
    random.shuffle(unique_cell_ids)
    total_ids = len(unique_cell_ids)
    train_end = int(total_ids * 0.8)
    val_end = train_end + int(total_ids * 0.1)
    train_ids = unique_cell_ids[:train_end]
    val_ids = unique_cell_ids[train_end:val_end]
    test_ids = unique_cell_ids[val_end:]
    return {"train": train_ids, "val": val_ids, "test": test_ids}


def main(base_dir, chemistries): 
    model_folders = ['train', 'val', 'test']
    exclusion_list = ['all_data', 'model_prep', 'stored_models']
    base_output_folderpath = os.path.join(base_dir, 'model_prep')
    create_folder_structure(base_dir, model_folders, exclusion_list)

    for chemistry in chemistries: 
        input_folderpath = os.path.join(base_dir, chemistry)
        train_val_test_dict = prep_train_test_split(input_folderpath,random_seed=42)
        transfer_files(train_val_test_dict, base_dir, base_output_folderpath, chemistry, 
                       downsample_ratio=0.25, max_cycles=100 )


if __name__ == "__main__":
    base_dir = r'C:\Users\MJone\Documents\SIADS699\processed_datasets'
    chemistries = ["NMC", "LFP", "LCO"]

    main(base_dir, chemistries)