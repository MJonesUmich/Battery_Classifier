import os 
import random 
import shutil 
import re


def get_folders(parent_path, exclude_folders):
    """Get chemistry folders from root path, excluding specified folders and their subfolders."""
    # Get immediate subdirectories only
    chemistry_folders = []
    for name in os.listdir(parent_path):
        full_path = os.path.join(parent_path, name)
        # Check if it's a directory and not in excluded list
        if os.path.isdir(full_path) and name not in exclude_folders:
            chemistry_folders.append(full_path)
    
    if not chemistry_folders:
        raise ValueError(f"No valid chemistry folders found in {parent_path}")
    
    # Count PNG files in each chemistry folder
    folder_counts = {}
    for folder in chemistry_folders:
        # Only count PNG files in the immediate directory
        png_count = len([f for f in os.listdir(folder) 
                        if f.endswith('.png') 
                        and os.path.isfile(os.path.join(folder, f))])
        folder_counts[folder] = png_count
        print(f"Found {png_count} PNG files in {os.path.basename(folder)}")
    
    lowest_count = min(folder_counts.values()) if folder_counts else 0
    print(f"Smallest chemistry folder contains {lowest_count} images")
    
    return chemistry_folders, lowest_count


def delete_existing_images(input_folder_path): 
    if os.path.exists(input_folder_path):
        shutil.rmtree(input_folder_path)
        print(f"Deleted existing folder: {input_folder_path}")


def get_unique_cell_ids(input_folderpath):
    unique_files = os.listdir(input_folderpath)
    unique_cell_ids = set()
    for file in unique_files:
        match = re.search(r'batteryID_(.+)\.png', file)
        if match:
            battery_id = match.group(1)
            unique_cell_ids.add(battery_id)             
    return unique_cell_ids


def get_train_val_test(chemistry_folders, parent_path, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    train_val_test_total_dict = {}

    for chemistry in chemistry_folders:
        if os.path.exists(os.path.join(parent_path, chemistry)) is True:
            chemistry_path = os.path.join(parent_path, chemistry)
            unique_cell_ids = list(get_unique_cell_ids(chemistry_path))
            random.shuffle(unique_cell_ids)
            total_ids = len(unique_cell_ids)
            train_end = int(total_ids * train_ratio)
            val_end = train_end + int(total_ids * val_ratio)
            
            train_ids = unique_cell_ids[:train_end]
            val_ids = unique_cell_ids[train_end:val_end]
            test_ids = unique_cell_ids[val_end:]

            chem_path_files = os.listdir(chemistry_path)
            train_files = [file for file in chem_path_files for train_id in train_ids if train_id in file]
            val_files = [file for file in chem_path_files for val_id in val_ids if val_id in file]
            test_files = [file for file in chem_path_files for test_id in test_ids if test_id in file]
            
            train_val_test_total_dict[chemistry] = {
                'train': train_files,
                'val': val_files,
                'test': test_files
            }

    return train_val_test_total_dict


def transfer_files(train_val_test_total_dict, parent_path, store_path):
    """
    Transfer files to chemistry-specific subfolders within train/val/test directories.
    Structure: model_prep/[train|val|test]/[chemistry]/image.png
    """
    for chemistry, train_val_test_dict in train_val_test_total_dict.items():
        transfer_counter = {'train': 0, 'val': 0, 'test': 0}
        for train_val_test_selection, file_list in train_val_test_dict.items():
            for file in file_list:
                specific_file_name = os.path.basename(file)
                src_path = os.path.join(parent_path, chemistry, file)
                base_chem = os.path.basename(chemistry)
                dest_dir = os.path.join(store_path, train_val_test_selection, base_chem)
                os.makedirs(dest_dir, exist_ok=True)
                dest_filepath = os.path.join(dest_dir, specific_file_name)
                shutil.copy2(src_path, dest_filepath)
                transfer_counter[train_val_test_selection] += 1
        for split_type, transfer_count in transfer_counter.items():
            print(f"Transferred {transfer_count} images to {split_type}/{chemistry}")


def main(parent_path, exclude_folders): 
    chemistry_folders, lowest_count = get_folders(parent_path, exclude_folders)
    del_path = os.path.join(parent_path, 'model_prep')
    print(f'Removing any prior training images from {del_path}.....')
    delete_existing_images(del_path)

    print("Separating files into train, val, and test randomly by unique cell id.....")
    train_val_test_total_dict = get_train_val_test(chemistry_folders, parent_path)

    print('Transfering Files.....')
    transfer_files(train_val_test_total_dict, parent_path, store_path=del_path)


if __name__ == "__main__":
    parent_path = r'C:\Users\MJone\Documents\SIADS699\processed_images'
    exclude_folders = ['all_images', 'model_prep', 'stored_models']
    chemistry_folders = ['LFP', 'LCO', 'NCA', 'NMC']
    main(parent_path, exclude_folders)