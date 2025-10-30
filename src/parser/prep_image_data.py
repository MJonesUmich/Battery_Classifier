import os 
import random
import shutil


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


def transfer_files(folder_path, image_dict):
    """
    Transfer files to chemistry-specific subfolders within train/val/test directories.
    Structure: model_prep/[train|val|test]/[chemistry]/image.png
    """
    for split_type, image_paths in image_dict.items():
        # Count successful transfers for reporting
        transfer_count = 0
        
        for img_path in image_paths:
            try:
                # Get chemistry type from parent folder name
                chemistry = os.path.basename(os.path.dirname(img_path))
                
                # Create full destination path: split_type/chemistry/
                chem_dest = os.path.join(folder_path, split_type, chemistry)
                os.makedirs(chem_dest, exist_ok=True)
                
                # Copy to chemistry-specific subfolder
                dest_path = os.path.join(chem_dest, os.path.basename(img_path))
                shutil.copy2(img_path, dest_path)
                transfer_count += 1
                
            except Exception as e:
                print(f"Error copying {img_path}: {str(e)}")
                continue
        
        print(f"Transferred {transfer_count} images to {split_type}/{chemistry}")


def sample_data(chemistry_folders, lowest_count, store_path, seed=42):
    """Sample and distribute images maintaining chemistry balance."""
    random.seed(seed)
    print('--------- Sampling files ---------')

    for folder in chemistry_folders:
        chemistry = os.path.basename(folder)
        print(f'Processing {chemistry}...')
        
        # Get only PNG files from immediate directory
        all_images = [f for f in os.listdir(folder) 
                     if f.endswith('.png') 
                     and os.path.isfile(os.path.join(folder, f))]
        all_image_paths = [os.path.join(folder, image) for image in all_images]
        
        if len(all_image_paths) > lowest_count:
            all_image_paths = random.sample(all_image_paths, lowest_count)

        # Calculate splits
        train_size = int(0.8 * lowest_count)
        val_size = int(0.1 * lowest_count)
        test_size = lowest_count - train_size - val_size

        # Randomly split images
        random.shuffle(all_image_paths)
        train_images = all_image_paths[:train_size]
        val_images = all_image_paths[train_size:train_size + val_size]
        test_images = all_image_paths[train_size + val_size:]

        print(f"{chemistry}: train={len(train_images)}, val={len(val_images)}, test={len(test_images)}")
        
        # Transfer files
        image_dict = {'train': train_images, 'val': val_images, 'test': test_images}
        transfer_files(store_path, image_dict)


if __name__ == "__main__":
    parent_path = r'C:\Users\MJone\Documents\SIADS699\processed_images'
    exclude_folders = ['all_images', 'model_prep']
    store_path = os.path.join(parent_path, 'model_prep')
    
    chemistry_folders, lowest_count = get_folders(parent_path, exclude_folders)
    print(f'Found {len(chemistry_folders)} chemistry folders')
    print(f'Will sample {lowest_count} images from each')
    
    if lowest_count > 0:
        sample_data(chemistry_folders, lowest_count, store_path, seed=42)
    else:
        print("No images found in chemistry folders")