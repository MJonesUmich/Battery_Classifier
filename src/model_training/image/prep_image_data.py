import os
import random
import shutil


def get_folders(parent_path, exclude_folders):
    """Get chemistry folders from root path, excluding specified ones."""
    chemistry_folders = []
    for name in os.listdir(parent_path):
        full_path = os.path.join(parent_path, name)
        if os.path.isdir(full_path) and name not in exclude_folders:
            chemistry_folders.append(full_path)

    if not chemistry_folders:
        raise ValueError(f"No valid chemistry folders found in {parent_path}")

    print(f"Found {len(chemistry_folders)} chemistry folders.")
    return chemistry_folders


def delete_existing_images(input_folder_path): 
    if os.path.exists(input_folder_path):
        shutil.rmtree(input_folder_path)
        print(f"Deleted existing folder: {input_folder_path}")


def get_cell_folders(chemistry_folder):
    """Return all cell_id subfolders under a chemistry folder."""
    return [
        os.path.join(chemistry_folder, d)
        for d in os.listdir(chemistry_folder)
        if os.path.isdir(os.path.join(chemistry_folder, d))
    ]


def sample_images_from_cell(cell_folder, n):
    """Randomly select up to n images from a cell folder."""
    all_images = [
        os.path.join(cell_folder, f)
        for f in os.listdir(cell_folder)
        if f.endswith(".png")
    ]
    return random.sample(all_images, min(n, len(all_images)))


def get_train_val_test(chemistry_folders, n_per_cell, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2):
    """Split images into train/val/test for each chemistry folder."""
    all_data = {}

    for chemistry in chemistry_folders:
        chemistry_name = os.path.basename(chemistry)
        cell_folders = get_cell_folders(chemistry)
        all_images = []

        # Collect images from each cell folder
        for cell_folder in cell_folders:
            sampled_imgs = sample_images_from_cell(cell_folder, n_per_cell)
            all_images.extend(sampled_imgs)

        random.shuffle(all_images)
        total = len(all_images)
        train_end = int(total * train_ratio)
        val_end = train_end + int(total * val_ratio)

        all_data[chemistry_name] = {
            "train": all_images[:train_end],
            "val": all_images[train_end:val_end],
            "test": all_images[val_end:]
        }

        print(f"{chemistry_name}: {len(all_images)} total images → "
              f"{len(all_data[chemistry_name]['train'])} train, "
              f"{len(all_data[chemistry_name]['val'])} val, "
              f"{len(all_data[chemistry_name]['test'])} test")

    return all_data


def transfer_files(split_dict, store_path):
    """Copy images into train/val/test/chemistry subfolders."""
    for chemistry, splits in split_dict.items():
        for split_name, files in splits.items():
            dest_dir = os.path.join(store_path, split_name, chemistry)
            os.makedirs(dest_dir, exist_ok=True)
            for src_file in files:
                dest_file = os.path.join(dest_dir, os.path.basename(src_file))
                shutil.copy2(src_file, dest_file)
            print(f"Transferred {len(files)} → {split_name}/{chemistry}")


def main():
    parent_path = os.path.join("..", "..", "..", "assets", "images")
    exclude_folders = ['all_images', 'model_prep', 'stored_models']
    n_per_cell = 1  # number of images per cell folder

    chemistry_folders = get_folders(parent_path, exclude_folders)

    # ✅ store_path now correctly points inside src/model_training/image/
    store_path = os.path.join("model_prep")

    delete_existing_images(store_path)

    split_dict = get_train_val_test(chemistry_folders, n_per_cell)
    transfer_files(split_dict, store_path)


if __name__ == "__main__":
    main()
