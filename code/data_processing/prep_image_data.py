import os 
import random
import shutil


def get_folders(parent_path, store_path):
    #get folder path for each of the chemistries 
    chemistry_folders = [
        name for name in os.listdir(parent_path)
        if os.path.isdir(os.path.join(parent_path, name))
        and name != store_path
    ]

    chemistry_folders = [os.path.join(parent_path,folder) for folder in chemistry_folders]
    print('folder for lowst_counts: ', chemistry_folders)
    #Find the lowest set if images allowed to downsample training, val, and test 
    lowest_count = None
    for folder in chemistry_folders: 
        num_files = len(os.listdir(folder))
        print(num_files)
        if lowest_count == None: 
            lowest_count = num_files 
        else: 
            if num_files < lowest_count: 
                lowest_count = num_files 
    return chemistry_folders, lowest_count


def transfer_files(folder_path, image_dict):
    #Dump the chemistry's train, val, and test to the desired location
    for subfolder, image_paths in image_dict.items():
        temp_folder = os.path.join(folder_path, subfolder)
        os.makedirs(temp_folder, exist_ok=True)
        for img_path in image_paths:
            shutil.copy(img_path, temp_folder)



def sample_data(chemistry_folders, lowest_count, store_path, seed=42):
    random.seed(seed)
    print('--------- Sampling files ---------')

    for folder in chemistry_folders:
        print('folder:  ', folder, 'chem_folder: ', chemistry_folders)
        if folder != store_path:
            all_images = [image for image in os.listdir(folder) if image.endswith('.png')]
            all_image_paths = [os.path.join(folder, image) for image in all_images]
            print('len img path', len(all_image_paths), ' folder: ', folder)
            if len(all_image_paths) > lowest_count:
                all_image_paths = random.sample(all_image_paths, lowest_count)

            # Get train, val, and test sizes
            train_size = int(0.8 * lowest_count)
            val_size = int(0.1 * lowest_count)
            test_size = lowest_count - train_size - val_size  # ensures total = lowest_count

            # Partition the data accordingly
            print(train_size, val_size, test_size)
            train_images = random.sample(all_image_paths, train_size)
            remaining = [img for img in all_image_paths if img not in train_images]

            val_images = random.sample(remaining, val_size)
            test_images = [img for img in remaining if img not in val_images]

            # Transfer data to appropriate directory 
            image_dict = {'train': train_images, 'val': val_images, 'test': test_images}
            transfer_files(store_path, image_dict)


if __name__ == "__main__": 
    parent_path = r'C:\Users\MJone\Documents\SIADS699\processed_images'
    store_path = os.path.join(parent_path, r'model_prep')
    chemistry_folders, lowest_count = get_folders(parent_path, store_path)
    print('lowest count: ', lowest_count)
    print(chemistry_folders)
    sample_data(chemistry_folders, lowest_count, store_path, seed=42)