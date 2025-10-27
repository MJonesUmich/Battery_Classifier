import os 
import re 
import pandas as pd


def load_meta_data(): 
    #Load battery mapper data: 
    sheet_id = "19L7_7HpOUagvRAh6GNOrhcjQpbEu97kx"
    url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=xlsx"
    meta_df = pd.read_excel(url, sheet_name='Sheet1')  
    return meta_df 


def extract_labels(file):
    direction = "discharge" if "discharge" in file.lower() else "charge"
    regex_patterns = [r"_tempK_(.*?)_batteryID_",
                        r"batteryID_(.*?).png",
                        r"Crate_(.*?)_tempK_",
                        ]

    matches = []
    for pattern in regex_patterns:
        m = re.search(pattern, file, re.IGNORECASE)
        matches.append(m.group(1) if m else None)

    tempK, batteryID, Crate = matches
    return tempK, batteryID, Crate, direction


def extract_chemistry(meta_df, input_batteryID): 
    temp_batteryID = input_batteryID.lower() 
    temp_df = meta_df[meta_df["Battery_ID"].str.lower()==temp_batteryID]
    chemistry = temp_df["Chemistry"].iloc[0]
    return chemistry 


def get_folders(input_folderpath): 
    folders = os.listdir(input_folderpath)
    folders = [os.path.join(input_folderpath, folder) for folder in folders]
    folderpaths = [folder for folder in folders if os.path.isdir(folder)]
    return folderpaths, folders 


def label_builder(folderpaths, folders): 
    #iteratively add chemistry: 
    for i, folderpath in enumerate(folderpaths): 
        files = os.listdir(folderpath)
        files = [file for file in files if file.endswith(".png")]
        folder = folders[i]
        file_label_dict = {}
        for file in files: 
            file_path = os.path.join(folder, file)
            tempK, batteryID, Crate, direction = extract_labels(file)
            if batteryID == None: 
                print('No ID match for: ', file, " !")
            chemistry = extract_chemistry(meta_df, batteryID)
            file_label_dict[file_path] = {"chemistry": chemistry,
                                        "temperature": tempK,
                                        "Crate protocol": Crate,
                                        "direction": direction}
        temp_df = pd.DataFrame.from_dict(file_label_dict, orient="index")
        outpath = os.path.join(folderpath,f'{folder}_labels.csv' )
        temp_df.to_csv(outpath, index=False)
        print(f'output {folder} label csv file to: {outpath}')


if __name__ == "__main__": 
    desired_folderpath = r'C:\Users\MJone\Documents\SIADS699\processed_images\model_prep'
    meta_df = load_meta_data()
    folderpaths, folders = get_folders(desired_folderpath)
    label_builder(folderpaths, folders)

