import pandas as pd 
import os

import cx_cell_parser as cx


meta_df = cx.load_meta_properties()

folder_path = r'C:\Users\MJone\Documents\GitHub\Battery_Classifier\code\data_processing'
INR_files = ['11_5_2015_low current OCV test_SP20-1.xlsx', 
             '11_16_2015_low current OCV test_SP20-3.xlsx',
             ]

file_paths = [os.join(folder_path, file) for file in INR_files]
print(file_paths)