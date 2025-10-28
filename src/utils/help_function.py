import os
import re

import pandas as pd


def load_meta_properties(sheet_name="Sheet1"):
    # Finish function to associate file name with cell capacity, c-rate, and temperatures
    # Load battery mapper data:
    sheet_id = "19L7_7HpOUagvRAh6GNOrhcjQpbEu97kx"
    url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=xlsx"
    meta_df = pd.read_excel(url, sheet_name=sheet_name)
    return meta_df


def check_file_string(file_name):
    name, extension = os.path.splitext(file_name)
    if re.search(r"[a-zA-Z]+$", name):
        return "bad"
    else:
        return "good"
