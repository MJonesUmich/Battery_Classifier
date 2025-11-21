import os
import re

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d


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


def get_chem_dict():
    lfp_dict = {
        "SOC": [
            0,
            5.3,
            10.5,
            15.8,
            21.1,
            26.3,
            31.6,
            36.8,
            42.1,
            47.4,
            52.6,
            57.9,
            63.2,
            68.4,
            73.7,
            78.9,
            84.2,
            89.5,
            94.7,
            100.0,
        ],
        "OCV": [
            2.0,
            2.974,
            3.127,
            3.159,
            3.192,
            3.216,
            3.232,
            3.243,
            3.248,
            3.252,
            3.255,
            3.258,
            3.261,
            3.267,
            3.279,
            3.294,
            3.301,
            3.305,
            3.309,
            3.600,
        ],
    }

    nmc_dict = {
        "SOC": [
            0,
            5.3,
            10.5,
            15.8,
            21.1,
            26.3,
            31.6,
            36.8,
            42.1,
            47.4,
            52.6,
            57.9,
            63.2,
            68.4,
            73.7,
            78.9,
            84.2,
            89.5,
            94.7,
            100.0,
        ],
        "OCV": [
            2.5,
            3.385,
            3.432,
            3.478,
            3.525,
            3.555,
            3.572,
            3.587,
            3.603,
            3.622,
            3.648,
            3.693,
            3.744,
            3.794,
            3.847,
            3.902,
            3.961,
            4.022,
            4.088,
            4.200,
        ],
    }

    nca_dict = {
        "SOC": [
            0,
            5.3,
            10.5,
            15.8,
            21.1,
            26.3,
            31.6,
            36.8,
            42.1,
            47.4,
            52.6,
            57.9,
            63.2,
            68.4,
            73.7,
            78.9,
            84.2,
            89.5,
            94.7,
            100.0,
        ],
        "OCV": [
            2.600,
            3.308,
            3.357,
            3.427,
            3.493,
            3.536,
            3.567,
            3.596,
            3.625,
            3.659,
            3.701,
            3.761,
            3.810,
            3.855,
            3.897,
            3.942,
            3.998,
            4.058,
            4.107,
            4.200,
        ],
    }

    lco_dict = {
        "SOC": [
            0,
            5.3,
            10.5,
            15.8,
            21.1,
            26.3,
            31.6,
            36.8,
            42.1,
            47.4,
            52.6,
            57.9,
            63.2,
            68.4,
            73.7,
            78.9,
            84.2,
            89.5,
            94.7,
            100.0,
        ],
        "OCV": [
            2.7,
            3.17,
            3.31,
            3.44,
            3.53,
            3.61,
            3.68,
            3.74,
            3.79,
            3.84,
            3.88,
            3.93,
            3.97,
            4.01,
            4.05,
            4.09,
            4.12,
            4.15,
            4.18,
            4.20,
        ],
    }

    chemistry_dict = {
        "LCO": lco_dict,
        "NCA": nca_dict,
        "NMC": nmc_dict,
        "LFP": lfp_dict,
    }

    return chemistry_dict


def soc_correction(input_df):
    soc = input_df["SOC"].to_numpy()
    soc_min, soc_max = np.min(soc), np.max(soc)

    # Adjust only if necessary
    if soc_min < 0:
        soc = soc - soc_min

    if np.max(soc) > 100:
        soc = soc / np.max(soc) * 100

    input_df["SOC"] = soc
    return input_df


def impute_soc(input_chemistry, input_df, input_capacity):
    # Unpack chemistry-specific OCV
    chemistry_dict = get_chem_dict()
    ocv_ref = chemistry_dict[input_chemistry]
    soc_ref = np.array(ocv_ref["SOC"])
    v_ref = np.array(ocv_ref["OCV"])

    # Interpolation intial SOC from initial voltage
    f_soc = interp1d(v_ref, soc_ref, kind="linear", fill_value="extrapolate")
    V_init = input_df["Voltage(V)"].iloc[0]
    SOC_init = float(f_soc(V_init))

    time = input_df["Test_Time(s)"].values
    current = input_df["Current(A)"].values

    # Now Coloumb count to get remaining SOC
    dt = np.diff(time, prepend=time[0])
    delta_soc = np.cumsum(current * dt / 3600) / input_capacity * 100
    SOC = SOC_init + delta_soc
    result_df = input_df.copy()
    result_df["SOC"] = SOC

    # Apply Correction Factor
    output_df = soc_correction(result_df.copy())
    return output_df

keep_isu_cells = [
    "G5C1",
    "G5C2",
    "G5C3",
    "G5C4",
    "G9C1",
    "G9C2",
    "G9C3",
    "G9C4",
    "G11C1",
    "G11C2",
    "G11C3",
    "G11C4",
    "G16C1",
    "G16C2",
    "G16C3",
    "G16C4",
    "G20C1",
    "G20C2",
    "G20C3",
    "G20C4",
    "G21C1",
    "G21C2",
    "G21C3",
    "G21C4",
    "G22C1",
    "G22C2",
    "G22C3",
    "G22C4",
    "G23C1",
    "G23C2",
    "G23C3",
    "G23C4",
    "G24C1",
    "G24C2",
    "G24C3",
    "G24C4",
    "G28C1",
    "G28C2",
    "G28C3",
    "G28C4",
    "G35C1",
    "G35C2",
    "G35C3",
    "G35C4",
    "G36C1",
    "G36C2",
    "G36C3",
    "G36C4",
    "G37C1",
    "G37C2",
    "G37C3",
    "G37C4",
    "G38C1",
    "G38C2",
    "G38C3",
    "G38C4",
    "G39C1",
    "G39C2",
    "G39C3",
    "G39C4",
    "G45C1",
    "G45C2",
    "G45C3",
    "G45C4",
    "G53C1",
    "G53C2",
    "G53C3",
    "G53C4",
    "G60C1",
    "G60C2",
    "G60C3",
    "G60C4",
    "G63C1",
    "G63C2",
    "G63C3",
    "G63C4",
    "G64C1",
    "G64C2",
    "G64C3",
    "G64C4",
]