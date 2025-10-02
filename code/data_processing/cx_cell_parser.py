import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime


def load_meta_properties():
    #Finish function to associate file name with cell capacity, c-rate, and temperatures
    df = pd.read_excel(r'C:\Users\MJone\Downloads\battery_data_mapper.xlsx', sheet_name='General_Infos')
    return df


def extract_date(file_name):
    # Extract MM, DD, YR from the file name
    parts = file_name.split('_')
    month, day, year = int(parts[-4]), int(parts[-3]), int(parts[-2])
    return datetime(year, month, day).date()


def sort_files(file_names):

    file_dates = []

    # Extract dates and sort files
    for file_name in file_names:
        file_date = extract_date(file_name)
        file_dates.append(file_date)

    # Sort files by their corresponding dates
    sorted_files = [file for _, file in sorted(zip(file_dates, file_names))]

    return sorted_files, file_dates


def load_file(file_path):
        
    # Read Excel (choose the sheet including Current/Voltage)
    xls = pd.ExcelFile(file_path)
    chosen = None
    for s in xls.sheet_names:
        cols = set(pd.read_excel(file_path, sheet_name=s, nrows=1).columns.astype(str))
        if {"Current(A)", "Voltage(V)"} <= cols:
            chosen = s
            break
    if chosen is None:
        chosen = xls.sheet_names[0]

    df = pd.read_excel(file_path, sheet_name=chosen)
    df.columns = [str(c).strip() for c in df.columns]

    # Get the desired columns out: 
    desired_cols = ["Current(A)", "Voltage(V)", "Test_Time(s)", "Date_Time"]
    df = df[desired_cols].dropna().reset_index(drop=True)
    return df 


def get_indices(df):
    
        I = df["Current(A)"]

        # Set threshold to avoid 0 current jitter
        thr = max(0.05 * np.nanmedian(np.abs(I[I != 0])), 1e-4)
        sign3 = np.zeros_like(I, dtype=int)
        sign3[I >  thr] =  1
        sign3[I < -thr] = -1

        charge_indices, discharge_indices = [], []
        prev = 0
        for i, s in enumerate(sign3):
            if s == 0: 
                continue
            if prev == 0:
                prev = s
                continue
            if s != prev:
                if s ==  1:
                    charge_indices.append(i)
                elif s == -1:
                    discharge_indices.append(i)
                prev = s

        # print(len(charge_indices), len(discharge_indices))
        # print(charge_indices[0], discharge_indices[0])
        # print(charge_indices[1], discharge_indices[1])
        # print(charge_indices[2], discharge_indices[2])
        # print(charge_indices[3], discharge_indices[3])

        complexity, expected_order = check_indices(charge_indices, discharge_indices)
        if complexity == "High":
            #Skip this file 
            raise ValueError("Indices do not alternate correctly.")
        else:
            if expected_order[0] == 'discharge' and len(discharge_indices) > len(charge_indices):
                discharge_indices = discharge_indices[:-1]
            elif expected_order[0] == 'charge' and len(charge_indices) > len(discharge_indices):
                charge_indices = charge_indices[:-1]

        assert len(charge_indices) == len(discharge_indices)
        return charge_indices, discharge_indices


def check_indices(charge_indices, discharge_indices):

    # Determine which starts first
    if charge_indices[0] < discharge_indices[0]:
        expected_order = ['charge', 'discharge']
        combined = [(idx, 'charge') for idx in charge_indices] + [(idx, 'discharge') for idx in discharge_indices]
    else:
        expected_order = ['discharge', 'charge']
        combined = [(idx, 'discharge') for idx in discharge_indices] + [(idx, 'charge') for idx in charge_indices]

    # Sort combined list by index
    combined.sort(key=lambda x: x[0])  # Sort by index

    # Check for alternating order
    for i, (_, label) in enumerate(combined):
        if label != expected_order[i % 2]:
            print(f"Error: Indices do not alternate correctly at position {i} ({combined[i - 1]} followed by {combined[i]})")
            complexity = "High"
            return complexity, expected_order

    complexity = "Low"
    print("Indices alternate correctly.")
    return complexity, expected_order


def scrub_and_tag(df, charge_indices, discharge_indices):
    # Downsample to just between charge cycles
    df = df.iloc[charge_indices[0]:discharge_indices[-1] + 1].reset_index(drop=True)

    # Adjust charge_indices and discharge_indices to match the new DataFrame
    adjusted_charge_indices = [i - charge_indices[0] for i in charge_indices]
    adjusted_discharge_indices = [i - charge_indices[0] for i in discharge_indices]

    # Create a new column for tagging
    df['Cycle_Count'] = None

    # Assign "Charge {i}" tags
    for i, (start, end) in enumerate(zip(adjusted_charge_indices, adjusted_charge_indices[1:] + [len(df)]), start=1):
        df.loc[start:end - 1, 'Cycle_Count'] = f"{i}"

    #Coloumb count Ah throughput for each cycle
    df['Delta_Time(s)'] = df['Test_Time(s)'].diff().fillna(0)
    df['Delta_Ah'] = np.abs(df['Current(A)']) * df['Delta_Time(s)'] / 3600
    df['Ah_throughput'] = df['Delta_Ah'].cumsum()

    #now calculate Equivalent Full Cycles (EFC) & Capacity Fade
    df['EFC'] = df['Ah_throughput'] / cell_initial_capacity
    return df


def update_df(df, agg_df): 
    if len(agg_df) == 0:
        return df
    else: 
        max_cycle = agg_df['Cycle_Count'].max()
        df['Cycle_Count'] = df['Cycle_Count'].astype(int) + int(max_cycle)
        df['Ah_throughput'] = df['Ah_throughput'] + agg_df['Ah_throughput'].max()
        df['EFC'] = df['EFC'] + agg_df['EFC'].max()
        return df


def parse_file(file_path, cell_initial_capacity, cell_C_rate):
    df = load_file(file_path)
    charge_indices, discharge_indices = get_indices(df)
    df = scrub_and_tag(df, charge_indices, discharge_indices)
    df["C_rate"] = cell_C_rate
    return df


def generate_figures(df, vmax, vmin, c_rate, temperature, battery_ID, tolerance=0.01,one_fig_only=False):
    unique_cycles = df['Cycle_Count'].unique()
    for i, cycle in enumerate(unique_cycles):
        cycle_df = df[df['Cycle_Count'] == cycle]
        
        #find where voltage first hits vmax and vmin, and where first discharge occurs
        vmax_idx = cycle_df[cycle_df['Voltage(V)'] >= vmax - tolerance].index[0]
        vmin_idx = cycle_df[cycle_df['Voltage(V)'] <= vmin + tolerance].index[0]
        disch_start = cycle_df[cycle_df['Current(A)'] < 0 - tolerance].index[0] 
        
        #clip data to initial until Vmax, then from discharge start to Vmin
        charge_cycle_df = cycle_df.loc[0:vmax_idx]
        discharge_cycle_df = cycle_df.loc[disch_start:vmin_idx]
        charge_cycle_df["Charge_Time(s)"] = charge_cycle_df["Test_Time(s)"] - charge_cycle_df["Test_Time(s)"].iloc[0]
        discharge_cycle_df["Discharge_Time(s)"] = discharge_cycle_df["Test_Time(s)"] - discharge_cycle_df["Test_Time(s)"].iloc[0]
        
        #generate plot, clipped last datum in case current reset to rest
        plt.figure(figsize=(10, 6))
        plt.plot(charge_cycle_df['Charge_Time(s)'], charge_cycle_df['Voltage(V)'], color='blue')
        plt.xlabel('Charge Time (s)')
        plt.ylabel('Voltage (V)', color='blue')
        plt.title(f'Cycle {cycle} Charge Profile')
        save_string = f"Cycle_{i+1}_charge_Crate_{c_rate}_tempK_{temperature}_batteryID_{battery_ID}.png"
        plt.savefig(save_string)

        #plot current on secondary axis
        plt.figure(figsize=(10, 6))
        plt.plot(discharge_cycle_df['Discharge_Time(s)'], discharge_cycle_df['Voltage(V)'], 'r-') #remove last few points to avoid voltage recovery
        plt.ylabel('Voltage (V)', color='red')
        plt.title(f'Cycle {cycle} Discharge Profile')
        save_string = f"Cycle_{i+1}_discharge_Crate_{c_rate}_tempK_{temperature}_batteryID_{battery_ID}.png"
        plt.savefig(save_string)

        #Exit function after 1st run if one_fig_only is True
        if one_fig_only:
            break


#Example run through on 1 file
meta_df = load_meta_properties()

folder_path = r'C:\Users\MJone\Downloads\CX2_8\cx2_8'
file_names = [file for file in os.listdir(folder_path)]
sorted_files, file_dates = sort_files(file_names)
sorted_files = sorted_files[::-1]
file_dates = file_dates[::-1]

print(sorted_files)
error_dict = {}

agg_df = pd.DataFrame()

cell_id = folder_path.split('\\')[-1]
cell_df = meta_df[meta_df["Battery_ID"].str.lower() == cell_id]

cell_initial_capacity = cell_df["Initial_Capacity_Ah"].values[0]
cell_C_rate = cell_df["C_rate"].values[0]
cell_temperature = cell_df["Temperature (K)"].values[0]
cell_vmax = cell_df["Max_Voltage"].values[0]
cell_vmin = cell_df["Min_Voltage"].values[0]

for i_count, file_name in enumerate(file_names): 
    try: 
        file_path   = os.path.join(folder_path, file_name)
        df = parse_file(file_path, cell_initial_capacity, cell_C_rate)
        update_df(df, agg_df)
        agg_df = pd.concat([agg_df, df], ignore_index=True)

    #except add failed files to dictionary with error message
    except Exception as e:
        error_dict[file_name] = str(e)
    
    print(f'{i_count/len(file_names)}% Complete')

    #send to df and output: 
error_df = pd.DataFrame(list(error_dict.items()), columns=['File_Name', 'Error_Message'])
error_df.to_csv('error_log.csv', index=False)
agg_df.to_csv('aggregated_data.csv', index=False)
generate_figures(df, cell_vmax, cell_vmin, cell_C_rate, cell_temperature, battery_ID=cell_id, one_fig_only=True)

