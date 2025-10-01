import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def load_meta_properties():
    #Finish function to associate file name with cell capacity, c-rate, and temperatures
    df = pd.read_excel(r'C:\Users\MJone\Downloads\battery_data_mapper.xlsx', sheet_name='General_Infos')
    return df


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

        assert len(charge_indices) == len(discharge_indices)
        return charge_indices, discharge_indices


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


def parse_file(file_path, cell_initial_capacity, cell_C_rate):
    df = load_file(file_path)
    charge_indices, discharge_indices = get_indices(df)
    df = scrub_and_tag(df, charge_indices, discharge_indices)
    df["C_rate"] = cell_C_rate
    return df


def generate_figures(df, vmax, vmin, c_rate, temperature, tolerance=0.01):
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
        save_string = f"Cycle_{i+1}_charge_Crate_{c_rate}_tempK_{temperature}.png"
        plt.savefig(save_string)

        #plot current on secondary axis
        plt.figure(figsize=(10, 6))
        plt.plot(discharge_cycle_df['Discharge_Time(s)'], discharge_cycle_df['Voltage(V)'], 'r-') #remove last few points to avoid voltage recovery
        plt.ylabel('Voltage (V)', color='red')
        plt.title(f'Cycle {cycle} Discharge Profile')
        save_string = f"Cycle_{i+1}_discharge_Crate_{c_rate}_tempK_{temperature}.png"
        plt.savefig(save_string)


#Example run through on 1 file
meta_df = load_meta_properties()

folder_path = r'C:\Users\MJone\Downloads\CX2_8\cx2_8'
file_names = [file for file in os.listdir(folder_path)]

file_name = 'CX2_8_6_30_11.xlsx'  # Just take the first file for now
file_path   = os.path.join(folder_path, file_name)

cell_id = folder_path.split('\\')[-1]
cell_df = meta_df[meta_df["Battery_ID"].str.lower() == cell_id]

cell_initial_capacity = cell_df["Initial_Capacity_Ah"].values[0]
cell_C_rate = cell_df["C_rate"].values[0]
cell_temperature = cell_df["Temperature (K)"].values[0]
cell_vmax = cell_df["Max_Voltage"].values[0]
cell_vmin = cell_df["Min_Voltage"].values[0]

df = parse_file(file_path, cell_initial_capacity, cell_C_rate)
generate_figures(df, cell_vmax, cell_vmin, cell_C_rate, cell_temperature)
print(df.Cycle_Count.max())
