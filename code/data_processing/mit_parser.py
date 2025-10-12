import h5py
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd 

#Load battery mapper data: 
sheet_id = "19L7_7HpOUagvRAh6GNOrhcjQpbEu97kx"
url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=xlsx"
meta_df = pd.read_excel(url, sheet_name='Sheet1')     


folder = r'C:\Users\MJone\Downloads\archive (2)'
file = '2017-05-12_batchdata_updated_struct_errorcorrect.mat'
file_path = os.path.join(folder, file)

def get_largest_index(input_index):
    best_indice = None 
    best_delta = 0 
    for indice in input_index:
        temp_delta = indice[1] - indice[0]
        if temp_delta > best_delta: 
            best_indice = indice
            best_delta = temp_delta 
    return best_indice
        

def clip_cv(input_df): 
    stop_index = None
    indice = 0
    while stop_index is None: 
        if input_df["Voltage"].iloc[indice] <= 2.1: 
            stop_index = indice
        indice += 1
    return indice


def split_direction(temp_df):
    df_charge = None
    df_discharge = None 
    charge_start = []
    charge_stop = []
    discharge_start = []
    discharge_stop = []
    
    #Iterate to pull the counters
    for item in range(len(temp_df)): 
        if item < len(temp_df) - 1: 
            temp_charge = temp_df["Current"].iloc[item]
            next_charge =  temp_df["Current"].iloc[item+1]
            temp_voltage = temp_df["Voltage"].iloc[item]
            if temp_charge >1 and next_charge <=1: 
                charge_stop.append(item)
            if temp_charge <=1 and next_charge >1: 
                charge_start.append(item)
            if temp_charge <0 and next_charge >=0 and temp_voltage<2.1:
                discharge_stop.append(item)
            if temp_charge >=0 and next_charge <0: 
                discharge_start.append(item)

    if len(charge_start) == len(charge_stop) and len(discharge_start) == len(discharge_stop):
        charge_indices = list(zip(charge_start,charge_stop))
        discharge_indices = list(zip(discharge_start,discharge_stop))
        #print(discharge_indices)
        if len(charge_indices) > 1: 
            charge_index = get_largest_index(charge_indices)
        else: 
            charge_index = [charge_indices[0][0], charge_indices[0][1]]
        if len(discharge_indices) >1: 
            discharge_index = get_largest_index(discharge_indices)
        else: 
            discharge_index = [discharge_indices[0][0], discharge_indices[0][1]]
        
        df_charge = temp_df[charge_index[0]:charge_index[1]+1]
        df_discharge = temp_df[discharge_index[0]:discharge_index[1]+1]
        discharge_indice = clip_cv(df_discharge)
        df_discharge = df_discharge[0:discharge_indice+1]
    return df_charge, df_discharge 


def parse_meta_data(meta_df, battery_id_tag):    

    #connect to cell infos: 
    cell_df = meta_df[meta_df["Battery_ID"].str.lower() == str.lower(battery_id_tag)]
    cell_C_rate_charge = cell_df["C_rate_Charge"].values[0]
    cell_C_rate_discharge = cell_df["C_rate_Discharge"].values[0]
    cell_temperature = cell_df["Temperature (K)"].values[0]

    return cell_C_rate_charge, cell_C_rate_discharge, battery_id_tag, cell_temperature


def generate_figures(df_charge, df_discharge, cell_C_rate_charge, cell_C_rate_discharge, 
                     cell_temperature, battery_id_tag):
    
    #generate plot, clipped last datum in case current reset to rest
    plt.figure(figsize=(10, 6))
    plt.plot(df_charge['Time'], df_charge['Voltage'], color='blue')
    plt.xlabel('Charge Time (s)')
    plt.ylabel('Voltage (V)', color='blue')
    plt.title(f'Cycle {1} Charge Profile')
    save_string = f"Cycle_{1}_charge_Crate_{cell_C_rate_charge}_tempK_{cell_temperature}_batteryID_{battery_id_tag}.png"
    plt.savefig(save_string)

    #plot current on secondary axis
    plt.figure(figsize=(10, 6))
    plt.plot(df_discharge['Time'], df_discharge['Voltage'], 'r-') #remove last few points to avoid voltage recovery
    plt.ylabel('Voltage (V)', color='red')
    plt.title(f'Cycle {1} Discharge Profile')
    save_string = f"Cycle_{1}_discharge_Crate_{cell_C_rate_discharge}_tempK_{cell_temperature}_batteryID_{battery_id_tag}.png"
    plt.savefig(save_string)


one_fig_only = True

with h5py.File(file_path, 'r') as f:
    batch = f['batch']
    #print('batch keys: ', batch.keys())
    num_cells = batch["cycles"].shape[0]

    #cycles_ref = batch['cycles'][0, 0]
    #print("num cells ", num_cells)
    for cell_id in range(num_cells):
        battery_id_tag = file[0:-4] + '_' + str(cell_id)
        #print('battery_id_tag: ', battery_id_tag)
        cycles_ref = batch['cycles'][cell_id, 0]
        # Flatten the list of cycle references (each one is a cycle)
        if isinstance(cycles_ref, h5py.Reference):
            cycle_refs = [cycles_ref]
        else:
            cycle_refs = cycles_ref.flatten()

        #print(f"Number of cycles: {len(cycle_refs)}")

        for i, ref in enumerate(cycle_refs[:2]):  # limit to first 10 for clarity
            cycle_group = f[ref]
            #print(cycle_group.keys())
            if 'V' not in cycle_group or 't' not in cycle_group:
                continue  # skip incomplete cycles

            V_refs = cycle_group['V']
            t_refs = cycle_group['t']
            i_irefs = cycle_group['I']

            # Some cycles may have only discharge or only charge
            for j in range(V_refs.shape[0]):
                if j == 1: 
                    V_data = np.array(f[V_refs[j][0]]).squeeze()
                    i_data = np.array(f[i_irefs[j][0]]).squeeze()
                    t_data = np.array(f[t_refs[j][0]]).squeeze()
                    if (len(V_data)==0 or len(i_data)==0 or len(t_data)==0): 
                        break
                    else: 
                        temp_df = pd.DataFrame({'Voltage': V_data, 'Time': t_data, 'Current': i_data})
                        temp_df['Time'] = temp_df['Time'] * 60
                        print(type(temp_df), len(temp_df))
                        df_charge, df_discharge = split_direction(temp_df)
                        if df_charge is not None and df_discharge is not None: 
                            cell_C_rate_charge, cell_C_rate_discharge, battery_id_tag, cell_temperature = parse_meta_data(meta_df, battery_id_tag)
                            print(f"generating figures for {battery_id_tag}")
                            generate_figures(df_charge, df_discharge, cell_C_rate_charge, cell_C_rate_discharge, cell_temperature,
                                            battery_id_tag)
                            # fig, ax1 = plt.subplots(figsize=(8, 4))
                            
                            #Exit function after 1st run if one_fig_only is True
                            if one_fig_only:
                                break