import h5py
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd 

import cx_cell_parser as cx


def gen_bat_df(input_path, input_file):
    """Generate battery dataframe from MIT .mat files with reduced precision"""
    input_filepath = os.path.join(input_path, input_file)
    bol_cap = 1.1
    try:
        with h5py.File(input_filepath, 'r') as f:
            batch = f.get('batch')
            if batch is None or 'cycles' not in batch:
                print(f"Warning: Required data not found in {input_file}")
                return
            
            cell_qty = batch['summary'].shape[0]
            for cell in range(cell_qty):
                all_cycles_data = []
                cycles_ref = batch['cycles'][cell, 0]
                cycles = f[cycles_ref]
                
                if 'I' not in cycles:
                    continue
                    
                n_cycles = cycles['I'].shape[0]
                for cycle in range(n_cycles):
                    # Read cycle data
                    def get_cycle_data(key):
                        try:
                            ref = cycles[key][cycle, 0]
                            return np.array(f[ref]).squeeze()
                        except Exception:
                            return np.array([])

                    current = get_cycle_data('I')
                    voltage = get_cycle_data('V')
                    temperature = get_cycle_data('T')
                    time = get_cycle_data('t')
                    
                    # Skip empty cycles
                    if len(current) == 0 or len(voltage) == 0:
                        continue
                    
                    # Create cycle DataFrame with reduced precision
                    cycle_data = pd.DataFrame({
                        'Cell_ID': np.full(len(current), cell, dtype=np.int16),
                        'Cycle_Count': np.full(len(current), cycle, dtype=np.int16),
                        'Current(A)': current.astype(np.float32),
                        'Voltage(V)': voltage.astype(np.float32),
                        'Test_Time(s)': time.astype(np.float32),
                        'direction': np.where(current > 0, 'charge', 'discharge')
                    })

                    cycle_data["Test_Time(s)"] = (cycle_data["Test_Time(s)"]*60).astype(np.float32)
                    cycle_data["Delta_Time(s)"] = cycle_data["Test_Time(s)"].diff().fillna(0).astype(np.float32)
                    cycle_data["Delta_Ah"] = (cycle_data["Current(A)"] * cycle_data["Delta_Time(s)"] / 3600).astype(np.float32)
                    cycle_data["C_rate"] = (cycle_data["Current(A)"] / bol_cap).astype(np.float16)  # BOL 1.1Ah capacity

                    all_cycles_data.append(cycle_data)
                
                if all_cycles_data:
                    # Combine all cycles and save with reduced memory footprint
                    cell_df = pd.concat(all_cycles_data, ignore_index=True)
                    cell_df["Ah_throughput"] = np.abs(cell_df["Delta_Ah"]).cumsum().astype(np.float32)
                    cell_df["EFC"] = (cell_df["Ah_throughput"] / bol_cap).astype(np.float32)  # BOL 1.1Ah capacity
                    
                    output_df = cell_df.copy()
                    output_df = output_df[["Current(A)", "Voltage(V)", "Test_Time(s)", 
                                           "Cycle_Count", "Delta_Time(s)", "Delta_Ah", 
                                           "Ah_throughput", "EFC", "C_rate", "direction",
                                           ]]
                    
                    # Export with reduced size
                    out_name = input_file.replace('.mat', f'_cell_{cell}_processed.csv')
                    output_df.to_csv(
                        os.path.join(input_path, out_name),
                        index=False,
                        float_format='%.4f'  # Limit decimal places in output
                    )
                    
                    print(f"Processed cell {cell}: {len(all_cycles_data)} cycles, {len(cell_df)} total points")
                    
    except Exception as e:
        print(f"Error processing {input_file}: {str(e)}")
        return None


def generate_figures(output_image_folder, df_charge, df_discharge, 
                     cell_C_rate_charge, cell_C_rate_discharge, 
                     cell_temperature, battery_id_tag, cycle, ):
    
    #Set plot directory
    if not os.path.exists(output_image_folder):
        os.makedirs(output_image_folder)

    #generate plot, clipped last datum in case current reset to rest
    plt.figure(figsize=(10, 6))
    plt.plot(df_charge['Test_Time(s)'], df_charge['Voltage(V)'], color='blue')
    plt.xlabel('Charge Time (s)')
    plt.ylabel('Voltage (V)', color='blue')
    plt.title(f'Cycle {cycle} Charge Profile')
    save_string = f"{output_image_folder}\Cycle_{cycle}_charge_Crate_{cell_C_rate_charge}_tempK_{cell_temperature}_batteryID_{battery_id_tag}.png"
    plt.savefig(save_string)
    plt.close()

    #plot current on secondary axis
    plt.figure(figsize=(10, 6))
    plt.plot(df_discharge['Test_Time(s)'], df_discharge['Voltage(V)'], 'r-') #remove last few points to avoid voltage recovery
    plt.ylabel('Voltage (V)', color='red')
    plt.title(f'Cycle {cycle} Discharge Profile')
    save_string = f"{output_image_folder}\Cycle_{cycle}_discharge_Crate_{cell_C_rate_discharge}_tempK_{cell_temperature}_batteryID_{battery_id_tag}.png"
    plt.savefig(save_string)
    plt.close()

def clip_cv(input_df, direction):
    """Vectorized clipping: find first index where delta-current > 0.5 AND voltage crosses threshold.
    If no trigger found, return the full input_df unchanged.
    """
    if input_df.shape[0] < 2:
        return input_df.copy()

    curr = input_df["Current(A)"].to_numpy(dtype=float)
    volt = input_df["Voltage(V)"].to_numpy(dtype=float)

    # delta_current at position i corresponds to abs(curr[i] - curr[i-1]); set first element to 0
    delta_curr = np.abs(np.concatenate(([0.0], np.diff(curr))))

    # previous-voltage array (volt[i-1]) with first element = volt[0]
    volt_prev = np.concatenate(([volt[0]], volt[:-1]))

    if direction == 'discharge':
        # trigger when delta_current > 0.5 and either current or previous voltage <= 2.5
        mask = (delta_curr > 0.5) & ((volt <= 2.5) | (volt_prev <= 2.5))
    elif direction == 'charge':
        # trigger when delta_current > 0.5 and either current or previous voltage >= 3.5
        mask = (delta_curr > 0.5) & ((volt >= 3.5) | (volt_prev >= 3.5))
    else:
        return input_df.copy()

    idxs = np.where(mask)[0]
    if idxs.size == 0:
        # no trigger found -> return full dataframe
        return input_df.copy()

    stop_index = int(idxs[0])
    return input_df.iloc[: stop_index + 1].reset_index(drop=True)


def parse_meta_data(meta_df=None, battery_id_tag=None):    

    #connect to cell infos: 
    if meta_df is None or battery_id_tag is None:
        cell_C_rate_charge = 'fast'
        cell_C_rate_discharge = 4
        cell_temperature = 303
        battery_id_tag = 'Test_Battery_01'
    else:     
        cell_df = meta_df[meta_df["Battery_ID"].str.lower() == str.lower(battery_id_tag)]
        cell_C_rate_charge = cell_df["C_rate_Charge"].values[0]
        cell_C_rate_discharge = cell_df["C_rate_Discharge"].values[0]
        cell_temperature = cell_df["Temperature (K)"].values[0]

    return cell_C_rate_charge, cell_C_rate_discharge, battery_id_tag, cell_temperature


def scrub_data(input_df,cell_C_rate_charge, cell_C_rate_discharge, 
               cell_temperature, battery_id_tag, output_folder_paths): 
    
    output_data_folder, output_image_folder = output_folder_paths
    if not os.path.exists(output_data_folder):
        os.makedirs(output_data_folder)
    
    if not os.path.exists(output_image_folder):
        os.makedirs(output_image_folder)

    agg_df = pd.DataFrame()
    #Examine cycles, apply voltage clipping under CV conditions 
    #Remove cycles with no charge/discharge data  
    cycles = input_df.Cycle_Count.unique()
    for cycle in cycles: 
        cycle_df = input_df[input_df['Cycle_Count'] == cycle]
        df_charge_cycle = cycle_df[cycle_df['Current(A)'] > 0]
        df_discharge_cycle = cycle_df[cycle_df['Current(A)'] < 0]
        if len(df_charge_cycle) == 0 or len(df_discharge_cycle) == 0: 
            print('zero length')
            continue
        else: 
            #now clip if cv hold: 
            df_charge_cycle = df_charge_cycle.reset_index(drop=True)
            df_discharge_cycle = df_discharge_cycle.reset_index(drop=True)
            df_charge_cycle = clip_cv(df_charge_cycle, 'charge')
            df_discharge_cycle = clip_cv(df_discharge_cycle, 'discharge')
            #NEED TO DEBUG WHY THIS IS COMING UP BLANK>.....
            if df_charge_cycle is not None and df_discharge_cycle is not None:
                if len(df_charge_cycle) >0 and len(df_discharge_cycle) >0: 
                    
                    #generate plots 
                    generate_figures(output_image_folder, df_charge_cycle, df_discharge_cycle, cell_C_rate_charge, cell_C_rate_discharge, 
                        cell_temperature, battery_id_tag, cycle)

                    #append data to dataframe 
                    cycle_df = pd.concat([df_charge_cycle, df_discharge_cycle], ignore_index=True)
                    agg_df = pd.concat([agg_df, cycle_df], ignore_index=True)
    
    if len(agg_df) >0:
        agg_df.to_csv(f"{output_data_folder}\{battery_id_tag}_aggregated_dataset.csv", index=False)



def mit_parser(input_files, input_folder, meta_df, output_folder_paths):

    for file in input_files: 
        print(file)
        gen_bat_df(input_folder, file)

    cell_datafiles = os.listdir(input_folder)
    cell_datafiles = [file for file in cell_datafiles if file.endswith('.csv')]
    for cell_file in cell_datafiles: 
        cell_filepath = os.path.join(input_folder, cell_file)
        cell_df = pd.read_csv(cell_filepath)
        battery_id_tag = cell_file[0:-14]
        print(battery_id_tag)
        cell_C_rate_charge, cell_C_rate_discharge, battery_id_tag, cell_temperature = parse_meta_data(meta_df=meta_df,
                                                                                                    battery_id_tag=battery_id_tag,
                                                                                                    )
        
        scrub_data(cell_df,cell_C_rate_charge, cell_C_rate_discharge, 
                cell_temperature, battery_id_tag, output_folder_paths)

if __name__ == "__main__":
    #Load battery mapper data: 
    meta_df = cx.load_meta_properties()
    input_folder = r'D:\Battery_Classification\MIT_Data'
    input_files = os.listdir(input_folder)
    input_files = [file for file in input_files if file.endswith('.mat')]

    output_image_folder = r'processed_images\LFP'
    output_data_folder = r'processed_datasets\LFP'
    output_folder_paths = (output_data_folder, output_image_folder)
    mit_parser(input_files, input_folder, meta_df, output_folder_paths)