import pandas as pd
import numpy as np 
import os 
import json 
import matplotlib.pyplot as plt 
import traceback

import CX_parser as cx
 

def load_cycling_json(file, path):
    """Load and convert cycling JSON file for a given cell."""
    with open(f'{path}/{file}', 'r') as f:
        data = json.loads(json.load(f))  # double decode!

    # Convert start/stop times to numpy datetime for consistency
    for i, start_time in enumerate(data['start_stop_time']['start']):
        if start_time != '[]':
            data['start_stop_time']['start'][i] = np.datetime64(start_time)
            data['start_stop_time']['stop'][i] = np.datetime64(data['start_stop_time']['stop'][i])
        else:
            data['start_stop_time']['start'][i] = []
            data['start_stop_time']['stop'][i] = []
    return data


def extract_cycle_data(cycling_dict):
    """
    Extract time, voltage, and capacity for each charge/discharge cycle.
    Returns two DataFrames: charge_df and discharge_df.
    """
    charge_cycles = []
    discharge_cycles = []
    num_cycles = len(cycling_dict['QV_charge']['t'])
    print(cycling_dict['QV_charge'].keys())
    print(cycling_dict['QV_discharge'].keys())

    # Each entry in QV_charge and QV_discharge corresponds to one cycle
    for i in range(num_cycles):
        # --- Extract charge data ---
        I_charge = cycling_dict['QV_charge']['I'][i]
        V_charge = cycling_dict['QV_charge']['V'][i]
        t_charge = cycling_dict['QV_charge']['t'][i]
        Q_charge = cycling_dict['QV_charge']['Q'][i]

        if len(Q_charge) > 0:
            df_c = pd.DataFrame({
                'Cycle_Count': i + 1,
                'Time': pd.to_datetime(t_charge),
                "Current(A)": I_charge,
                'Voltage(V)': V_charge,
                'Capacity': Q_charge,
                'direction': 'Charge'
            })
            # Normalize time to start from zero
            t0 = df_c['Time'].iloc[0]
            df_c['Time(s)'] = (df_c['Time'] - t0).dt.total_seconds()

            # Clean and ensure monotonic direction
            df_c["Voltage"] = df_c["Voltage(V)"].round(3)
            df_c = df_c.drop_duplicates(subset=["Voltage(V)"])
            df_c = df_c.sort_values(by='Time(s)').reset_index(drop=True)

            # Sanity check — ensure charge increases in voltage
            if df_c["Voltage"].iloc[0] < df_c["Voltage"].iloc[-1]:
                charge_cycles.append(df_c)
            else:
                # Misclassified (actually discharge)
                df_c["direction"] = "Discharge"
                discharge_cycles.append(df_c)

        # --- Extract discharge data ---
        I_discharge = cycling_dict['QV_discharge']['I'][i]
        V_discharge = cycling_dict['QV_discharge']['V'][i]
        t_discharge = cycling_dict['QV_discharge']['t'][i]
        Q_discharge = cycling_dict['QV_discharge']['Q'][i]


        if len(Q_discharge) > 0:
            df_d = pd.DataFrame({
                'Cycle_Count': i + 1,
                'Time': pd.to_datetime(t_discharge),
                "Current(A)": I_discharge,
                'Voltage(V)': V_discharge,
                'Capacity': Q_discharge,
                'direction': 'Discharge'
            })
            # Normalize time to start from zero
            t0 = df_d['Time'].iloc[0]
            df_d['Time(s)'] = (df_d['Time'] - t0).dt.total_seconds()

            # Clean and ensure monotonic direction
            df_d["Voltage"] = df_d["Voltage(V)"].round(3)
            df_d = df_d.drop_duplicates(subset=["Voltage(V)"])
            df_d = df_d.sort_values(by='Time(s)').reset_index(drop=True)

            # Sanity check — ensure discharge decreases in voltage
            if df_d["Voltage"].iloc[0] > df_d["Voltage"].iloc[-1]:
                discharge_cycles.append(df_d)
            else:
                # Misclassified (actually charge)
                df_d["Type"] = "Charge"
                charge_cycles.append(df_d)


    # Combine all cycles into single DataFrames
    charge_df = pd.concat(charge_cycles, ignore_index=True)
    discharge_df = pd.concat(discharge_cycles, ignore_index=True)

    return charge_df, discharge_df


def clip_data(input_df, direction): 
    voltage = input_df["Voltage(V)"].values
    if direction == "charge":
        limit = voltage.max() - 0.005
        clip_idx = np.argmax(voltage >= limit)
    elif direction == "discharge":
        limit = voltage.min() + 0.01
        clip_idx = np.argmax(voltage <= limit)
    else:
        return input_df.copy()

    return input_df.iloc[:clip_idx + 1].copy()


def monotonicity_check(input_df, direction): 
    if len(input_df) > 5: 

        if direction == 'charge':
            valid_profile =  input_df['Voltage(V)'].is_monotonic_increasing
            valid_profile = True

        elif direction == 'discharge':
            valid_profile =  input_df['Voltage(V)'].is_monotonic_decreasing
            valid_profile = True
    else: 
        valid_profile = False

    return valid_profile


def generate_figures(output_image_folder, charge_cycle_df, discharge_cycle_df,C_rate_charge,
                    C_rate_discharge, temperature, battery_ID, cycle):

    #Set plot directory
    if not os.path.exists(output_image_folder):
        os.makedirs(output_image_folder)

    #generate plot, clipped last datum in case current reset to rest
    fig = plt.figure(figsize=(10, 6))
    charge_cycle_df = charge_cycle_df.copy()
    discharge_cycle_df = discharge_cycle_df.copy()

    charge_cycle_df["step_time(s)"] = charge_cycle_df["Test_Time(s)"] - charge_cycle_df["Test_Time(s)"].iloc[0] 
    plt.plot(charge_cycle_df['step_time(s)'], charge_cycle_df['Voltage(V)'], color='blue')
    plt.xlabel('Charge Time (s)')
    plt.ylabel('Voltage (V)', color='blue')
    plt.title(f'Cycle {cycle} Charge Profile')
    save_string = f"{output_image_folder}\Cycle_{cycle}_charge_Crate_{C_rate_charge}_tempK_{temperature}_batteryID_{battery_ID}.png"
    fig.savefig(save_string)
    plt.close(fig)
    print(f'successfully exported {save_string} to {output_image_folder}')

    #plot current on secondary axis
    fig = plt.figure(figsize=(10, 6))
    discharge_cycle_df["step_time(s)"] = discharge_cycle_df["Test_Time(s)"] - discharge_cycle_df["Test_Time(s)"].iloc[0] 
    plt.plot(discharge_cycle_df['step_time(s)'], discharge_cycle_df['Voltage(V)'], 'r-') #remove last few points to avoid voltage recovery
    plt.ylabel('Voltage (V)', color='red')
    plt.title(f'Cycle {cycle} Discharge Profile')
    save_string = f"{output_image_folder}\Cycle_{cycle}_discharge_Crate_{C_rate_discharge}_tempK_{temperature}_batteryID_{battery_ID}.png"
    fig.savefig(save_string)
    plt.close(fig)
    print(f'successfully exported {save_string} to {output_image_folder}')


def scrub_cycles(input_df, cell_C_rate_charge, cell_C_rate_discharge, cell_id, cell_temperature): 
    unique_cycles = input_df["Cycle_Count"].unique()
    output_df = pd.DataFrame()
    for cycle in unique_cycles: 
        include_data = True
        cycle_df = input_df[input_df["Cycle_Count"] == cycle]
        df_charge = cycle_df[cycle_df["direction"] == "charge"]
        df_discharge = cycle_df[cycle_df["direction"] == "discharge"]

        approx_time_charge = 3600 * 0.9 / cell_C_rate_charge  / 2
        approx_time_discharge = 3600 * 0.9 / cell_C_rate_discharge  / 2
        if df_charge["Test_Time(s)"].max() - df_charge["Test_Time(s)"].min() < approx_time_charge:
            include_data = False
        if df_discharge["Test_Time(s)"].max() - df_discharge["Test_Time(s)"].min() < approx_time_discharge:
            include_data = False

        if include_data:
            try:
                #clip data to avoid rest periods and constant-voltage conditions
                charge_clip_df = clip_data(df_charge, direction="charge")
                discharge_clip_df = clip_data(df_discharge, direction="discharge")

                #now generate plot data
                generate_figures(output_image_folder, df_charge, df_discharge,
                                cell_C_rate_charge, cell_C_rate_discharge, cell_temperature, 
                                cell_id, cycle)
                output_df = pd.concat([output_df, charge_clip_df, discharge_clip_df], ignore_index=True)
            except Exception as e:
                continue

    output_df = output_df.sort_values(by=["Cycle_Count", "direction", "Test_Time(s)"], 
                                    ascending=[True, True, True]).reset_index(drop=True)
    return output_df
 

def isu_parser(input_data_folder, output_folder_paths):
    output_data_folder, output_image_folder = output_folder_paths
    meta_df = cx.load_meta_properties()
    files = os.listdir(input_data_folder)
    for counter, file in enumerate(files): 
        out_df = pd.DataFrame()
        print('beginning file: ', file)
        print(np.round(counter / len(files) * 100,2), "% Complete")
        error_log_df = pd.DataFrame()
        apple = 'yes'
        if apple == 'yes':
        #try: 
            #connect to cell infos: 
            cell_id = file.split('.')[0]
            cell_df = meta_df[meta_df["Battery_ID"].str.lower() == str.lower(cell_id)]
            
            #Extract Cycles, but only for datasets where they comprise >90% DOD profile
            if float(cell_df["DoD"]) > 0.9:
                cycling_dict = load_cycling_json(file, input_data_folder)
                df_charge, df_discharge = extract_cycle_data(cycling_dict)
                df_charge["direction"] = "charge"
                df_discharge["direction"] = "discharge"

                #Split between charge and discharge 
                valid_charge = monotonicity_check(df_charge, direction="charge")
                valid_discharge = monotonicity_check(df_discharge, direction = "discharge")
                print(valid_charge,valid_discharge)

                #Load meta_data properties
                if valid_charge and valid_discharge: 
                    cell_initial_capacity = cell_df["Initial_Capacity_Ah"].values[0]
                    cell_C_rate_charge = cell_df["C_rate_Charge"].values[0]
                    cell_C_rate_discharge = cell_df["C_rate_Discharge"].values[0]
                    cell_temperature = cell_df["Temperature (K)"].values[0]

                    #iterate through to create the plots and get the aggregated data:
                    #Generate Plot Data for Training: 
                    print('--------------')
                    print(df_charge.columns)
                    combined = pd.concat([df_charge, df_discharge])
                    combined = combined.sort_values(by=["Cycle_Count", "direction", "Time(s)"], 
                                                    ascending=[True, True, True]).reset_index(drop=True)

                    combined["Test_Time(s)"] = (combined['Time'] - combined['Time'].iloc[0]).dt.total_seconds()
                    combined['Delta_Time(s)'] = combined['Test_Time(s)'].diff().fillna(0)
                    combined['Delta_Ah'] = combined['Capacity'].diff().fillna(0)
                    combined['Ah_throughput'] = combined['Delta_Ah'].cumsum()
                    combined['EFC'] = combined['Ah_throughput'] / cell_initial_capacity
                    combined['C_rate'] = combined["Current(A)"] / cell_initial_capacity

                    print('----------------')
                    print('Sorted successfully')
                    print(combined.head())

                    #Add data to the training timeseries datafiles: 
                    print('these are the columns for the combined_df: ', combined.columns)
                    combined = combined[["Current(A)","Voltage(V)","Test_Time(s)","Cycle_Count",
                                    "Delta_Time(s)","Delta_Ah","Ah_throughput","EFC",
                                    "C_rate","direction",
                                    ]]

                    print('------------------')
                    print('This is combined')
                    print(combined.head())

                    #next need to downsample the cycles to just those that cover >0.9 DoD
                    agg_df = scrub_cycles(combined, cell_C_rate_charge, cell_C_rate_discharge, cell_id, cell_temperature)

                    #Export aggregated datafile
                    if not os.path.exists(output_data_folder):
                        os.makedirs(output_data_folder)
                    output_file_name = f'{cell_id}_aggregated_data.csv'
                    agg_df.to_csv(os.path.join(output_data_folder,output_file_name), index=False)

                # except Exception as e:
                #     tb_str = traceback.format_exc()
                #     error_entry = {
                #         "file": file,
                #         "error_type": type(e).__name__,
                #         "error_message": str(e),
                #         "traceback": tb_str,
                #         "timestamp": pd.Timestamp.now()
                #     }        

                #     # Append to the error DataFrame
                #     error_log_df = pd.concat([error_log_df, pd.DataFrame([error_entry])], ignore_index=True)

                #     # Save after each failure so nothing is lost if the script crashes
                #     error_log_df.to_csv(error_log_path, index=False)

                #     print(f"Error processing {file}: {e}")



if __name__ == "__main__":
    #Load battery mapper data & charging datafiles: 
    output_image_folder = r'processed_images\NMC'
    output_data_folder = r'processed_datasets\NMC'
    input_data_folder = r'C:\Users\MJone\Downloads\ISU_Data\Cycling_json\Cycling_json\all'
    output_folder_paths = (output_data_folder, output_image_folder)

    #Error handler setup
    error_log_path = "error_log.csv"
    error_log_df = pd.DataFrame(columns=["file", "error_type", "error_message", "traceback", "timestamp"])


    isu_parser(input_data_folder, output_folder_paths)