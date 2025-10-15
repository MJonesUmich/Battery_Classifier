import pandas as pd
import numpy as np 
import os 
import json 
import matplotlib.pyplot as plt 
import traceback


#Load json cycling data
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



#Extract to dataframe 
def extract_cycle_data(cycling_dict):
    """
    Extract time, voltage, and capacity for each charge/discharge cycle.
    Returns two DataFrames: charge_df and discharge_df.
    """
    charge_cycles = []
    discharge_cycles = []
    num_cycles = len(cycling_dict['QV_charge']['t'])
    # Each entry in QV_charge and QV_discharge corresponds to one cycle
    for i in range(num_cycles):
        # --- Extract charge data ---
        t_charge = cycling_dict['QV_charge']['t'][i]
        Q_charge = cycling_dict['QV_charge']['Q'][i]
        V_charge = cycling_dict['QV_charge']['V'][i]

        if len(Q_charge) > 0:
            df_c = pd.DataFrame({
                'Cycle_Count': i + 1,
                'Time': pd.to_datetime(t_charge),
                'Voltage(V)': V_charge,
                'Capacity': Q_charge,
                'Type': 'Charge'
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
                df_c["Type"] = "Discharge"
                discharge_cycles.append(df_c)

        # --- Extract discharge data ---
        t_discharge = cycling_dict['QV_discharge']['t'][i]
        Q_discharge = cycling_dict['QV_discharge']['Q'][i]
        V_discharge = cycling_dict['QV_discharge']['V'][i]

        if len(Q_discharge) > 0:
            df_d = pd.DataFrame({
                'Cycle_Count': i + 1,
                'Time': pd.to_datetime(t_discharge),
                'Voltage(V)': V_discharge,
                'Capacity': Q_discharge,
                'Type': 'Discharge'
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


def generate_figures(df_charge, df_discharge, c_rate_charge, c_rate_discharge, temperature, battery_ID, one_fig_only=False):
    out_folder = r'processed_images\NMC'
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)

    for df in [df_charge, df_discharge]: 
        unique_cycles = df['Cycle_Count'].unique()
        for i, cycle in enumerate(unique_cycles):
            cycle_df = df[df['Cycle_Count'] == cycle]
            cycle_df = cycle_df.sort_values(by='Time(s)').reset_index(drop=True)

            if cycle_df["Type"].iloc[0] == "Charge": 
                label = "charge"
                color = 'blue'
                approx_time = 3600 * 0.9 / c_rate_charge
                approx_time / 2 
                if cycle_df["Time(s)"].max() - cycle_df["Time(s)"].min() < approx_time:
                    include_data = False
                else: 
                    include_data = True 
            elif cycle_df["Type"].iloc[0] == "Discharge":
                label = "discharge"
                color = 'red'
                approx_time = 3600 * 0.9 / c_rate_charge
                approx_time / 2 
                if cycle_df["Time(s)"].max() - cycle_df["Time(s)"].min() < approx_time:
                    include_data = False
                else: 
                    include_data = True 
            
            delta_voltage = cycle_df["Voltage(V)"].max() - cycle_df["Voltage(V)"].min()
            #print(delta_voltage)
            if delta_voltage > 0.005 and include_data == True: 
                clipped_df = clip_data(cycle_df, direction=label)

                #generate plot, clipped last datum in case current reset to rest
                plt.figure(figsize=(10, 6))
                plt.plot(clipped_df['Time(s)'], clipped_df['Voltage(V)'], color=color)
                plt.xlabel(f'{label.capitalize()} Time (s)')
                plt.ylabel('Voltage (V)', color=color)
                plt.title(f'Cycle {cycle} {label.capitalize()} Profile')
                save_string = f"{out_folder}\Cycle_{i+1}_{label}_CrateCharge_{c_rate_charge}_CrateDisharge_{c_rate_discharge}_tempK_{temperature}_batteryID_{battery_ID}.png"
                plt.savefig(save_string)

                #Exit function after 1st run if one_fig_only is True
                if one_fig_only:
                    break


#Load battery mapper data: 
sheet_id = "19L7_7HpOUagvRAh6GNOrhcjQpbEu97kx"
url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=xlsx"
meta_df = pd.read_excel(url, sheet_name='Sheet1')

#Load charge data 
path = r'C:\Users\MJone\Downloads\ISU_Data\Cycling_json\Cycling_json\all'
files = os.listdir(path)

#Setup error handler: 
error_log_path = "error_log.csv"
error_log_df = pd.DataFrame(columns=["file", "error_type", "error_message", "traceback", "timestamp"])

for counter, file in enumerate(files): 
    print('beginning file: ', file)
    print(np.round(counter / len(files) * 100,2), "% Complete")
    try: 
        #connect to cell infos: 
        cell_id = file.split('.')[0]
        cell_df = meta_df[meta_df["Battery_ID"].str.lower() == str.lower(cell_id)]
        
        #Generate figures only for the near full DOD profile
        if float(cell_df["DoD"]) > 0.9:
            print(file)
            #file = 'G1C1.json'
            cycling_dict = load_cycling_json(file, path)
            df_charge, df_discharge = extract_cycle_data(cycling_dict)
            valid_charge = monotonicity_check(df_charge, direction="charge")
            valid_discharge = monotonicity_check(df_discharge, direction = "discharge")
            print(valid_charge,valid_discharge)
            if valid_charge and valid_discharge: 
                cell_initial_capacity = cell_df["Initial_Capacity_Ah"].values[0]
                cell_C_rate = cell_df["C_rate"].values[0]
                cell_C_rate_charge = cell_df["C_rate_Charge"].values[0]
                cell_C_rate_discharge = cell_df["C_rate_Discharge"].values[0]

                cell_temperature = cell_df["Temperature (K)"].values[0]

                generate_figures(df_charge, df_discharge, cell_C_rate_charge, cell_C_rate_discharge, cell_temperature,
                                cell_id, one_fig_only=True)
            
    except Exception as e:
        tb_str = traceback.format_exc()
        error_entry = {
            "file": file,
            "error_type": type(e).__name__,
            "error_message": str(e),
            "traceback": tb_str,
            "timestamp": pd.Timestamp.now()
        }        

        # Append to the error DataFrame
        error_log_df = pd.concat([error_log_df, pd.DataFrame([error_entry])], ignore_index=True)

        # Save after each failure so nothing is lost if the script crashes
        error_log_df.to_csv(error_log_path, index=False)

        print(f"⚠️ Error processing {file}: {e}")