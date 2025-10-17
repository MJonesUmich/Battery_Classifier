import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import loadmat

def load_meta_properties():
    #Finish function to associate file name with cell capacity, c-rate, and temperatures
    df = pd.read_excel(r'C:\Users\ShuS\Documents\github\Battery_Classifier\data\battery_data_mapper.xlsx', sheet_name='General_Infos')
    return df




def extract_voltage_limits_from_mat_full(file_path, cell_number=1, percentile=99):
    """
    Extract voltage limits (vmax and vmin) from Oxford_Battery_Degradation_Dataset_1.mat format
    
    Parameters:
    - file_path: path to .mat file
    - cell_number: which cell to analyze (1-8)
    - percentile: percentile to use for determining voltage limits
    
    Returns:
    - vmax: maximum charge voltage
    - vmin: minimum discharge voltage
    """
    mat_data = loadmat(file_path)
    
    # Get the cell data
    cell_key = f'Cell{cell_number}'
    if cell_key not in mat_data:
        raise ValueError(f"Cell {cell_number} not found in mat file")
    
    cell_data = mat_data[cell_key]
    dtype_names = cell_data.dtype.names
    
    if dtype_names is None:
        raise ValueError(f"No cycle data found for {cell_key}")
    
    cycle_names = [name for name in dtype_names if name.startswith('cyc')]
    
    all_charge_voltages = []
    all_discharge_voltages = []
    
    # Collect voltage data from all cycles
    for cycle_name in cycle_names[:5]:  # Use first 5 cycles to determine limits
        try:
            cycle_data = cell_data[cycle_name][0, 0]
            
            # C1 charge voltages
            c1ch_data = cycle_data['C1ch'][0, 0]
            v_ch = c1ch_data['v'][0, 0].flatten()
            all_charge_voltages.extend(v_ch)
            
            # C1 discharge voltages
            c1dc_data = cycle_data['C1dc'][0, 0]
            v_dc = c1dc_data['v'][0, 0].flatten()
            all_discharge_voltages.extend(v_dc)
            
        except Exception:
            continue
    
    if len(all_charge_voltages) == 0 or len(all_discharge_voltages) == 0:
        raise ValueError(f"Could not extract voltage data from {cell_key}")
    
    # Get voltage limits using percentile
    vmax = np.percentile(all_charge_voltages, percentile)
    vmin = np.percentile(all_discharge_voltages, 100 - percentile)
    
    return vmax, vmin


def load_from_mat_full(file_path, cell_number=1):
    """
    Load data from Oxford_Battery_Degradation_Dataset_1.mat format
    Structure: Cell[1-8] -> cyc#### -> C1ch/C1dc/OCVch/OCVdc -> t, v, q, T
    
    Parameters:
    - file_path: path to the .mat file
    - cell_number: which cell to extract (1-8)
    
    Returns combined DataFrame for all cycles of the specified cell
    """
    mat_data = loadmat(file_path)
    
    # Get the cell data
    cell_key = f'Cell{cell_number}'
    if cell_key not in mat_data:
        raise ValueError(f"Cell {cell_number} not found in mat file. Available keys: {list(mat_data.keys())}")
    
    cell_data = mat_data[cell_key]
    
    # Get all cycle names (cyc0100, cyc0200, etc.)
    dtype_names = cell_data.dtype.names
    if dtype_names is None:
        raise ValueError(f"No cycle data found for {cell_key}")
    
    cycle_names = [name for name in dtype_names if name.startswith('cyc')]
    cycle_names.sort()  # Sort to get chronological order
    
    all_dfs = []
    
    for cycle_name in cycle_names:
        cycle_num = int(cycle_name.replace('cyc', ''))
        cycle_data = cell_data[cycle_name][0, 0]
        
        # Extract C1 charge and discharge (1C rate tests)
        # For current, we need to infer from the test type since it's not in the data
        try:
            # C1 charge
            c1ch_data = cycle_data['C1ch'][0, 0]
            t_ch = c1ch_data['t'][0, 0].flatten()
            v_ch = c1ch_data['v'][0, 0].flatten()
            # For 1C charge, current = 740mA = 0.74A (based on readme)
            i_ch = np.ones_like(t_ch) * 0.74
            
            # C1 discharge  
            c1dc_data = cycle_data['C1dc'][0, 0]
            t_dc = c1dc_data['t'][0, 0].flatten()
            v_dc = c1dc_data['v'][0, 0].flatten()
            # For 1C discharge, current = -740mA = -0.74A
            i_dc = np.ones_like(t_dc) * (-0.74)
            
            # Adjust discharge time to continue from charge
            if len(t_ch) > 0:
                t_dc = t_dc + t_ch[-1]
            
            # Combine charge and discharge
            time = np.concatenate([t_ch, t_dc])
            voltage = np.concatenate([v_ch, v_dc])
            current = np.concatenate([i_ch, i_dc])
            
            # Create DataFrame for this cycle
            cycle_df = pd.DataFrame({
                'Test_Time(s)': time,
                'Voltage(V)': voltage,
                'Current(A)': current,
                'Characterization_Cycle': cycle_num
            })
            
            all_dfs.append(cycle_df)
            
        except Exception as e:
            print(f"Warning: Could not process {cycle_name}: {str(e)}")
            continue
    
    if len(all_dfs) == 0:
        raise ValueError(f"No valid cycle data found for {cell_key}")
    
    # Combine all cycles
    df = pd.concat(all_dfs, ignore_index=True)
    
    return df.dropna().reset_index(drop=True)


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


def scrub_and_tag(df, charge_indices, discharge_indices, cell_initial_capacity, vmax=None, vmin=None, tolerance=0.01):
    # Downsample to just between charge cycles
    df = df.iloc[charge_indices[0]:discharge_indices[-1] + 1].reset_index(drop=True)

    # Adjust charge_indices and discharge_indices to match the new DataFrame
    adjusted_charge_indices = [i - charge_indices[0] for i in charge_indices]
    adjusted_discharge_indices = [i - charge_indices[0] for i in discharge_indices]

    # Create a new column for tagging
    df['Cycle_Count'] = None

    # Process each cycle to filter out constant voltage holds
    filtered_cycles = []
    
    for i, (charge_start, charge_end) in enumerate(zip(adjusted_charge_indices, adjusted_charge_indices[1:] + [len(df)]), start=1):
        # Get the discharge start for this cycle
        if i <= len(adjusted_discharge_indices):
            discharge_start = adjusted_discharge_indices[i-1]
        else:
            discharge_start = len(df)
        
        # Extract cycle data
        cycle_data = df.iloc[charge_start:discharge_start].copy()
        cycle_data['Cycle_Count'] = i
        
        # Filter out constant voltage holds if vmax and vmin are provided
        if vmax is not None and vmin is not None:
            cycle_data = filter_voltage_range(cycle_data, vmax, vmin, tolerance)
        
        if len(cycle_data) > 0:
            filtered_cycles.append(cycle_data)
    
    # Combine all filtered cycles
    if filtered_cycles:
        df = pd.concat(filtered_cycles, ignore_index=True)
    else:
        # Fallback to original method if no cycles were filtered
        for i, (start, end) in enumerate(zip(adjusted_charge_indices, adjusted_charge_indices[1:] + [len(df)]), start=1):
            df.loc[start:end - 1, 'Cycle_Count'] = i

    #Coloumb count Ah throughput for each cycle
    df['Delta_Time(s)'] = df['Test_Time(s)'].diff().fillna(0)
    df['Delta_Ah'] = np.abs(df['Current(A)']) * df['Delta_Time(s)'] / 3600
    df['Ah_throughput'] = df['Delta_Ah'].cumsum()

    #now calculate Equivalent Full Cycles (EFC) & Capacity Fade
    df['EFC'] = df['Ah_throughput'] / cell_initial_capacity
    return df


def filter_voltage_range(cycle_data, vmax, vmin, tolerance=0.01):
    """
    Filter cycle data to include only the voltage range between vmin and vmax,
    excluding constant voltage holds (CV phase).
    
    Strategy:
    - For charge: Keep data from start until voltage reaches vmax, but exclude CV hold
    - For discharge: Keep data from discharge start until voltage reaches vmin
    - CV hold detection: voltage stays near vmax/vmin while current decreases
    """
    if len(cycle_data) == 0:
        return cycle_data
    
    voltage = cycle_data['Voltage(V)'].values
    current = cycle_data['Current(A)'].values
    
    # Find discharge start (first significant negative current)
    discharge_start = None
    current_threshold = 0.05 * np.abs(current[np.abs(current) > 1e-6]).max() if np.any(np.abs(current) > 1e-6) else 0.01
    
    for i, c in enumerate(current):
        if c < -current_threshold:
            discharge_start = i
            break
    
    if discharge_start is None:
        # No discharge found, only charge data
        discharge_start = len(voltage)
    
    # Process charge phase (before discharge)
    charge_end_idx = None
    if discharge_start > 0:
        charge_voltage = voltage[:discharge_start]
        charge_current = current[:discharge_start]
        
        # Find where voltage first reaches vmax
        vmax_first = None
        for i, v in enumerate(charge_voltage):
            if v >= vmax - tolerance:
                vmax_first = i
                break
        
        if vmax_first is not None:
            # Check if there's a CV hold after reaching vmax
            # CV hold: voltage stays near vmax, current decreases significantly
            cv_hold_start = None
            for i in range(vmax_first, len(charge_voltage)):
                if i + 5 < len(charge_voltage):  # Need at least 5 points to detect CV
                    # Check if voltage is stable near vmax
                    voltage_stable = np.all(np.abs(charge_voltage[i:i+5] - vmax) < tolerance)
                    # Check if current is decreasing (CV characteristic)
                    current_decreasing = charge_current[i] < 0.5 * charge_current[vmax_first] if charge_current[vmax_first] > 0 else False
                    
                    if voltage_stable and current_decreasing:
                        cv_hold_start = i
                        break
            
            # Set charge end: before CV hold starts, or at vmax if no clear CV hold
            charge_end_idx = cv_hold_start if cv_hold_start is not None else vmax_first + 1
        else:
            # vmax not reached, use all charge data
            charge_end_idx = discharge_start
    else:
        charge_end_idx = 0
    
    # Process discharge phase
    discharge_end_idx = None
    if discharge_start < len(voltage):
        discharge_voltage = voltage[discharge_start:]
        
        # Find where voltage first reaches vmin
        vmin_first = None
        for i, v in enumerate(discharge_voltage):
            if v <= vmin + tolerance:
                vmin_first = discharge_start + i
                break
        
        if vmin_first is not None:
            discharge_end_idx = vmin_first + 1
        else:
            # vmin not reached, use all discharge data
            discharge_end_idx = len(voltage)
    else:
        discharge_end_idx = len(voltage)
    
    # Combine charge and discharge portions
    filtered_indices = []
    
    # Add charge portion (up to CV hold or vmax)
    if charge_end_idx > 0:
        filtered_indices.extend(range(0, charge_end_idx))
    
    # Add discharge portion (from discharge start to vmin)
    if discharge_start < discharge_end_idx:
        filtered_indices.extend(range(discharge_start, discharge_end_idx))
    
    if len(filtered_indices) > 0:
        return cycle_data.iloc[filtered_indices].reset_index(drop=True)
    else:
        # Fallback: return original data if filtering fails
        return cycle_data


def update_df(df, agg_df): 
    if len(agg_df) == 0:
        return df
    else: 
        max_cycle = agg_df['Cycle_Count'].max()
        df['Cycle_Count'] = df['Cycle_Count'].astype(int) + int(max_cycle)
        df['Ah_throughput'] = df['Ah_throughput'] + agg_df['Ah_throughput'].max()
        df['EFC'] = df['EFC'] + agg_df['EFC'].max()
        return df


def parse_file(file_path, cell_initial_capacity, cell_C_rate, vmax=None, vmin=None, cell_number=1):
    """
    Parse Oxford .mat file and process battery data
    
    Parameters:
    - file_path: path to .mat file
    - cell_initial_capacity: initial capacity in Ah
    - cell_C_rate: C-rate value
    - vmax, vmin: voltage limits for filtering
    - cell_number: which cell to extract (1-8)
    """
    df = load_from_mat_full(file_path, cell_number)

    charge_indices, discharge_indices = get_indices(df)
    df = scrub_and_tag(df, charge_indices, discharge_indices, cell_initial_capacity, vmax, vmin)
    df["C_rate"] = cell_C_rate
    return df


def parse_mat_file(file_path, cell_initial_capacity, cell_C_rate, vmax=None, vmin=None, cell_number=1):
    """
    Wrapper function specifically for parsing Oxford .mat files
    
    Parameters:
    - file_path: path to .mat file
    - cell_initial_capacity: initial capacity in Ah
    - cell_C_rate: C-rate value
    - vmax, vmin: voltage limits for filtering
    - cell_number: which cell to extract (1-8)
    """
    return parse_file(file_path, cell_initial_capacity, cell_C_rate, vmax, vmin, cell_number)


def generate_figures(df, vmax, vmin, c_rate, temperature, battery_ID, tolerance=0.01,one_fig_only=False):
    unique_cycles = df['Cycle_Count'].unique()
    for i, cycle in enumerate(unique_cycles):
        cycle_df = df[df['Cycle_Count'] == cycle].reset_index(drop=True)
        
        #find where voltage first hits vmax and vmin, and where first discharge occurs
        try:
            vmax_idx = cycle_df[cycle_df['Voltage(V)'] >= vmax - tolerance].index[0]
        except IndexError:
            print(f"Warning: No voltage >= {vmax - tolerance} found in cycle {cycle}, using maximum voltage point...")
            vmax_idx = cycle_df['Voltage(V)'].idxmax()
            
        try:
            vmin_idx = cycle_df[cycle_df['Voltage(V)'] <= vmin + tolerance].index[0]
        except IndexError:
            print(f"Warning: No voltage <= {vmin + tolerance} found in cycle {cycle}, using minimum voltage point...")
            vmin_idx = cycle_df['Voltage(V)'].idxmin()
            
        try:
            disch_start = cycle_df[cycle_df['Current(A)'] < 0 - tolerance].index[0]
        except IndexError:
            print(f"Warning: No discharge current found in cycle {cycle}, skipping...")
            continue 
        
        # Debug: print indices to understand the issue
        print(f"Cycle {cycle}: vmax_idx={vmax_idx}, disch_start={disch_start}, vmin_idx={vmin_idx}, cycle_df length={len(cycle_df)}")
        
        #clip data to initial until Vmax, then from discharge start to Vmin
        charge_cycle_df = cycle_df.loc[0:vmax_idx].copy()
        
        # Find vmin_idx after discharge starts (not before)
        discharge_portion = cycle_df.loc[disch_start:]
        try:
            vmin_idx_adjusted = discharge_portion[discharge_portion['Voltage(V)'] <= vmin + tolerance].index[0]
        except IndexError:
            print(f"Warning: No vmin reached after discharge in cycle {cycle}, using end of discharge...")
            vmin_idx_adjusted = discharge_portion.index[-1]
        
        discharge_cycle_df = cycle_df.loc[disch_start:vmin_idx_adjusted].copy()
        
        # Check if we have valid data
        if len(charge_cycle_df) == 0:
            print(f"Warning: No charge data found in cycle {cycle}, skipping charge plot...")
        else:
            charge_cycle_df["Charge_Time(s)"] = charge_cycle_df["Test_Time(s)"] - charge_cycle_df["Test_Time(s)"].iloc[0]
            
            #generate plot, clipped last datum in case current reset to rest
            plt.figure(figsize=(10, 6))
            plt.plot(charge_cycle_df['Charge_Time(s)'], charge_cycle_df['Voltage(V)'], color='blue')
            plt.xlabel('Charge Time (s)')
            plt.ylabel('Voltage (V)', color='blue')
            plt.title(f'Cycle {cycle} Charge Profile')
            save_string = f"Cycle_{i+1}_charge_Crate_{c_rate}_tempK_{temperature}_batteryID_{battery_ID}.png"
            plt.savefig(save_string)
            plt.close()
        
        if len(discharge_cycle_df) == 0:
            print(f"Warning: No discharge data found in cycle {cycle}, skipping discharge plot...")
        else:
            discharge_cycle_df["Discharge_Time(s)"] = discharge_cycle_df["Test_Time(s)"] - discharge_cycle_df["Test_Time(s)"].iloc[0]
            
            #plot current on secondary axis
            plt.figure(figsize=(10, 6))
            plt.plot(discharge_cycle_df['Discharge_Time(s)'], discharge_cycle_df['Voltage(V)'], 'r-') #remove last few points to avoid voltage recovery
            plt.ylabel('Voltage (V)', color='red')
            plt.title(f'Cycle {cycle} Discharge Profile')
            save_string = f"Cycle_{i+1}_discharge_Crate_{c_rate}_tempK_{temperature}_batteryID_{battery_ID}.png"
            plt.savefig(save_string)
            plt.close()

        #Exit function after 1st run if one_fig_only is True
        if one_fig_only:
            break



if __name__ == "__main__": 
    # Process Oxford Battery Degradation Dataset
    meta_df = load_meta_properties()

    # Process the full Oxford dataset
    file_path = r'C:\Users\zhzha\Documents\github\Battery_Classifier\data\raw\Oxford\Oxford_Battery_Degradation_Dataset_1.mat'
    cells_to_process = range(1, 9)  # Full file has cells 1-8

    print(f"Processing {file_path}")
    print(f"Dataset: Oxford Battery Degradation Dataset")
    
    error_dict = {}
    all_cells_data = []

    # Get Oxford battery metadata
    # For Oxford cells, use a generic Battery_ID or specific IDs if available
    cell_id_base = 'Oxford_Cell'
    
    # Oxford battery specifications from readme
    # Kokam CO LTD, SLPB533459H4, 740mAh = 0.74Ah
    cell_initial_capacity = 0.74  # Ah
    cell_C_rate = 1.0  # 1C rate (740mA)
    cell_temperature = 273.15 + 40  # 40°C in Kelvin
    
    # Extract voltage limits from the data
    print("\nExtracting voltage limits from data...")
    try:
        # Extract from first cell to determine voltage limits
        cell_vmax, cell_vmin = extract_voltage_limits_from_mat_full(file_path, cell_number=1, percentile=99)
        
        print(f"[SUCCESS] Extracted voltage limits from data:")
        print(f"  - cell_vmax = {cell_vmax:.4f} V")
        print(f"  - cell_vmin = {cell_vmin:.4f} V")
    except Exception as e:
        # Fallback to default values if extraction fails
        cell_vmax = 4.2
        cell_vmin = 2.9
        print(f"[WARNING] Could not extract voltage limits: {str(e)}")
        print(f"  Using default values: vmax={cell_vmax}V, vmin={cell_vmin}V")
    
    # Try to get from metadata if available
    try:
        cell_df = meta_df[meta_df["Battery_ID"].str.contains('Oxford', case=False, na=False)]
        if len(cell_df) > 0:
            cell_initial_capacity = cell_df["Initial_Capacity_Ah"].values[0]
            cell_C_rate = cell_df["C_rate"].values[0]
            cell_temperature = cell_df["Temperature (K)"].values[0]
            # Don't override extracted voltage limits with metadata
            print("Using capacity/C-rate/temperature from battery_data_mapper.xlsx")
            print(f"Keeping extracted voltage limits: vmax={cell_vmax:.4f}V, vmin={cell_vmin:.4f}V")
    except Exception as e:
        print(f"Using default Oxford specifications from readme: {str(e)}")

    # Process each cell
    for cell_num in cells_to_process:
        try:
            print(f"\nProcessing Cell {cell_num}...")
            cell_id = f"{cell_id_base}{cell_num}"
            
            # Parse the mat file
            df = parse_mat_file(
                file_path, 
                cell_initial_capacity, 
                cell_C_rate, 
                cell_vmax, 
                cell_vmin,
                cell_number=cell_num
            )
            
            # Add cell identifier
            df['Battery_ID'] = cell_id
            df['Temperature(K)'] = cell_temperature
            
            all_cells_data.append(df)
            
            print(f"Cell {cell_num}: Processed {len(df)} data points")
            
            # Generate figures for the first cell only
            if cell_num == cells_to_process[0]:
                try:
                    generate_figures(
                        df, 
                        cell_vmax, 
                        cell_vmin, 
                        cell_C_rate, 
                        cell_temperature, 
                        battery_ID=cell_id, 
                        one_fig_only=True
                    )
                except Exception as e:
                    print(f"Warning: Could not generate figures: {str(e)}")
            
        except Exception as e:
            error_dict[f'Cell{cell_num}'] = str(e)
            print(f"Error processing Cell {cell_num}: {str(e)}")

    # Combine all cells data
    if len(all_cells_data) > 0:
        agg_df = pd.concat(all_cells_data, ignore_index=True)
        
        # Save to CSV
        output_filename = 'aggregated_oxford_data.csv'
        agg_df.to_csv(output_filename, index=False)
        print(f"\nSuccessfully saved {len(agg_df)} rows to {output_filename}")
        print(f"Columns: {list(agg_df.columns)}")
        print(f"\nData summary:")
        print(f"  - Total cells processed: {len(all_cells_data)}")
        print(f"  - Total cycles: {agg_df['Cycle_Count'].nunique()}")
        print(f"  - Total data points: {len(agg_df)}")
    else:
        print("No data was successfully processed")

    # Save error log
    if error_dict:
        error_df = pd.DataFrame(list(error_dict.items()), columns=['Cell', 'Error_Message'])
        error_df.to_csv('oxford_error_log.csv', index=False)
        print(f"\nErrors logged to oxford_error_log.csv")
    
    print("\nProcessing complete!")
