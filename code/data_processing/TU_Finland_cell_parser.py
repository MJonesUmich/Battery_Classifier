import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import loadmat

def load_meta_properties():
    """Load battery metadata from Excel file"""
    df = pd.read_excel(r'C:\Users\ShuS\Documents\github\Battery_Classifier\data\battery_data_mapper.xlsx', sheet_name='General_Infos')
    return df

def extract_battery_specs_from_mat(file_path, percentile=95):
    """
    Extract battery specifications (capacity, C-rate, voltage limits) from TU Finland .mat format
    
    Parameters:
    - file_path: path to .mat file
    - percentile: percentile to use for determining voltage limits
    
    Returns:
    - vmax: maximum charge voltage
    - vmin: minimum discharge voltage
    - capacity: estimated battery capacity in Ah
    - c_rate: estimated C-rate
    """
    mat_data = loadmat(file_path)
    table = mat_data['table']
    
    all_voltages = []
    charge_voltages = []
    discharge_voltages = []
    all_currents = []
    charge_currents = []
    
    # Collect voltage and current data from all entries
    for i, entry in enumerate(table[0]):
        if len(entry[1]) > 0 and len(entry[2]) > 0:  # Has voltage and current data
            voltage_data = entry[1][0]  # Get voltage array
            current_data = entry[2][0]  # Get current array
            
            # Add all voltages for overall range
            all_voltages.extend(voltage_data)
            all_currents.extend(current_data)
            
            # Find charge and discharge portions based on current
            non_zero_mask = np.abs(current_data) > 0.01
            if np.any(non_zero_mask):
                charge_mask = current_data > 0.01
                discharge_mask = current_data < -0.01
                
                if np.any(charge_mask):
                    charge_voltages.extend(voltage_data[charge_mask])
                    charge_currents.extend(current_data[charge_mask])
                if np.any(discharge_mask):
                    discharge_voltages.extend(voltage_data[discharge_mask])
    
    if len(all_voltages) == 0:
        # Fallback to default values
        return 3.6, 2.5, 2.5, 1.0
    
    # Extract voltage limits
    if len(charge_voltages) > 0 and len(discharge_voltages) > 0:
        vmax = np.percentile(charge_voltages, percentile)
        vmin = np.percentile(discharge_voltages, 100 - percentile)
    else:
        # Use overall voltage range with more conservative percentiles
        vmax = np.percentile(all_voltages, 98)  # Use 98th percentile for max
        vmin = np.percentile(all_voltages, 2)   # Use 2nd percentile for min
    
    # Extract capacity and C-rate information
    if len(charge_currents) > 0:
        max_charge_current = np.max(charge_currents)
        
        if max_charge_current > 0:
            # Estimate capacity based on voltage characteristics and current magnitude
            if vmax > 4.0:  # High voltage battery (NCA/NMC)
                estimated_capacity = max(2.5, min(5.0, max_charge_current / 0.8))
            else:  # Lower voltage battery (LFP)
                estimated_capacity = max(1.5, min(4.0, max_charge_current / 0.8))
            
            c_rate = 1.0
        else:
            estimated_capacity = 2.5
            c_rate = 1.0
    else:
        estimated_capacity = 2.5
        c_rate = 1.0
    
    return vmax, vmin, estimated_capacity, c_rate

def extract_voltage_limits_from_mat(file_path, percentile=95):
    """
    Extract voltage limits (vmax and vmin) from TU Finland .mat format
    
    Parameters:
    - file_path: path to .mat file
    - percentile: percentile to use for determining voltage limits
    
    Returns:
    - vmax: maximum charge voltage
    - vmin: minimum discharge voltage
    """
    mat_data = loadmat(file_path)
    table = mat_data['table']
    
    all_voltages = []
    charge_voltages = []
    discharge_voltages = []
    
    # Collect voltage data from all entries
    for i, entry in enumerate(table[0]):
        if len(entry[1]) > 0 and len(entry[2]) > 0:  # Has voltage and current data
            voltage_data = entry[1][0]  # Get voltage array
            current_data = entry[2][0]  # Get current array
            
            # Add all voltages for overall range
            all_voltages.extend(voltage_data)
            
            # Find charge and discharge portions based on current
            non_zero_mask = np.abs(current_data) > 0.01
            if np.any(non_zero_mask):
                charge_mask = current_data > 0.01
                discharge_mask = current_data < -0.01
                
                if np.any(charge_mask):
                    charge_voltages.extend(voltage_data[charge_mask])
                if np.any(discharge_mask):
                    discharge_voltages.extend(voltage_data[discharge_mask])
    
    if len(all_voltages) == 0:
        # Fallback to default values
        return 3.6, 2.5
    
    # If we have charge/discharge data, use that for more accurate limits
    if len(charge_voltages) > 0 and len(discharge_voltages) > 0:
        vmax = np.percentile(charge_voltages, percentile)
        vmin = np.percentile(discharge_voltages, 100 - percentile)
        print(f"  Extracted from charge/discharge data: vmax={vmax:.3f}V, vmin={vmin:.3f}V")
    else:
        # Use overall voltage range with more conservative percentiles
        vmax = np.percentile(all_voltages, 98)  # Use 98th percentile for max
        vmin = np.percentile(all_voltages, 2)   # Use 2nd percentile for min
        print(f"  Extracted from overall voltage range: vmax={vmax:.3f}V, vmin={vmin:.3f}V")
    
    return vmax, vmin

def load_from_mat(file_path):
    """
    Load data from TU Finland .mat format
    Structure: table -> entries with Time, Voltage, Current, Temperature, Comment
    
    Parameters:
    - file_path: path to the .mat file
    
    Returns combined DataFrame for all test entries
    """
    mat_data = loadmat(file_path)
    table = mat_data['table']
    
    all_dfs = []
    
    for i, entry in enumerate(table[0]):
        if len(entry[1]) > 0 and len(entry[2]) > 0:  # Has voltage and current data
            voltage_data = entry[1][0]  # Get voltage array
            current_data = entry[2][0]  # Get current array
            temperature_data = entry[3][0] if len(entry[3]) > 0 else np.zeros_like(voltage_data)  # Get temperature array
            comment = entry[4][0] if len(entry[4]) > 0 else f'Entry_{i}'  # Get comment
            
            # Create time array (assuming uniform sampling)
            time_data = np.arange(len(voltage_data))
            
            # Create DataFrame for this entry
            entry_df = pd.DataFrame({
                'Test_Time(s)': time_data,
                'Voltage(V)': voltage_data,
                'Current(A)': current_data,
                'Temperature(K)': temperature_data,
                'Comment': comment,
                'Entry_Index': i
            })
            
            all_dfs.append(entry_df)
    
    if len(all_dfs) == 0:
        raise ValueError("No valid data found in mat file")
    
    # Combine all entries
    df = pd.concat(all_dfs, ignore_index=True)
    
    return df.dropna().reset_index(drop=True)

def get_indices(df):
    """Extract charge and discharge indices from current data"""
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
    """Check if charge and discharge indices alternate correctly"""
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
            complexity = "High"
            return complexity, expected_order

    complexity = "Low"
    return complexity, expected_order

def scrub_and_tag(df, charge_indices, discharge_indices, cell_initial_capacity, vmax=None, vmin=None, tolerance=0.01):
    """Process and tag the data with cycle information"""
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

    # Coulomb count Ah throughput for each cycle
    df['Delta_Time(s)'] = df['Test_Time(s)'].diff().fillna(0)
    df['Delta_Ah'] = np.abs(df['Current(A)']) * df['Delta_Time(s)'] / 3600
    df['Ah_throughput'] = df['Delta_Ah'].cumsum()

    # now calculate Equivalent Full Cycles (EFC) & Capacity Fade
    df['EFC'] = df['Ah_throughput'] / cell_initial_capacity
    return df

def filter_voltage_range(cycle_data, vmax, vmin, tolerance=0.01):
    """
    Filter cycle data to include only the voltage range between vmin and vmax,
    excluding constant voltage holds (CV phase).
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
    """Update DataFrame with aggregated data"""
    if len(agg_df) == 0:
        return df
    else: 
        max_cycle = agg_df['Cycle_Count'].max()
        df['Cycle_Count'] = df['Cycle_Count'].astype(int) + int(max_cycle)
        df['Ah_throughput'] = df['Ah_throughput'] + agg_df['Ah_throughput'].max()
        df['EFC'] = df['EFC'] + agg_df['EFC'].max()
        return df

def parse_file(file_path, cell_initial_capacity, cell_C_rate, vmax=None, vmin=None):
    """
    Parse TU Finland .mat file and process battery data
    
    Parameters:
    - file_path: path to .mat file
    - cell_initial_capacity: initial capacity in Ah
    - cell_C_rate: C-rate value
    - vmax, vmin: voltage limits for filtering
    """
    df = load_from_mat(file_path)

    charge_indices, discharge_indices = get_indices(df)
    df = scrub_and_tag(df, charge_indices, discharge_indices, cell_initial_capacity, vmax, vmin)
    df["C_rate"] = cell_C_rate
    return df

def parse_mat_file(file_path, cell_initial_capacity, cell_C_rate, vmax=None, vmin=None):
    """
    Wrapper function specifically for parsing TU Finland .mat files
    
    Parameters:
    - file_path: path to .mat file
    - cell_initial_capacity: initial capacity in Ah
    - cell_C_rate: C-rate value
    - vmax, vmin: voltage limits for filtering
    """
    return parse_file(file_path, cell_initial_capacity, cell_C_rate, vmax, vmin)

def generate_figures(df, vmax, vmin, c_rate, temperature, battery_ID, output_dir, tolerance=0.01, one_fig_only=False):
    """Generate charge and discharge profile figures"""
    # Create images subdirectory
    images_dir = os.path.join(output_dir, 'images')
    os.makedirs(images_dir, exist_ok=True)
    
    unique_cycles = df['Cycle_Count'].unique()
    for i, cycle in enumerate(unique_cycles):
        cycle_df = df[df['Cycle_Count'] == cycle].reset_index(drop=True)
        
        #find where voltage first hits vmax and vmin, and where first discharge occurs
        try:
            vmax_idx = cycle_df[cycle_df['Voltage(V)'] >= vmax - tolerance].index[0]
        except IndexError:
            vmax_idx = cycle_df['Voltage(V)'].idxmax()
            
        try:
            vmin_idx = cycle_df[cycle_df['Voltage(V)'] <= vmin + tolerance].index[0]
        except IndexError:
            vmin_idx = cycle_df['Voltage(V)'].idxmin()
            
        try:
            disch_start = cycle_df[cycle_df['Current(A)'] < 0 - tolerance].index[0]
        except IndexError:
            # For charge-only data, set discharge start to end of cycle
            disch_start = len(cycle_df) 
        
        #clip data to initial until Vmax, then from discharge start to Vmin
        charge_cycle_df = cycle_df.loc[0:vmax_idx].copy()
        
        # Find vmin_idx after discharge starts (not before)
        if disch_start < len(cycle_df):
            discharge_portion = cycle_df.loc[disch_start:]
            try:
                vmin_idx_adjusted = discharge_portion[discharge_portion['Voltage(V)'] <= vmin + tolerance].index[0]
            except IndexError:
                vmin_idx_adjusted = discharge_portion.index[-1]
            
            discharge_cycle_df = cycle_df.loc[disch_start:vmin_idx_adjusted].copy()
        else:
            # No discharge data available
            discharge_cycle_df = pd.DataFrame()
        
        # Check if we have valid data
        if len(charge_cycle_df) > 0:
            charge_cycle_df["Charge_Time(s)"] = charge_cycle_df["Test_Time(s)"] - charge_cycle_df["Test_Time(s)"].iloc[0]
            
            #generate plot, clipped last datum in case current reset to rest
            plt.figure(figsize=(10, 6))
            plt.plot(charge_cycle_df['Charge_Time(s)'], charge_cycle_df['Voltage(V)'], color='blue')
            plt.xlabel('Charge Time (s)')
            plt.ylabel('Voltage (V)', color='blue')
            plt.title(f'Cycle {cycle} Charge Profile')
            save_string = f"Cycle_{i+1}_charge_Crate_{c_rate}_tempK_{temperature}_batteryID_{battery_ID}.png"
            save_path = os.path.join(images_dir, save_string)
            plt.savefig(save_path)
            plt.close()
        
        if len(discharge_cycle_df) > 0:
            discharge_cycle_df["Discharge_Time(s)"] = discharge_cycle_df["Test_Time(s)"] - discharge_cycle_df["Test_Time(s)"].iloc[0]
            
            #plot current on secondary axis
            plt.figure(figsize=(10, 6))
            plt.plot(discharge_cycle_df['Discharge_Time(s)'], discharge_cycle_df['Voltage(V)'], 'r-') #remove last few points to avoid voltage recovery
            plt.ylabel('Voltage (V)', color='red')
            plt.title(f'Cycle {cycle} Discharge Profile')
            save_string = f"Cycle_{i+1}_discharge_Crate_{c_rate}_tempK_{temperature}_batteryID_{battery_ID}.png"
            save_path = os.path.join(images_dir, save_string)
            plt.savefig(save_path)
            plt.close()

        #Exit function after 1st run if one_fig_only is True
        if one_fig_only:
            break

def process_battery_type(battery_type, data_dir, output_dir, cell_initial_capacity=None, cell_C_rate=None, cell_temperature=298.15):
    """Process a specific battery type dataset"""
    print(f"\nProcessing TU Finland {battery_type} dataset...")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    error_dict = {}
    
    # Get list of .mat files in the directory
    files_to_process = [f for f in os.listdir(data_dir) if f.endswith('.mat') and not f.startswith('OCV')]
    print(f"Found {len(files_to_process)} files to process")
    
    # Extract battery specifications from the first file
    try:
        first_file = os.path.join(data_dir, files_to_process[0])
        cell_vmax, cell_vmin, cell_initial_capacity, cell_C_rate = extract_battery_specs_from_mat(first_file, percentile=95)
        print(f"Extracted specs: vmax={cell_vmax:.2f}V, vmin={cell_vmin:.2f}V, capacity={cell_initial_capacity:.1f}Ah, C-rate={cell_C_rate:.1f}C")
    except Exception as e:
        # Fallback to default values if extraction fails
        cell_vmax = 3.6
        cell_vmin = 2.5
        cell_initial_capacity = 2.5
        cell_C_rate = 1.0
        print(f"Using default values: vmax={cell_vmax}V, vmin={cell_vmin}V, capacity={cell_initial_capacity}Ah, C-rate={cell_C_rate}C")

    # Process each cell and save individual CSV files
    for file_name in files_to_process:
        try:
            cell_id = file_name.replace('.mat', '')
            file_path = os.path.join(data_dir, file_name)
            
            # Parse the mat file
            df = parse_mat_file(
                file_path, 
                cell_initial_capacity, 
                cell_C_rate, 
                cell_vmax, 
                cell_vmin
            )
            
            # Select only the required columns
            required_columns = [
                'Current(A)', 'Voltage(V)', 'Test_Time(s)', 'Cycle_Count', 
                'Delta_Time(s)', 'Delta_Ah', 'Ah_throughput', 'EFC', 'C_rate'
            ]
            
            # Filter to only include columns that exist in the dataframe
            available_columns = [col for col in required_columns if col in df.columns]
            df_filtered = df[available_columns]
            
            # Create filename: tu_finland_LFP_cell1_298K.csv
            temperature_str = f"{int(cell_temperature)}K"
            output_filename = f"tu_finland_{battery_type}_{cell_id}_{temperature_str}.csv"
            output_path = os.path.join(output_dir, output_filename)
            
            # Save individual CSV file
            df_filtered.to_csv(output_path, index=False)
            print(f"[OK] {file_name}: {len(df_filtered)} data points -> {output_filename}")
            
            # Generate figures for all cells
            try:
                generate_figures(
                    df, 
                    cell_vmax, 
                    cell_vmin, 
                    cell_C_rate, 
                    cell_temperature, 
                    battery_ID=cell_id,
                    output_dir=output_dir,
                    one_fig_only=True
                )
            except Exception as e:
                print(f"[WARN] Could not generate figures for {file_name}")
            
        except Exception as e:
            error_dict[file_name] = str(e)
            print(f"[ERROR] {file_name}: {str(e)}")

    # Save error log if there are errors
    if error_dict:
        error_df = pd.DataFrame(list(error_dict.items()), columns=['File', 'Error_Message'])
        error_log_path = os.path.join(output_dir, f'{battery_type.lower()}_error_log.csv')
        error_df.to_csv(error_log_path, index=False)
        print(f"Errors logged to {error_log_path}")
    
    print(f"[DONE] {battery_type} processing complete!")
    return error_dict

if __name__ == "__main__": 
    # Battery type configurations (capacity and C-rate will be extracted from data)
    battery_configs = {
        'LFP': {
            'data_dir': r'C:\Users\ShuS\Documents\github\Battery_Classifier\data\raw\TU_Finland\DATASET\LFP',
            'output_dir': 'TU_Finland_LFP',
            'cell_temperature': 298  
        },
        'NCA': {
            'data_dir': r'C:\Users\ShuS\Documents\github\Battery_Classifier\data\raw\TU_Finland\DATASET\NCA',
            'output_dir': 'TU_Finland_NCA',
            'cell_temperature': 298
        },
        'NMC': {
            'data_dir': r'C:\Users\ShuS\Documents\github\Battery_Classifier\data\raw\TU_Finland\DATASET\NMC',
            'output_dir': 'TU_Finland_NMC',
            'cell_temperature': 298
        }
    }
    
    # Process all battery types
    all_errors = {}
    for battery_type, config in battery_configs.items():
        try:
            errors = process_battery_type(
                battery_type=battery_type,
                data_dir=config['data_dir'],
                output_dir=config['output_dir'],
                cell_initial_capacity=None,  # Will be extracted from data
                cell_C_rate=None,  # Will be extracted from data
                cell_temperature=config['cell_temperature']
            )
            all_errors[battery_type] = errors
        except Exception as e:
            print(f"Error processing {battery_type}: {str(e)}")
            all_errors[battery_type] = {'general_error': str(e)}
    
    # Summary
    print(f"\n{'='*40}")
    print("PROCESSING SUMMARY")
    print(f"{'='*40}")
    for battery_type, errors in all_errors.items():
        if errors:
            print(f"[FAIL] {battery_type}: {len(errors)} errors")
        else:
            print(f"[PASS] {battery_type}: SUCCESS")
    print(f"{'='*40}")
