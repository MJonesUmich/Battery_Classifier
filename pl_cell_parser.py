import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import re

def load_meta_properties():
    #Finish function to associate file name with cell capacity, c-rate, and temperatures
    df = pd.read_excel(r'C:\Users\ShuS\Documents\github\Battery_Classifier\data\battery_data_mapper.xlsx', sheet_name='General_Infos')
    return df

def extract_date(file_name):
    # Extract MM, DD, YR from the file name
    # Remove file extension first
    base_name = os.path.splitext(file_name)[0]
    parts = base_name.split('_')
    
    # Handle PL file name formats:
    # 06_02_2015_25C_3M_Final.xls
    # 2_24_2015_PLN_IC_25C_3M_1.xls
    # 04_08_2015_PLN_IC_25C_6.xls
    # PL11, PL13 (no date in filename)
    
    # Find the date parts by looking for numeric parts from the beginning
    # Date should be: month (1-12), day (1-31), year (typically 14-15)
    numeric_parts = []
    for i, part in enumerate(parts):
        try:
            val = int(part)
            # Only consider reasonable values for date components
            if 1 <= val <= 31:  # Day range
                numeric_parts.append(val)
            elif 1 <= val <= 12:  # Month range
                numeric_parts.append(val)
            elif 10 <= val <= 99:  # Year range (10-99, will be converted to 2010-2099)
                numeric_parts.append(val)
            elif 2000 <= val <= 2099:  # Full year range
                numeric_parts.append(val)
            
            if len(numeric_parts) == 3:
                break
        except ValueError:
            # Skip non-numeric parts
            continue
    
    if len(numeric_parts) < 3:
        # For files without date in filename (like PL11), use a default date
        print(f"No date found in filename: {file_name}, using default date")
        return datetime(2015, 1, 1).date()
    
    # The first 3 numeric parts should be: month, day, year
    month, day, year = numeric_parts[0], numeric_parts[1], numeric_parts[2]
    
    # Fix year: if it's a 2-digit year, assume it's 20xx
    if year < 100:
        year = 2000 + year
    
    # Validate the date components
    if not (1 <= month <= 12):
        raise ValueError(f"Invalid month: {month}")
    if not (1 <= day <= 31):
        raise ValueError(f"Invalid day: {day}")
    if not (2000 <= year <= 2099):
        raise ValueError(f"Invalid year: {year}")
    
    print(month, day, year)
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
    
    # Look for sheets that contain the required columns
    for s in xls.sheet_names:
        try:
            # Read just the header to check columns
            cols = set(pd.read_excel(file_path, sheet_name=s, nrows=1).columns.astype(str))
            if {"Current(A)", "Voltage(V)", "Test_Time(s)"} <= cols:
                chosen = s
                break
        except Exception as e:
            # Skip sheets that can't be read
            continue
    
    if chosen is None:
        # If no suitable sheet found, try the first sheet
        chosen = xls.sheet_names[0]

    df = pd.read_excel(file_path, sheet_name=chosen)
    df.columns = [str(c).strip() for c in df.columns]

    # Get the desired columns out: 
    quant_cols = ["Current(A)", "Voltage(V)", "Test_Time(s)"]
    #In the dataframe force these columns to be float
    for col in quant_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Only keep the desired columns that exist
    available_cols = [col for col in quant_cols if col in df.columns]
    if len(available_cols) < 3:
        raise ValueError(f"Required columns not found. Available columns: {df.columns.tolist()}")
    
    df = df[available_cols].dropna().reset_index(drop=True)
    return df

def load_from_csv_file(file_path):
    """Load data from CSV file with the new format"""
    try:
        df = pd.read_csv(file_path)
        df.columns = [str(c).strip() for c in df.columns]
        
        # Map the new CSV column names to the expected format
        column_mapping = {
            'Time_sec': 'Test_Time(s)',
            'Current_Amp': 'Current(A)', 
            'Voltage_Volt': 'Voltage(V)'
        }
        
        # Rename columns if they exist
        for old_col, new_col in column_mapping.items():
            if old_col in df.columns:
                df = df.rename(columns={old_col: new_col})
        
        # Get the desired columns
        quant_cols = ["Current(A)", "Voltage(V)", "Test_Time(s)"]
        
        # Force these columns to be float
        for col in quant_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Only keep the desired columns that exist
        available_cols = [col for col in quant_cols if col in df.columns]
        if len(available_cols) < 3:
            raise ValueError(f"Required columns not found. Available columns: {df.columns.tolist()}")
        
        df = df[available_cols].dropna().reset_index(drop=True)
        return df
        
    except Exception as e:
        print(f"Error loading CSV file {file_path}: {e}")
        return pd.DataFrame() 

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


def scrub_and_tag(df, charge_indices, discharge_indices, cell_initial_capacity, vmax=None, vmin=None, tolerance=0.01):
    # Determine the correct order and range
    all_indices = sorted(charge_indices + discharge_indices)
    start_idx = all_indices[0]
    end_idx = all_indices[-1]
    
    # Downsample to the range containing all cycles
    df = df.iloc[start_idx:end_idx + 1].reset_index(drop=True)

    # Adjust charge_indices and discharge_indices to match the new DataFrame
    adjusted_charge_indices = [i - start_idx for i in charge_indices]
    adjusted_discharge_indices = [i - start_idx for i in discharge_indices]

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
    excluding constant voltage holds.
    """
    if len(cycle_data) == 0:
        return cycle_data
    
    # Find voltage range indices
    voltage = cycle_data['Voltage(V)'].values
    current = cycle_data['Current(A)'].values
    
    # Find the first point where voltage reaches vmax (during charge)
    vmax_reached = False
    vmax_idx = None
    for i, v in enumerate(voltage):
        if v >= vmax - tolerance and not vmax_reached:
            vmax_reached = True
            vmax_idx = i
            break
    
    # Find the first point where voltage reaches vmin (during discharge)
    vmin_reached = False
    vmin_idx = None
    for i, v in enumerate(voltage):
        if v <= vmin + tolerance and not vmin_reached:
            vmin_reached = True
            vmin_idx = i
            break
    
    # Find discharge start (first negative current)
    discharge_start = None
    for i, c in enumerate(current):
        if c < -tolerance:
            discharge_start = i
            break
    
    # Filter data based on voltage range
    if vmax_idx is not None and vmin_idx is not None and discharge_start is not None:
        # Include charge portion up to vmax and discharge portion from start to vmin
        charge_portion = cycle_data.iloc[:vmax_idx+1]
        discharge_portion = cycle_data.iloc[discharge_start:vmin_idx+1]
        
        # Combine charge and discharge portions
        filtered_data = pd.concat([charge_portion, discharge_portion], ignore_index=True)
        return filtered_data
    else:
        # If we can't find proper voltage bounds, return original data
        return cycle_data


def update_df(df, agg_df): 
    if len(agg_df) == 0:
        return df
    else: 
        max_cycle = agg_df['Cycle_Count'].max()
        # Handle None values in Cycle_Count
        df['Cycle_Count'] = df['Cycle_Count'].fillna(0).astype(int) + int(max_cycle)
        df['Ah_throughput'] = df['Ah_throughput'] + agg_df['Ah_throughput'].max()
        df['EFC'] = df['EFC'] + agg_df['EFC'].max()
        return df


def load_from_text_file(file_path):
    # Load data from a text file
    df = pd.read_csv(file_path, delimiter='\t')
    #rename columns:
    df.rename(columns={'Time': 'Test_Time(s)', 'mA': 'Current(A)', 'mV': 'Voltage(V)'}, inplace=True)
    desired_cols = ["Current(A)", "Voltage(V)", "Test_Time(s)"]
    df = df[desired_cols].dropna().reset_index(drop=True)
    df["Voltage(V)"] = df["Voltage(V)"] / 1000  # Convert mV to V
    df["Current(A)"] = df["Current(A)"] / 1000  # Convert mA to A
    return df




def parse_file(file_path, cell_initial_capacity, cell_C_rate, method = 'excel', vmax=None, vmin=None):
    if method == 'excel': 
        df = load_file(file_path)
    elif method == 'text': 
        df = load_from_text_file(file_path)
    elif method == 'csv':
        df = load_from_csv_file(file_path)

    if len(df) == 0:
        print(f"No data loaded from {file_path}")
        return pd.DataFrame()
    
    # Debug info only for successful loads
    if len(df) > 0:
        print(f"Successfully loaded {len(df)} rows from {file_path}")
        try:
            print(f"Current range: {df['Current(A)'].min():.3f} to {df['Current(A)'].max():.3f} A")
            print(f"Voltage range: {df['Voltage(V)'].min():.3f} to {df['Voltage(V)'].max():.3f} V")
        except Exception as e:
            print(f"Data loaded but may have format issues: {e}")

    charge_indices, discharge_indices = get_indices(df)
    
    if len(charge_indices) == 0 or len(discharge_indices) == 0:
        print(f"No charge/discharge indices found in {file_path}")
        return pd.DataFrame()
    
    print(f"Found {len(charge_indices)} charge indices and {len(discharge_indices)} discharge indices")
    
    df = scrub_and_tag(df, charge_indices, discharge_indices, cell_initial_capacity, vmax, vmin)
    df["C_rate"] = cell_C_rate
    return df

def generate_figures(df, vmax, vmin, c_rate, temperature, battery_ID, tolerance=0.01,one_fig_only=False):
    unique_cycles = df['Cycle_Count'].dropna().unique()
    if len(unique_cycles) == 0:
        print("No valid cycles found for figure generation")
        return
        
    for i, cycle in enumerate(unique_cycles):
        cycle_df = df[df['Cycle_Count'] == cycle]
        
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


if __name__ == "__main__": 
    #Example run through on 1 file
    meta_df = load_meta_properties()

    folder_path = r'C:\Users\ShuS\Documents\github\Battery_Classifier\data\raw\PL\Capacity_25C'
    #folder_path = r'C:\Users\ShuS\Documents\github\Battery_Classifier\data\raw\PL\Capacity_50C'
    #folder_path = r'C:\Users\ShuS\Documents\github\Battery_Classifier\data\raw\PL\SOC_0%-100%_HalfC'

    # Recursively find all data files in subdirectories
    file_names = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith(('.xls', '.xlsx', '.csv')):
                file_path_full = os.path.join(root, file)
                #skip files < 200kb since they don't have enough data to actually consider:  
                if os.path.getsize(file_path_full) > 200*1024:
                    file_names.append(file_path_full)
    
    # Extract just the filenames for sorting
    file_names_only = [os.path.basename(f) for f in file_names]

    sorted_files, file_dates = sort_files(file_names_only)
    sorted_files = sorted_files[::-1]
    file_dates = file_dates[::-1]

    print(f"Found {len(file_names)} files to process")
    print("Sample files:", sorted_files[:3])
    print("Processing files...")
    error_dict = {}

    agg_df = pd.DataFrame()

    # Derive Battery ID from folder name (e.g., "Capacity_25C" -> "PL")
    cell_id = "PL"  # Default for PL data

    print(f"Looking for battery ID: {cell_id}")
    cell_df = meta_df[meta_df["Battery_ID"].str.lower() == str.lower(cell_id)]
    print(f"Found {len(cell_df)} matching records")
    
    if len(cell_df) == 0:
        print(f"Available Battery_IDs: {meta_df['Battery_ID'].dropna().unique()}")
        # Use default values if no metadata found
        cell_initial_capacity = 1.0
        cell_C_rate = 1.0
        cell_temperature = 298.0
        cell_vmax = 4.2
        cell_vmin = 3.0
        print("Using default values for PL data")
    else:
        cell_initial_capacity = cell_df["Initial_Capacity_Ah"].values[0]
        cell_C_rate = cell_df["C_rate"].values[0]
        cell_temperature = cell_df["Temperature (K)"].values[0]
        cell_vmax = cell_df["Max_Voltage"].values[0]
        cell_vmin = cell_df["Min_Voltage"].values[0]

    for i_count, file_name in enumerate(sorted_files): 
        print(f"Processing file {i_count+1}/{len(sorted_files)}: {file_name}")
        try: 
            # Find the full path for this file
            file_path = None
            for full_path in file_names:
                if os.path.basename(full_path) == file_name:
                    file_path = full_path
                    break
            
            if file_path is None:
                print(f"Could not find full path for {file_name}")
                continue
                
            if file_name.endswith('.txt'):
                method = 'text'
            elif file_name.endswith('.xlsx') or file_name.endswith('.xls'):
                method = 'excel'
            elif file_name.endswith('.csv'):
                method = 'csv'

            df = parse_file(file_path, cell_initial_capacity, cell_C_rate, method, cell_vmax, cell_vmin)
            if len(df) > 0:
                update_df(df, agg_df)
                agg_df = pd.concat([agg_df, df], ignore_index=True)

        #except add failed files to dictionary with error message
        except Exception as e:
            error_dict[file_name] = str(e)
        
        print(f'{round(i_count/len(sorted_files)*100,1)}% Complete')

    #send to df and output: 
    error_df = pd.DataFrame(list(error_dict.items()), columns=['File_Name', 'Error_Message'])
    error_df.to_csv('error_log.csv', index=False)
    agg_df.to_csv('aggregated_pl_data.csv', index=False)
    
    # Generate figures using the last processed file's data
    if not agg_df.empty:
        generate_figures(agg_df, cell_vmax, cell_vmin, cell_C_rate, cell_temperature, battery_ID=cell_id, one_fig_only=True)
    else:
        print("No data available for figure generation")
