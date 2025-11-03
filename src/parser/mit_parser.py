"""MIT parser aligned with the consolidated aggregation workflow."""

import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import h5py
import numpy as np
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.help_function import load_meta_properties


@dataclass
class ProcessingConfig:
    '''Configuration for MIT data processing.'''

    raw_data_rel_path: str = os.path.join('assets', 'raw', 'MIT')
    processed_rel_root: str = os.path.join('assets', 'processed')
    chemistry: str = 'LFP'
    sample_points: int = 100
    thread_count: int = 8
    max_cycles: int = 100

    def get_raw_data_path(self, project_root: str) -> str:
        return os.path.join(project_root, self.raw_data_rel_path)

    def get_processed_dir(
        self, project_root: str, battery_id: Optional[str] = None
    ) -> str:
        base = os.path.join(project_root, self.processed_rel_root, self.chemistry)
        if battery_id:
            return os.path.join(base, battery_id)
        return base


@dataclass
class CellMetadata:
    '''Metadata describing a MIT cell.'''

    initial_capacity: float
    c_rate_charge: float
    c_rate_discharge: float
    temperature: float


MIN_SEGMENT_SAMPLE_COUNT = 100


def safe_float(value, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def load_cell_dataframe_from_csv(file_path: str) -> pd.DataFrame:
    '''Load a CSV that contains per-cycle MIT measurements.'''

    df = pd.read_csv(file_path)

    rename_map = {
        'current(A)': 'Current(A)',
        'voltage(V)': 'Voltage(V)',
        'test_time(s)': 'Test_Time(s)',
        'cycle_count': 'Cycle_Count',
    }
    df.rename(columns=rename_map, inplace=True)

    required_cols = {'Cycle_Count', 'Current(A)', 'Voltage(V)', 'Test_Time(s)'}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f'Missing required columns {missing} in {file_path!r}')

    df = df[list(required_cols)]
    df = df.dropna(subset=['Cycle_Count', 'Current(A)', 'Voltage(V)', 'Test_Time(s)'])

    df['Cycle_Count'] = df['Cycle_Count'].astype(int)
    df['Current(A)'] = df['Current(A)'].astype(float)
    df['Voltage(V)'] = df['Voltage(V)'].astype(float)
    df['Test_Time(s)'] = df['Test_Time(s)'].astype(float)

    df = df.sort_values(['Cycle_Count', 'Test_Time(s)']).reset_index(drop=True)
    return df


def _read_mat_cycle(dataset, index: int, handle: h5py.File) -> np.ndarray:
    try:
        ref = dataset[index, 0]
        return np.array(handle[ref]).squeeze()
    except Exception:  # noqa: BLE001 - h5 datasets raise various errors
        return np.array([])


def extract_cells_from_mat(file_path: str) -> Dict[str, pd.DataFrame]:
    '''Extract per-cell dataframes from a MIT .mat file.'''

    cells: Dict[str, pd.DataFrame] = {}
    file_name = os.path.splitext(os.path.basename(file_path))[0]

    with h5py.File(file_path, 'r') as handle:
        batch = handle.get('batch')
        if batch is None or 'cycles' not in batch:
            return cells

        cell_qty = batch['summary'].shape[0]
        for cell_idx in range(cell_qty):
            cycles_ref = batch['cycles'][cell_idx, 0]
            cycles = handle[cycles_ref]
            if 'I' not in cycles or 'V' not in cycles or 't' not in cycles:
                continue

            num_cycles = cycles['I'].shape[0]
            segments: List[pd.DataFrame] = []

            for cycle_idx in range(num_cycles):
                currents = _read_mat_cycle(cycles['I'], cycle_idx, handle)
                voltages = _read_mat_cycle(cycles['V'], cycle_idx, handle)
                times = _read_mat_cycle(cycles['t'], cycle_idx, handle)

                if currents.size == 0 or voltages.size == 0 or times.size == 0:
                    continue

                try:
                    currents = currents.astype(float)
                    voltages = voltages.astype(float)
                    times = times.astype(float)
                except ValueError:
                    continue

                times_seconds = times * 60.0
                times_seconds = times_seconds - times_seconds[0]

                segment = pd.DataFrame(
                    {
                        'Cycle_Count': np.full_like(currents, cycle_idx + 1, dtype=int),
                        'Current(A)': currents,
                        'Voltage(V)': voltages,
                        'Test_Time(s)': times_seconds,
                    }
                )

                segments.append(segment)

            if segments:
                battery_id = f'{file_name}_cell_{cell_idx}'
                cell_df = pd.concat(segments, ignore_index=True)
                cells[battery_id] = cell_df

    return cells


def contiguous_blocks(indices: np.ndarray) -> List[np.ndarray]:
    if indices.size == 0:
        return []
    splits = np.where(np.diff(indices) > 1)[0] + 1
    return np.split(indices, splits)


def select_block(
    blocks: List[np.ndarray],
    cycle_df: pd.DataFrame,
    min_length: int = 5,
    start_after: Optional[int] = None,
) -> Optional[np.ndarray]:
    def block_duration(block: np.ndarray) -> float:
        start = block[0]
        end = block[-1]
        return float(
            cycle_df.iloc[end]['Test_Time(s)'] - cycle_df.iloc[start]['Test_Time(s)']
        )

    filtered: List[np.ndarray] = []
    for block in blocks:
        if start_after is not None and block[0] <= start_after:
            continue
        if len(block) < min_length:
            continue
        filtered.append(block)

    if filtered:
        return max(filtered, key=block_duration)

    if start_after is not None:
        later_blocks = [block for block in blocks if block[0] > start_after]
        if later_blocks:
            return max(later_blocks, key=block_duration)

    return max(blocks, key=block_duration) if blocks else None


def split_cycle_segments(
    cycle_df: pd.DataFrame, tolerance: float = 1e-4
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if cycle_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    current = cycle_df['Current(A)'].to_numpy()
    positive_indices = np.where(current > tolerance)[0]
    negative_indices = np.where(current < -tolerance)[0]

    positive_blocks = contiguous_blocks(positive_indices)
    negative_blocks = contiguous_blocks(negative_indices)

    charge_segment = pd.DataFrame()
    discharge_segment = pd.DataFrame()

    first_positive = positive_blocks[0][0] if positive_blocks else None
    first_negative = negative_blocks[0][0] if negative_blocks else None

    if first_positive is not None and (
        first_negative is None or first_positive <= first_negative
    ):
        charge_block = select_block(positive_blocks, cycle_df)
        if charge_block is not None:
            charge_segment = cycle_df.iloc[charge_block[0] : charge_block[-1] + 1].copy()

        if negative_blocks:
            charge_last = charge_block[-1] if charge_block is not None else None
            discharge_block = select_block(
                negative_blocks, cycle_df, start_after=charge_last
            )
            if discharge_block is None:
                discharge_block = select_block(negative_blocks, cycle_df)
            if discharge_block is not None:
                discharge_segment = cycle_df.iloc[
                    discharge_block[0] : discharge_block[-1] + 1
                ].copy()
    elif first_negative is not None:
        discharge_block = select_block(negative_blocks, cycle_df)
        if discharge_block is not None:
            discharge_segment = cycle_df.iloc[
                discharge_block[0] : discharge_block[-1] + 1
            ].copy()

        if positive_blocks:
            discharge_last = discharge_block[-1] if discharge_block is not None else None
            charge_block = select_block(
                positive_blocks, cycle_df, start_after=discharge_last
            )
            if charge_block is None:
                charge_block = select_block(positive_blocks, cycle_df)
            if charge_block is not None:
                charge_segment = cycle_df.iloc[
                    charge_block[0] : charge_block[-1] + 1
                ].copy()

    return charge_segment, discharge_segment


def resample_cycle_segment(segment_df: pd.DataFrame, sample_points: int) -> pd.DataFrame:
    required_cols = ['Test_Time(s)', 'Voltage(V)', 'Current(A)']
    segment = (
        segment_df[required_cols]
        .dropna()
        .drop_duplicates(subset=['Test_Time(s)'])
        .sort_values('Test_Time(s)')
    )

    if segment.empty:
        return pd.DataFrame()

    time_values = segment['Test_Time(s)'].to_numpy(dtype=float)
    elapsed = time_values - time_values[0]

    result = pd.DataFrame(
        {
            'Sample_Index': np.arange(sample_points, dtype=int),
            'Normalized_Time': np.linspace(0.0, 1.0, sample_points),
        }
    )

    if len(segment) == 1 or np.isclose(elapsed[-1], 0.0):
        result['Elapsed_Time(s)'] = np.full(sample_points, float(elapsed[-1]))
        result['Voltage(V)'] = np.full(sample_points, segment['Voltage(V)'].iloc[0])
        result['Current(A)'] = np.full(sample_points, segment['Current(A)'].iloc[0])
        return result

    normalized = elapsed / (elapsed[-1] if elapsed[-1] else 1.0)
    target = result['Normalized_Time'].to_numpy()

    result['Elapsed_Time(s)'] = np.interp(target, normalized, elapsed)
    result['Voltage(V)'] = np.interp(
        target, normalized, segment['Voltage(V)'].to_numpy(dtype=float)
    )
    result['Current(A)'] = np.interp(
        target, normalized, segment['Current(A)'].to_numpy(dtype=float)
    )

    return result


def prepare_cycle_segment(
    segment_df: pd.DataFrame,
    min_samples: int,
) -> Optional[pd.DataFrame]:
    '''Sanitize segment data and enforce minimum raw samples.'''

    required_cols = ['Test_Time(s)', 'Voltage(V)', 'Current(A)']

    if segment_df is None or segment_df.empty:
        return None

    if any(col not in segment_df.columns for col in required_cols):
        return None

    sanitized = segment_df[required_cols].copy()

    for col in required_cols:
        sanitized[col] = pd.to_numeric(sanitized[col], errors='coerce')

    sanitized = sanitized.dropna()
    if sanitized.empty:
        return None

    sanitized = sanitized.drop_duplicates(subset=['Test_Time(s)'])
    sanitized = sanitized.sort_values('Test_Time(s)')

    if len(sanitized) < min_samples:
        return None

    if sanitized['Test_Time(s)'].iloc[-1] <= sanitized['Test_Time(s)'].iloc[0]:
        return None

    return sanitized.reset_index(drop=True)


def format_resampled_segment(
    resampled: pd.DataFrame,
    battery_id: str,
    chemistry: str,
    cycle_index: int,
    c_rate: float,
    temperature: float,
) -> pd.DataFrame:
    resampled = resampled.rename(
        columns={
            'Sample_Index': 'sample_index',
            'Normalized_Time': 'normalized_time',
            'Elapsed_Time(s)': 'elapsed_time_s',
            'Voltage(V)': 'voltage_v',
            'Current(A)': 'current_a',
        }
    )

    resampled['battery_id'] = battery_id
    resampled['chemistry'] = chemistry
    resampled['cycle_index'] = int(cycle_index)
    resampled['c_rate'] = float(c_rate)
    resampled['temperature_k'] = float(temperature)

    resampled = resampled[
        [
            'battery_id',
            'chemistry',
            'cycle_index',
            'sample_index',
            'normalized_time',
            'elapsed_time_s',
            'voltage_v',
            'current_a',
            'c_rate',
            'temperature_k',
        ]
    ]

    resampled['sample_index'] = resampled['sample_index'].astype(int)
    return resampled


def process_battery_dataframe(
    df: pd.DataFrame,
    battery_id: str,
    cell_meta: CellMetadata,
    config: ProcessingConfig,
) -> Tuple[pd.DataFrame, pd.DataFrame, List[Dict[str, str]]]:
    errors: List[Dict[str, str]] = []
    charge_segments: List[pd.DataFrame] = []
    discharge_segments: List[pd.DataFrame] = []

    cycles = sorted(int(cycle) for cycle in pd.unique(df['Cycle_Count']))

    for idx, cycle in enumerate(cycles, start=1):
        if idx > config.max_cycles:
            errors.append(
                {
                    'Cycle_Index': cycle,
                    'Direction': 'both',
                    'Message': 'skipped due to max cycle limit',
                }
            )
            continue

        cycle_df = df[df['Cycle_Count'] == cycle].copy()
        cycle_df = cycle_df.dropna(subset=['Current(A)', 'Voltage(V)', 'Test_Time(s)'])
        if cycle_df.empty:
            errors.append(
                {
                    'Cycle_Index': cycle,
                    'Direction': 'both',
                    'Message': 'cycle has no data',
                }
            )
            continue

        cycle_df = (
            cycle_df.sort_values('Test_Time(s)')
            .drop_duplicates(subset=['Test_Time(s)'])
            .reset_index(drop=True)
        )
        cycle_df['Test_Time(s)'] = cycle_df['Test_Time(s)'] - cycle_df['Test_Time(s)'].iloc[0]

        charge_segment_raw, discharge_segment_raw = split_cycle_segments(cycle_df)

        min_samples_required = max(config.sample_points, MIN_SEGMENT_SAMPLE_COUNT)

        charge_segment = prepare_cycle_segment(
            charge_segment_raw, min_samples_required
        )
        discharge_segment = prepare_cycle_segment(
            discharge_segment_raw, min_samples_required
        )

        if charge_segment is None:
            errors.append(
                {
                    'Cycle_Index': cycle,
                    'Direction': 'charge',
                    'Message': 'no positive current segment meeting sample requirement',
                }
            )
        else:
            resampled = resample_cycle_segment(charge_segment, config.sample_points)
            if resampled.empty:
                errors.append(
                    {
                        'Cycle_Index': cycle,
                        'Direction': 'charge',
                        'Message': 'resampling produced empty output',
                    }
                )
            else:
                formatted = format_resampled_segment(
                    resampled,
                    battery_id,
                    config.chemistry,
                    idx,
                    cell_meta.c_rate_charge,
                    cell_meta.temperature,
                )
                charge_segments.append(formatted)

        if discharge_segment is None:
            errors.append(
                {
                    'Cycle_Index': cycle,
                    'Direction': 'discharge',
                    'Message': 'no negative current segment meeting sample requirement',
                }
            )
        else:
            resampled = resample_cycle_segment(discharge_segment, config.sample_points)
            if resampled.empty:
                errors.append(
                    {
                        'Cycle_Index': cycle,
                        'Direction': 'discharge',
                        'Message': 'resampling produced empty output',
                    }
                )
            else:
                formatted = format_resampled_segment(
                    resampled,
                    battery_id,
                    config.chemistry,
                    idx,
                    cell_meta.c_rate_discharge,
                    cell_meta.temperature,
                )
                discharge_segments.append(formatted)

    expected_columns = [
        'battery_id',
        'chemistry',
        'cycle_index',
        'sample_index',
        'normalized_time',
        'elapsed_time_s',
        'voltage_v',
        'current_a',
        'c_rate',
        'temperature_k',
    ]

    charge_df = (
        pd.concat(charge_segments, ignore_index=True)
        if charge_segments
        else pd.DataFrame(columns=expected_columns)
    )
    discharge_df = (
        pd.concat(discharge_segments, ignore_index=True)
        if discharge_segments
        else pd.DataFrame(columns=expected_columns)
    )

    if not charge_df.empty:
        charge_df = charge_df.sort_values(['cycle_index', 'sample_index'])
        charge_df.reset_index(drop=True, inplace=True)

    if not discharge_df.empty:
        discharge_df = discharge_df.sort_values(['cycle_index', 'sample_index'])
        discharge_df.reset_index(drop=True, inplace=True)

    return charge_df, discharge_df, errors


def get_cell_metadata(meta_df: pd.DataFrame, battery_id: str) -> CellMetadata:
    cell_df = meta_df[meta_df['Battery_ID'].str.lower() == battery_id.lower()]
    if cell_df.empty:
        return CellMetadata(
            initial_capacity=1.1,
            c_rate_charge=1.0,
            c_rate_discharge=1.0,
            temperature=298.0,
        )

    row = cell_df.iloc[0]
    return CellMetadata(
        initial_capacity=safe_float(row.get('Initial_Capacity_Ah'), 1.1),
        c_rate_charge=safe_float(row.get('C_rate_Charge'), 1.0),
        c_rate_discharge=safe_float(row.get('C_rate_Discharge'), 1.0),
        temperature=safe_float(row.get('Temperature (K)'), 298.0),
    )


def save_processed_data(
    charge_df: pd.DataFrame,
    discharge_df: pd.DataFrame,
    battery_id: str,
    config: ProcessingConfig,
    output_dir: str,
) -> Tuple[str, str]:
    os.makedirs(output_dir, exist_ok=True)

    charge_path = os.path.join(output_dir, f'{battery_id}_charge_aggregated_data.csv')
    discharge_path = os.path.join(
        output_dir, f'{battery_id}_discharge_aggregated_data.csv'
    )

    charge_df.to_csv(charge_path, index=False)
    discharge_df.to_csv(discharge_path, index=False)

    print(f'💾 Saved charge CSV: {charge_path}')
    print(f'💾 Saved discharge CSV: {discharge_path}')

    return charge_path, discharge_path


def save_error_log(errors: List[Dict[str, str]], battery_id: str, output_dir: str) -> None:
    if not errors:
        return

    os.makedirs(output_dir, exist_ok=True)
    error_df = pd.DataFrame(errors)
    error_log_path = os.path.join(output_dir, f'error_log_{battery_id}.csv')
    error_df.to_csv(error_log_path, index=False)
    print(f'📝 Saved error log: {error_log_path}')


def process_raw_file(
    file_name: str,
    raw_base_path: str,
    processed_base_path: str,
    meta_df: pd.DataFrame,
    config: ProcessingConfig,
) -> Dict[str, str]:
    file_path = os.path.join(raw_base_path, file_name)
    root, ext = os.path.splitext(file_name)
    ext = ext.lower()

    cell_map: Dict[str, pd.DataFrame]

    if ext == '.csv':
        try:
            cell_df = load_cell_dataframe_from_csv(file_path)
        except Exception as exc:  # noqa: BLE001
            return {file_name: str(exc)}
        cell_map = {root: cell_df}
    elif ext == '.mat':
        try:
            cell_map = extract_cells_from_mat(file_path)
        except Exception as exc:  # noqa: BLE001
            return {file_name: str(exc)}
        if not cell_map:
            return {file_name: 'no usable cells in mat file'}
    else:
        return {file_name: f"unsupported extension '{ext}'"}

    errors: Dict[str, str] = {}

    for battery_id, df in cell_map.items():
        if df.empty:
            errors[battery_id] = 'no data available'
            continue

        cell_meta = get_cell_metadata(meta_df, battery_id)
        charge_df, discharge_df, cycle_errors = process_battery_dataframe(
            df, battery_id, cell_meta, config
        )

        output_dir = os.path.join(processed_base_path, battery_id)
        save_processed_data(charge_df, discharge_df, battery_id, config, output_dir)
        save_error_log(cycle_errors, battery_id, output_dir)

    return errors


def main(config: Optional[ProcessingConfig] = None) -> None:
    if config is None:
        config = ProcessingConfig()

    start_time = time.time()
    print(
        f'🚀 Starting MIT battery data processing with {config.thread_count} threads...'
    )

    meta_df = load_meta_properties()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.join(script_dir, '..', '..')
    raw_base_path = config.get_raw_data_path(project_root)
    processed_base_path = config.get_processed_dir(project_root)
    os.makedirs(processed_base_path, exist_ok=True)

    raw_files = sorted(
        file_name
        for file_name in os.listdir(raw_base_path)
        if os.path.splitext(file_name)[1].lower() in {'.csv', '.mat'}
    )

    print(f'📂 Found {len(raw_files)} MIT source files')

    aggregated_errors: Dict[str, str] = {}

    with ThreadPoolExecutor(max_workers=config.thread_count) as executor:
        future_to_file = {
            executor.submit(
                process_raw_file,
                file_name,
                raw_base_path,
                processed_base_path,
                meta_df,
                config,
            ): file_name
            for file_name in raw_files
        }

        for future in as_completed(future_to_file):
            file_name = future_to_file[future]
            try:
                errors = future.result()
                aggregated_errors.update(errors)
                print(f'✅ Completed processing source file: {file_name}')
            except Exception as exc:  # noqa: BLE001
                aggregated_errors[file_name] = str(exc)
                print(f'✗ Error processing {file_name}: {exc}')

    if aggregated_errors:
        error_log_path = os.path.join(processed_base_path, 'error_log_mit.csv')
        pd.DataFrame(
            list(aggregated_errors.items()), columns=['Source', 'Error_Message']
        ).to_csv(error_log_path, index=False)
        print(f'📝 Saved global error log: {error_log_path}')

    total_time = time.time() - start_time
    hours = int(total_time // 3600)
    minutes = int((total_time % 3600) // 60)
    seconds = int(total_time % 60)

    print(f"\n{'=' * 60}")
    print('🎉 MIT parsing completed!')
    print(f'⏱️  Total processing time: {hours:02d}:{minutes:02d}:{seconds:02d}')
    print(f'📊 Processed {len(raw_files)} source files')
    if raw_files:
        print(f'⚡ Average time per file: {total_time / len(raw_files):.2f} seconds')
    print(f"{'=' * 60}")


if __name__ == '__main__':
    main()
# ============================================================
# 🎉 MIT parsing completed!
# ⏱️  Total processing time: 00:03:26
# 📊 Processed 140 source files
# ⚡ Average time per file: 1.48 seconds
# ============================================================