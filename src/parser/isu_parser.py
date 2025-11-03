"""ISU parser aligned with the consolidated aggregation workflow."""

import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from utils.help_function import load_meta_properties


@dataclass
class ProcessingConfig:
    """Configuration for ISU data processing."""

    raw_data_rel_path: str = os.path.join("assets", "raw", "ISU")
    processed_rel_root: str = os.path.join("assets", "processed")
    chemistry: str = "NMC"
    sample_points: int = 100
    thread_count: int = 10
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
    """Container for ISU cell metadata."""

    initial_capacity: float
    c_rate_charge: float
    c_rate_discharge: float
    temperature: float
    dod: float


MIN_SEGMENT_SAMPLE_COUNT = 100


def load_cycling_json(file_path: str) -> Dict:
    """Load and decode the nested ISU JSON structure."""

    with open(file_path, "r") as fh:
        return json.loads(json.load(fh))


def build_cycle_segment(
    data_dict: Dict[str, List[List]], cycle_index: int
) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    """Build a single cycle dataframe for either charge or discharge."""

    try:
        raw_currents = data_dict["I"][cycle_index]
        raw_voltages = data_dict["V"][cycle_index]
        raw_times = data_dict["t"][cycle_index]
    except (KeyError, IndexError):
        return None, "missing data"

    if not raw_currents or not raw_voltages or not raw_times:
        return None, "empty segment"

    currents = pd.to_numeric(pd.Series(raw_currents), errors="coerce")
    voltages = pd.to_numeric(pd.Series(raw_voltages), errors="coerce")
    times_series = pd.Series(raw_times)

    numeric_time = pd.to_numeric(times_series, errors="coerce")
    if numeric_time.notna().all():
        time_values = numeric_time.to_numpy(dtype=float)
    else:
        datetime_values = pd.to_datetime(
            times_series, format="%Y-%m-%d %H:%M:%S", errors="coerce"
        )
        if datetime_values.isna().all():
            datetime_values = pd.to_datetime(times_series, errors="coerce")
        if datetime_values.isna().all():
            return None, "invalid timestamps"
        time_values = (
            datetime_values - datetime_values.iloc[0]
        ).dt.total_seconds().to_numpy(dtype=float)

    time_values = time_values - float(time_values[0])

    segment = pd.DataFrame(
        {
            "Test_Time(s)": time_values,
            "Voltage(V)": voltages.to_numpy(dtype=float),
            "Current(A)": currents.to_numpy(dtype=float),
        }
    )

    segment.replace([np.inf, -np.inf], np.nan, inplace=True)
    segment = segment.dropna()
    segment = segment[~segment["Test_Time(s)"].duplicated(keep="first")]
    segment = segment.sort_values("Test_Time(s)").reset_index(drop=True)

    if len(segment) < 5:
        return None, "segment too short"
    if segment["Test_Time(s)"].iloc[-1] <= 0:
        return None, "segment duration zero"

    return segment, None


def resample_cycle_segment(segment_df: pd.DataFrame, sample_points: int) -> pd.DataFrame:
    """Resample a cycle segment to a fixed number of samples."""

    required_cols = ["Test_Time(s)", "Voltage(V)", "Current(A)"]
    segment = segment_df[required_cols].dropna().drop_duplicates("Test_Time(s)")
    segment = segment.sort_values("Test_Time(s)")

    if segment.empty:
        return pd.DataFrame()

    time_values = segment["Test_Time(s)"].to_numpy(dtype=float)
    elapsed = time_values - time_values[0]

    result = pd.DataFrame(
        {
            "Sample_Index": np.arange(sample_points, dtype=int),
            "Normalized_Time": np.linspace(0.0, 1.0, sample_points),
        }
    )

    if len(segment) == 1 or np.isclose(elapsed[-1], 0.0):
        result["Elapsed_Time(s)"] = np.zeros(sample_points)
        for column in ["Voltage(V)", "Current(A)"]:
            result[column] = np.full(sample_points, segment[column].iloc[0])
        return result

    normalized_time = elapsed / elapsed[-1]
    target = result["Normalized_Time"].to_numpy()

    result["Elapsed_Time(s)"] = np.interp(target, normalized_time, elapsed)
    result["Voltage(V)"] = np.interp(
        target, normalized_time, segment["Voltage(V)"].to_numpy(dtype=float)
    )
    result["Current(A)"] = np.interp(
        target, normalized_time, segment["Current(A)"].to_numpy(dtype=float)
    )

    return result


def prepare_cycle_segment(
    segment: Optional[pd.DataFrame],
    min_samples: int,
) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    """Validate a raw segment and enforce minimum sample requirements."""

    if segment is None or segment.empty:
        return None, "missing segment data"

    required_cols = ["Test_Time(s)", "Voltage(V)", "Current(A)"]
    missing_cols = [col for col in required_cols if col not in segment.columns]
    if missing_cols:
        return None, f"missing columns: {', '.join(missing_cols)}"

    sanitized = segment.copy()
    for col in required_cols:
        sanitized[col] = pd.to_numeric(sanitized[col], errors="coerce")

    sanitized = sanitized.dropna()
    if sanitized.empty:
        return None, "segment contains no valid samples"

    sanitized = sanitized.drop_duplicates(subset=["Test_Time(s)"])
    sanitized = sanitized.sort_values("Test_Time(s)").reset_index(drop=True)

    if len(sanitized) < min_samples:
        return None, f"segment has fewer than {min_samples} samples"

    if sanitized["Test_Time(s)"].iloc[-1] <= sanitized["Test_Time(s)"].iloc[0]:
        return None, "segment duration is non-positive"

    return sanitized, None


def format_resampled_segment(
    resampled: pd.DataFrame,
    battery_id: str,
    chemistry: str,
    cycle_index: int,
    c_rate: float,
    temperature: float,
) -> pd.DataFrame:
    """Rename columns and attach metadata to a resampled segment."""

    resampled = resampled.rename(
        columns={
            "Sample_Index": "sample_index",
            "Normalized_Time": "normalized_time",
            "Elapsed_Time(s)": "elapsed_time_s",
            "Voltage(V)": "voltage_v",
            "Current(A)": "current_a",
        }
    )

    resampled["battery_id"] = battery_id
    resampled["chemistry"] = chemistry
    resampled["cycle_index"] = int(cycle_index)
    resampled["c_rate"] = float(c_rate)
    resampled["temperature_k"] = float(temperature)

    resampled = resampled[
        [
            "battery_id",
            "chemistry",
            "cycle_index",
            "sample_index",
            "normalized_time",
            "elapsed_time_s",
            "voltage_v",
            "current_a",
            "c_rate",
            "temperature_k",
        ]
    ]

    resampled["sample_index"] = resampled["sample_index"].astype(int)

    return resampled


def parse_cycles_to_resampled(
    cycling_dict: Dict,
    battery_id: str,
    cell_meta: CellMetadata,
    config: ProcessingConfig,
) -> Tuple[pd.DataFrame, pd.DataFrame, List[Dict[str, str]]]:
    """Convert raw ISU cycles into resampled charge and discharge tables."""

    columns = [
        "battery_id",
        "chemistry",
        "cycle_index",
        "sample_index",
        "normalized_time",
        "elapsed_time_s",
        "voltage_v",
        "current_a",
        "c_rate",
        "temperature_k",
    ]

    charge_segments: List[pd.DataFrame] = []
    discharge_segments: List[pd.DataFrame] = []
    errors: List[Dict[str, str]] = []

    charge_dict = cycling_dict.get("QV_charge", {})
    discharge_dict = cycling_dict.get("QV_discharge", {})

    num_charge_cycles = len(charge_dict.get("t", []))
    num_discharge_cycles = len(discharge_dict.get("t", []))

    requested_cycles = max(num_charge_cycles, num_discharge_cycles)
    max_cycles_allowed = min(requested_cycles, config.max_cycles)

    if requested_cycles > config.max_cycles:
        for skipped_cycle in range(config.max_cycles + 1, requested_cycles + 1):
            errors.append(
                {
                    "Cycle_Index": skipped_cycle,
                    "Direction": "both",
                    "Message": "skipped due to max cycle limit",
                }
            )

    min_samples_required = max(config.sample_points, MIN_SEGMENT_SAMPLE_COUNT)
    valid_cycle_counter = 0

    for idx in range(max_cycles_allowed):
        cycle_position = idx + 1

        charge_segment_sanitized: Optional[pd.DataFrame] = None
        discharge_segment_sanitized: Optional[pd.DataFrame] = None

        if idx < num_charge_cycles:
            charge_segment_raw, build_error = build_cycle_segment(charge_dict, idx)
            if build_error:
                errors.append(
                    {
                        "Cycle_Index": cycle_position,
                        "Direction": "charge",
                        "Message": build_error,
                    }
                )
            else:
                charge_segment_sanitized, sanitize_error = prepare_cycle_segment(
                    charge_segment_raw, min_samples_required
                )
                if sanitize_error:
                    errors.append(
                        {
                            "Cycle_Index": cycle_position,
                            "Direction": "charge",
                            "Message": sanitize_error,
                        }
                    )
        else:
            errors.append(
                {
                    "Cycle_Index": cycle_position,
                    "Direction": "charge",
                    "Message": "missing charge cycle",
                }
            )

        if idx < num_discharge_cycles:
            discharge_segment_raw, build_error = build_cycle_segment(discharge_dict, idx)
            if build_error:
                errors.append(
                    {
                        "Cycle_Index": cycle_position,
                        "Direction": "discharge",
                        "Message": build_error,
                    }
                )
            else:
                (
                    discharge_segment_sanitized,
                    sanitize_error,
                ) = prepare_cycle_segment(discharge_segment_raw, min_samples_required)
                if sanitize_error:
                    errors.append(
                        {
                            "Cycle_Index": cycle_position,
                            "Direction": "discharge",
                            "Message": sanitize_error,
                        }
                    )
        else:
            errors.append(
                {
                    "Cycle_Index": cycle_position,
                    "Direction": "discharge",
                    "Message": "missing discharge cycle",
                }
            )

        if charge_segment_sanitized is None or discharge_segment_sanitized is None:
            continue

        charge_resampled = resample_cycle_segment(
            charge_segment_sanitized, config.sample_points
        )
        discharge_resampled = resample_cycle_segment(
            discharge_segment_sanitized, config.sample_points
        )

        if charge_resampled.empty:
            errors.append(
                {
                    "Cycle_Index": cycle_position,
                    "Direction": "charge",
                    "Message": "resampling produced empty output",
                }
            )
            continue

        if discharge_resampled.empty:
            errors.append(
                {
                    "Cycle_Index": cycle_position,
                    "Direction": "discharge",
                    "Message": "resampling produced empty output",
                }
            )
            continue

        valid_cycle_counter += 1

        charge_formatted = format_resampled_segment(
            charge_resampled,
            battery_id,
            config.chemistry,
            valid_cycle_counter,
            cell_meta.c_rate_charge,
            cell_meta.temperature,
        )
        discharge_formatted = format_resampled_segment(
            discharge_resampled,
            battery_id,
            config.chemistry,
            valid_cycle_counter,
            cell_meta.c_rate_discharge,
            cell_meta.temperature,
        )

        charge_segments.append(charge_formatted)
        discharge_segments.append(discharge_formatted)

    charge_df = (
        pd.concat(charge_segments, ignore_index=True)
        if charge_segments
        else pd.DataFrame(columns=columns)
    )
    discharge_df = (
        pd.concat(discharge_segments, ignore_index=True)
        if discharge_segments
        else pd.DataFrame(columns=columns)
    )

    if not charge_df.empty:
        charge_df = charge_df.sort_values(["cycle_index", "sample_index"]).reset_index(
            drop=True
        )

    if not discharge_df.empty:
        discharge_df = discharge_df.sort_values(
            ["cycle_index", "sample_index"]
        ).reset_index(drop=True)

    return charge_df, discharge_df, errors


def get_cell_metadata(meta_df: pd.DataFrame, battery_id: str) -> Optional[CellMetadata]:
    """Retrieve metadata for a given battery ID."""

    cell_df = meta_df[meta_df["Battery_ID"].str.lower() == battery_id.lower()]
    if cell_df.empty:
        print(f"No metadata found for battery ID: {battery_id}")
        return None

    return CellMetadata(
        initial_capacity=cell_df["Initial_Capacity_Ah"].values[0],
        c_rate_charge=cell_df["C_rate_Charge"].values[0],
        c_rate_discharge=cell_df["C_rate_Discharge"].values[0],
        temperature=cell_df["Temperature (K)"].values[0],
        dod=cell_df.get("DoD", pd.Series([1.0])).values[0],
    )


def save_processed_data(
    charge_df: pd.DataFrame,
    discharge_df: pd.DataFrame,
    battery_id: str,
    config: ProcessingConfig,
    output_dir: str,
) -> Tuple[str, str]:
    """Persist charge and discharge aggregates to disk."""

    os.makedirs(output_dir, exist_ok=True)

    charge_path = os.path.join(output_dir, f"{battery_id}_charge_aggregated_data.csv")
    discharge_path = os.path.join(
        output_dir, f"{battery_id}_discharge_aggregated_data.csv"
    )

    charge_df.to_csv(charge_path, index=False)
    discharge_df.to_csv(discharge_path, index=False)

    print(f"💾 Saved charge CSV: {charge_path}")
    print(f"💾 Saved discharge CSV: {discharge_path}")

    return charge_path, discharge_path


def save_error_log(errors: List[Dict[str, str]], battery_id: str, output_dir: str) -> None:
    """Write per-battery parsing errors to disk."""

    if not errors:
        return

    os.makedirs(output_dir, exist_ok=True)
    error_df = pd.DataFrame(errors)
    error_log_path = os.path.join(output_dir, f"error_log_{battery_id}.csv")
    error_df.to_csv(error_log_path, index=False)
    print(f"📝 Saved error log: {error_log_path}")


def process_single_battery(
    file_name: str,
    raw_base_path: str,
    processed_base_path: str,
    meta_df: pd.DataFrame,
    config: ProcessingConfig,
) -> Dict[str, str]:
    """Process a single ISU JSON file end-to-end."""

    battery_id = os.path.splitext(file_name)[0]
    file_path = os.path.join(raw_base_path, file_name)
    print(f"\nProcessing battery: {battery_id}")

    try:
        cell_meta = get_cell_metadata(meta_df, battery_id)
        if cell_meta is None:
            return {battery_id: "metadata missing"}

        cycling_dict = load_cycling_json(file_path)

        charge_df, discharge_df, errors = parse_cycles_to_resampled(
            cycling_dict, battery_id, cell_meta, config
        )

        output_dir = os.path.join(processed_base_path, battery_id)
        save_processed_data(charge_df, discharge_df, battery_id, config, output_dir)
        save_error_log(errors, battery_id, output_dir)

        return {}

    except Exception as exc:  # pylint: disable=broad-except
        print(f"❌ Error processing {battery_id}: {exc}")
        return {battery_id: str(exc)}


def main(config: Optional[ProcessingConfig] = None) -> None:
    """Main entry point for ISU processing."""

    if config is None:
        config = ProcessingConfig()

    start_time = time.time()
    print(
        f"🚀 Starting ISU battery data processing with {config.thread_count} threads..."
    )

    meta_df = load_meta_properties()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.join(script_dir, "..", "..")
    raw_base_path = config.get_raw_data_path(project_root)
    processed_base_path = config.get_processed_dir(project_root)
    os.makedirs(processed_base_path, exist_ok=True)

    files = sorted(f for f in os.listdir(raw_base_path) if f.endswith(".json"))
    print(f"📂 Found {len(files)} ISU JSON files")

    aggregated_errors: Dict[str, str] = {}

    with ThreadPoolExecutor(max_workers=config.thread_count) as executor:
        future_to_file = {
            executor.submit(
                process_single_battery,
                file_name,
                raw_base_path,
                processed_base_path,
                meta_df,
                config,
            ): file_name
            for file_name in files
        }

        for future in as_completed(future_to_file):
            file_name = future_to_file[future]
            try:
                errors = future.result()
                aggregated_errors.update(errors)
                print(f"✅ Completed processing battery: {file_name}")
            except Exception as exc:  # pylint: disable=broad-except
                aggregated_errors[file_name] = str(exc)
                print(f"✗ Error processing battery {file_name}: {exc}")

    if aggregated_errors:
        error_log_path = os.path.join(processed_base_path, "error_log_isu.csv")
        pd.DataFrame(
            list(aggregated_errors.items()), columns=["Battery_ID", "Error_Message"]
        ).to_csv(error_log_path, index=False)
        print(f"📝 Saved global error log: {error_log_path}")

    total_time = time.time() - start_time
    hours = int(total_time // 3600)
    minutes = int((total_time % 3600) // 60)
    seconds = int(total_time % 60)

    print(f"\n{'=' * 60}")
    print("🎉 All ISU batteries processed successfully!")
    print(f"⏱️  Total processing time: {hours:02d}:{minutes:02d}:{seconds:02d}")
    print(f"📊 Processed {len(files)} batteries with {config.thread_count} threads")
    if files:
        print(f"⚡ Average time per battery: {total_time / len(files):.2f} seconds")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
# ============================================================
# 🎉 All ISU batteries processed successfully!
# ⏱️  Total processing time: 00:26:44
# 📊 Processed 251 batteries with 10 threads
# ⚡ Average time per battery: 6.39 seconds
# ============================================================