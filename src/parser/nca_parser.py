import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Ensure sibling imports work when executed as a script
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

# Reuse canonical helpers for splitting and resampling
from .cs_cell_parser import (  # type: ignore
    CellMetadata,
    prepare_cycle_segment,
    prepare_resampled_outputs,
    resample_cycle_segment,
    split_cycle_segments,
)


@dataclass
class ProcessingConfig:
	"""Configuration for Dataset_1_NCA_battery processing."""

	raw_data_rel_path: str = os.path.join("assets", "raw", "Dataset_1_NCA_battery")
	processed_rel_root: str = os.path.join("assets", "processed")
	chemistry: str = "NCA"
	sample_points: int = 100
	max_cycles: int = 100
	current_tolerance: float = 1e-4
	default_c_rate: float = 1.0
	default_temperature_k: float = 298.15

	def get_raw_data_path(self, project_root: str) -> str:
		return os.path.join(project_root, self.raw_data_rel_path)

	def get_processed_dir(self, project_root: str, battery_id: Optional[str] = None) -> str:
		base = os.path.join(project_root, self.processed_rel_root, self.chemistry)
		if battery_id:
			return os.path.join(base, battery_id)
		return base


def _extract_battery_id(file_name: str) -> str:
	"""
	Derive battery_id from file name by trimming trailing '-#<idx>'.
	Example: 'CY45-05_1-#12.csv' -> 'CY45-05_1'
	"""
	name, _ = os.path.splitext(os.path.basename(file_name))
	if "-#" in name:
		return name.split("-#")[0]
	return name


def _extract_sequence_index(file_name: str) -> int:
	"""
	Extract the numeric sequence after '#', defaulting to large number if not present.
	Used to sort files belonging to the same battery_id.
	"""
	name, _ = os.path.splitext(os.path.basename(file_name))
	if "#" in name:
		try:
			return int(name.split("#")[-1])
		except ValueError:
			return 10**9
	return 10**9


def load_nca_csv(file_path: str) -> pd.DataFrame:
	"""
	Load a single NCA CSV and normalize to required columns:
	- Test_Time(s): seconds
	- Voltage(V): volts (from 'Ecell/V')
	- Current(A): amps (from '<I>/mA' divided by 1000)
	- Cycle_Count: integer cycle number (from 'cycle number')
	"""
	df = pd.read_csv(file_path)
	rename_map = {
		"time/s": "Test_Time(s)",
		"Ecell/V": "Voltage(V)",
		"<I>/mA": "Current(A)",
		"cycle number": "Cycle_Count",
	}
	# Standardize column names if they include stray spaces/cases
	df.columns = [str(c).strip() for c in df.columns]
	df = df.rename(columns=rename_map)

	required = ["Test_Time(s)", "Voltage(V)", "Current(A)", "Cycle_Count"]
	missing = [c for c in required if c not in df.columns]
	if missing:
		raise ValueError(f"Missing required columns {missing} in {os.path.basename(file_path)}")

	# Coerce types and unit conversions
	df["Test_Time(s)"] = pd.to_numeric(df["Test_Time(s)"], errors="coerce")
	df["Voltage(V)"] = pd.to_numeric(df["Voltage(V)"], errors="coerce")
	df["Current(A)"] = pd.to_numeric(df["Current(A)"], errors="coerce") / 1000.0  # mA -> A
	df["Cycle_Count"] = pd.to_numeric(df["Cycle_Count"], errors="coerce").astype("Int64")

	df = df.dropna(subset=["Test_Time(s)", "Voltage(V)", "Current(A)", "Cycle_Count"]).reset_index(drop=True)

	# Ensure numeric dtypes
	df["Test_Time(s)"] = df["Test_Time(s)"].astype(float)
	df["Voltage(V)"] = df["Voltage(V)"].astype(float)
	df["Current(A)"] = df["Current(A)"].astype(float)
	df["Cycle_Count"] = df["Cycle_Count"].astype(int)
	return df[["Test_Time(s)", "Voltage(V)", "Current(A)", "Cycle_Count"]]


def aggregate_battery_files(folder_path: str, file_names: List[str]) -> Tuple[pd.DataFrame, Dict[str, str]]:
	"""
	Concatenate normalized frames from files of a single battery.
	Errors per file are collected and returned.
	"""
	errors: Dict[str, str] = {}
	agg = pd.DataFrame(columns=["Test_Time(s)", "Voltage(V)", "Current(A)", "Cycle_Count"])
	for fn in sorted(file_names, key=_extract_sequence_index):
		try:
			df = load_nca_csv(os.path.join(folder_path, fn))
			agg = pd.concat([agg, df], ignore_index=True)
		except Exception as e:
			errors[fn] = str(e)
	return agg, errors


def save_outputs(agg_df: pd.DataFrame, battery_id: str, config: ProcessingConfig, project_root: str) -> Tuple[str, str]:
	"""
	Prepare resampled outputs and write charge/discharge CSVs.
	"""
	cell_meta = CellMetadata(
		initial_capacity=1.0,  # unknown for this dataset; not used in resampling
		c_rate=config.default_c_rate,
		temperature=config.default_temperature_k,
		vmax=0.0,
		vmin=0.0,
	)

	charge_df, discharge_df = prepare_resampled_outputs(
		agg_df=agg_df,
		cell_meta=cell_meta,
		config=config,
		battery_id=battery_id,
	)

	out_dir = config.get_processed_dir(project_root, battery_id)
	os.makedirs(out_dir, exist_ok=True)
	charge_path = os.path.join(out_dir, f"{battery_id}_charge_aggregated_data.csv")
	discharge_path = os.path.join(out_dir, f"{battery_id}_discharge_aggregated_data.csv")
	charge_df.to_csv(charge_path, index=False)
	discharge_df.to_csv(discharge_path, index=False)
	return charge_path, discharge_path


def save_error_log(errors: Dict[str, str], battery_id: str, config: ProcessingConfig, project_root: str) -> None:
	if not errors:
		return
	out_dir = config.get_processed_dir(project_root, battery_id)
	os.makedirs(out_dir, exist_ok=True)
	err_df = pd.DataFrame(list(errors.items()), columns=["File_Name", "Error_Message"])
	err_path = os.path.join(out_dir, f"error_log_{battery_id}.csv")
	err_df.to_csv(err_path, index=False)


def main(config: Optional[ProcessingConfig] = None) -> None:
	"""
	Process Dataset_1_NCA_battery:
	- Group files by battery_id derived from name
	- Aggregate and sanitize data
	- Produce charge/discharge 100-point resampled CSVs per battery
	"""
	if config is None:
		config = ProcessingConfig()

	script_dir = os.path.dirname(os.path.abspath(__file__))
	project_root = os.path.join(script_dir, "..", "..")
	raw_dir = config.get_raw_data_path(project_root)
	if not os.path.isdir(raw_dir):
		raise FileNotFoundError(f"Raw data directory not found: {raw_dir}")

	# Group files by battery_id
	all_files = [f for f in os.listdir(raw_dir) if f.lower().endswith(".csv")]
	battery_to_files: Dict[str, List[str]] = {}
	for fn in all_files:
		bid = _extract_battery_id(fn)
		battery_to_files.setdefault(bid, []).append(fn)

	print(f"Found {len(battery_to_files)} batteries in {config.raw_data_rel_path}")

	for battery_id, files in battery_to_files.items():
		print(f"Processing battery {battery_id} with {len(files)} files")
		agg_df, errors = aggregate_battery_files(raw_dir, files)

		if agg_df.empty:
			print(f"Warning: no data aggregated for {battery_id}")
			save_error_log(errors, battery_id, config, project_root)
			continue

		try:
			charge_path, discharge_path = save_outputs(agg_df, battery_id, config, project_root)
			print(f"Saved: {charge_path}")
			print(f"Saved: {discharge_path}")
		except Exception as e:
			errors["__aggregation__"] = str(e)

		save_error_log(errors, battery_id, config, project_root)


if __name__ == "__main__":
	main()


