# Interim CSV Plotting Guide

## Overview
This guide explains how to generate State of Charge (SOC) vs Voltage plots from the interim CSV datasets stored in `assets\interim`. The plotting workflow is implemented in Python with `matplotlib` and leverages multiprocessing to speed up batch generation.

## Data Requirements
- Source files: `*_aggregated_data.csv` located within `assets\interim\{chemistry}\{battery_id}`.
- Required columns: `soc` (0–100 range), `voltage` (in volts), and `cycle` (integer per cycle). If column names differ, map them to these logical fields during ingestion.
- Ensure SOC values are sorted ascending before plotting; if not, sort within the script.

## Plotting Standards
- Use `matplotlib` for all figures.
- X-axis: SOC limited to 0–100 with fixed ticks every 10 units.
- Y-axis: Voltage constrained to 2.0–4.3 V with fixed ticks (default script uses 12 evenly spaced ticks).
- Global style defaults: white backgrounds, consistent fonts/sizes, 8x6 in figure, 100 DPI, single blue trace, grid enabled.
- Adopt any additional stylistic conventions not specified here from `src\parser\README_cs_cell_parser.md`.

## Parallel Processing
- Instantiate a `multiprocessing.Pool` in the main process. Each worker handles one CSV at a time.
- Recommended pool size: `min(os.cpu_count() - 1, 8)` to balance throughput and system load.
- Capture worker failures and log them to an error file following the existing convention (`processed_datasets/{chemistry}/error_log_{battery_id}.csv`).

## Output
- Save plots under `assets/images/{chemistry}/{battery_id}/Cycle_{N}_{charge|discharge}_Crate_{rate}_tempK_{temp}_batteryID_{id}.png` (matching the naming pattern from `README_cs_cell_parser.md`).
- Each aggregated CSV generates one PNG per `cycle` value; naming pads numeric cycles to three digits (e.g., `Cycle_003_...`).
- Close figures (`plt.close(fig)`) after saving to avoid memory leaks.

## Suggested Workflow
1. Discover all matching interim CSV files.
2. Submit file paths to the process pool.
3. Inside each worker:
   - Read the CSV with `pandas.read_csv`.
   - Validate/convert the SOC and voltage columns.
   - Sort by SOC if needed.
   - Create the plot (`fig, ax = plt.subplots(figsize=(10, 6))`).
   - Apply axis limits, titles, labels, and stylistic rules.
   - Save the figure to the designated path.
4. Aggregate success/failure results in the master process and print a summary.

## Example Command
```
python src/plotters/plot_interim_voltage_soc.py --chemistry LCO --workers 6
```

Update the command flags to target different chemistries or to process the full dataset. Document the script usage alongside this README so collaborators can reproduce the plots.

