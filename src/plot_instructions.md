# Processed Voltage Plotting Guide

This document captures the workflow for generating voltage-trace images from
the consolidated **processed** CSV outputs.

## Source Data
- Inputs live under `assets/processed/{chemistry}/{battery_id}/`.
- Each CSV follows the aggregated schema:
  - `battery_id`, `chemistry`, `cycle_index`, `sample_index`
  - `normalized_time`, `elapsed_time_s`, `voltage_v`, `current_a`
  - `c_rate`, `temperature_k`
- Voltage samples are pre-clipped to the shared safe window (3.0–3.6 V).

## Plotter Script
- Run `python src/plot_processed_voltage.py` (no arguments required).
- Defaults:
  - Input root: `assets/processed`.
  - Output root: `assets/images/processed_voltage`.
  - Worker count: `cpu_count()-1` (capped at 8).
  - Optional tweaks (chemistry filter, file limit, verbosity) can be set by editing the constants at the top of the script (`CHEMISTRY_FILTER`, `FILE_LIMIT`, `VERBOSE`), but the normal workflow needs zero flags.

## Rendering Rules
1. For every CSV and each `cycle_index`, draw one PNG.
2. X-axis: `normalized_time` (0→1). Y-axis: `voltage_v` rendered within 2.9–3.7 V to keep traces away from the frame borders even when the data is clipped at 3.0/3.6 V.
3. Background is dark to improve contrast for CV models; voltage trace color is cyan.
4. Output path: `assets/images/processed_voltage/{chemistry}/{battery_id}/{phase}/cycle_{cycle:03d}_{phase}_battery_{battery_id}.png`.
5. File naming encodes chemistry, battery ID, phase (charge/discharge), and 3-digit cycle index.

## Best Practices
- Run parsers first (via `python src/run_all_parsers.py`) to ensure processed CSVs exist.
- Use `--limit` for quick validation before bulk rendering.
- For large fleets, render chemistry-by-chemistry to simplify error triage.
- The voltage range is already clipped; avoid additional scaling unless a new production window is defined.
- Generated PNGs are deterministic (same inputs → same image), which is useful for caching in ML workflows.

