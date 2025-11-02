# Parser Refactor Instructions

## Background
- `README_cs_cell_parser.md` captures the legacy behaviour: parsers wrote aggregates under `processed_datasets/<chemistry>/` and rendered cycle plots into `processed_images/<chemistry>/`.
- The new baseline **replaces** that behaviour. All parsers must adopt the consolidated aggregation workflow and stop creating images during parsing.
- Raw inputs now live under `assets/raw/`; adjust loader defaults so each parser discovers its own chemistry subfolder.
- Refer to `cs_cell_parser.py` for the canonical implementation (100-point resampling, charge/discharge splits, new directory layout).

## Parser Modules in Scope
The following modules under `src/parser/` must conform to this spec:
- `cs_cell_parser.py`
- `cx_cell_parser.py`
- `inr_cell_parser.py`
- `isu_parser.py`
- `mit_parser.py`
- `oxford_cell_parser.py`
- `pl_cell_parser.py`
- `stanford_cell_parser.py`
- `TU_finland_cell_parser.py`

## Input Handling Requirements
- Loaders should accept Excel `.xlsx` and the legacy tab-delimited `.txt` exports.
  - For `.txt` sources, rename columns to `Test_Time(s)`, `Current(A)`, `Voltage(V)`.
  - Convert units immediately: `mV → V`, `mA → A`, and interpret `Time` as **minutes** (multiply by 60 to store in seconds).
- Ensure all downstream logic operates on seconds (`Test_Time(s)` in SI units) and floats.
- Inject parser-specific metadata (chemistry, battery ID, temperature, etc.) via a configuration object to avoid hard-coded paths.

## Aggregation Requirements
1. Detect charge/discharge cycles and tag each row with `Cycle_Count`.
2. Clip the dataframe to complete cycles (drop warm-up/cleanup noise outside the charge↔discharge alternation window).
3. For every cycle:
   - Split into charge and discharge segments using current sign (positive = charge, negative = discharge) with a small tolerance (`1e-4 A`) to filter jitter.
   - Resample each segment to **exactly 100 points**:
     - If the raw segment has more than 100 samples, downsample evenly.
     - If it has fewer, interpolate to reach 100 samples.
   - Preserve monotonic `Test_Time(s)` → `elapsed_time_s` so the last resampled row reflects the real cycle duration.
4. Attach metadata columns to every row in the aggregated output:
   - `battery_id`
   - `chemistry`
   - `cycle_index`
   - `sample_index`
   - `normalized_time`
   - `elapsed_time_s`
   - `voltage_v`
   - `current_a`
   - `c_rate`
   - `temperature_k`
5. Do **not** generate plots or images inside the parser.

## Output Specification
- Emit CSV files only; one per mode:
  - `{battery_id}_charge_aggregated_data.csv`
  - `{battery_id}_discharge_aggregated_data.csv`
- Output directory: `assets/processed/{chemistry}/{battery_id}/`.
  - Create the directory if missing.
  - Ensure file names are lowercase/consistent with legacy naming where applicable.
- If failures occur while parsing individual source files, write an error log to the same folder: `error_log_{battery_id}.csv`.
- Verify that each CSV contains:
  - Continuous `cycle_index` starting at 1.
  - `sample_index` spanning `0..99` for every cycle.
  - `elapsed_time_s` in seconds (0 at start, increasing to the segment duration).

## Testing & Validation
- Unit-test the resampling helper(s) to cover:
  - Segments with >100 rows (downsampling) and <100 rows (interpolation).
  - Sign-change detection at low currents.
  - Minute-to-second conversion for `.txt` imports.
- Add smoke/integration tests that run a parser against a known small fixture and assert:
  - Output location and filename structure.
  - Column schemas match the specification above.
  - First/last `elapsed_time_s` align with the underlying raw durations.
- Where large real datasets exist (e.g. CS2), run a spot-check script to confirm durations and sample counts, mirroring the checks in `cs_cell_parser.py`.

## Migration Checklist
- [ ] Remove legacy image-generation code, matplotlib imports, and any file writes under `processed_images/`.
- [ ] Update base paths to use `assets/raw/{chemistry}/` for inputs and `assets/processed/{chemistry}/{battery_id}/` for outputs.
- [ ] Share resampling and splitting utilities across parsers where practical (e.g. move stable helpers into a common module).
- [ ] Guarantee each parser emits **two** CSVs per battery (charge & discharge) with 100-point cycles.
- [ ] Refresh README/docstrings to describe the new workflow (no on-parser images, new directories, resampling rules).
- [ ] Re-run parsers on sample datasets and audit the generated CSVs for duration/unit correctness (e.g. verify multi-hour cycles no longer appear as ~100 s).
- [ ] Capture any deviations or edge cases (missing negative currents, single-step cycles, etc.) and document fallback behaviour.