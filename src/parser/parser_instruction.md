# Parser Refactor Instructions

## Background
- `README_cs_cell_parser.md` documents the legacy behaviour: each parser produced aggregated CSVs under `processed_datasets/LCO/` and generated cycle plot images under `processed_images/LCO/`.
- The new baseline described here supersedes the legacy standard. All parsers must follow the consolidated aggregation workflow and stop producing images during parsing.
- All raw input files have moved to `assets/raw/`; update loader paths accordingly.

## Parser Modules in Scope
Ensure the following modules live under `src/parser/` and comply with the new workflow:
- `cs_cell_parser.py`
- `cx_cell_parser.py`
- `inr_cell_parser.py`
- `isu_parser.py`
- `mit_parser.py`
- `oxford_cell_parser.py`
- `pl_cell_parser.py`
- `stanford_cell_parser.py`
- `TU_finland_cell_parser.py`

## Aggregation Requirements
- Aggregate every charge/discharge cycle to exactly 100 sample points.
  - If a cycle contains more than 100 points, downsample to 100.
  - If a cycle contains fewer than 100 points, interpolate up to 100.
- Store one aggregated timeseries for charge and one for discharge per cycle.
- Defer all image generation; remove any plotting, matplotlib, or file-writing logic for images from the parsers. (Image creation will occur in a separate step after aggregation.)

## Output Specification
- Each parser must emit CSV files only.
- Write data to: `assets/processed/{chemistry}/{battery_id}/`.
  - Save charge data as `{battery_id}_charge_aggregated_data.csv`.
  - Save discharge data as `{battery_id}_discharge_aggregated_data.csv`.
- Preserve or infer `{chemistry}` and `{battery_id}` consistently across parsers.
- Log errors per battery (if required) using the new directory structure.

## Migration Checklist
- [ ] Remove legacy image-generation code paths.
- [ ] Update file paths from the legacy `processed_datasets/` and `processed_images/` locations to the new `assets/processed/` hierarchy.
- [ ] Normalize aggregation logic across all parsers (shared utilities where practical).
- [ ] Verify split outputs: two CSVs per battery (charge/discharge) with 100-point cycles.
- [ ] Align README or module docstrings with this new behaviour once implementation is complete.