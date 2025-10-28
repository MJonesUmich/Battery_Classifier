# CS2 Cell Parser

Processes CS2 battery data files and generates cycle plots.

## Input Requirements

**File formats**: Excel (.xlsx) or Text (.txt)
**Required columns**: Current(A), Voltage(V), Test_Time(s)
**Naming**: `CS2_XX_MM_DD_YY.ext` (XX=battery_id, MM=month, DD=day, YY=year)

## Output Structure

### Data Files
- `processed_datasets/LCO/{battery_id}_aggregated_data.csv`
- `processed_datasets/LCO/error_log_{battery_id}.csv`

### Images
- `processed_images/LCO/{battery_id}/Cycle_{N}_{charge|discharge}_Crate_{rate}_tempK_{temp}_batteryID_{id}.png`

## Usage

```bash
python cs_cell_parser.py
```

**Performance**: ~34min for 10 batteries (20 threads)
