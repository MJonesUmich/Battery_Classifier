# Assets Directory Setup

## Raw Data

- Download the raw data package `raw_20251207.zip`: https://drive.google.com/file/d/1sHScf_HNTzuAurPBTFqm3j2pkNYALomt/view?usp=sharing (about 14.09 GB)
- Extract it to `assets/, resulting in:

```
assets/raw/
├── CS2
├── CX2
├── Dataset_1_NCA_battery
├── INR
├── ISU
├── MIT
├── Oxford
├── PL
├── Stanford
└── TU_Finland
```

## processed/ and images/ and images_clipped/

- Download the archive: https://drive.google.com/file/d/1Sg6yYnOG9Xf_9XegGZ_khLr2wJUaHR38/view?usp=sharing (about 4.46 GB)
- Extract it into `assets/` so the structure looks like:

```
assets/
├── images/           # processed image tiles (full set)
├── images_clipped/   # clipped image tiles
├── processed/        # processed datasets (per chemistry/battery)
├── raw/              # raw source datasets (per vendor/partner)

```