# Battery Classifier

Battery chemistry classification toolkit built with Python 3.11. The repo bundles:

- Data parsers that normalize vendor/partner test files into a common schema.
- Notebook experiments for tabular and image models (logistic regression, random forest, CNN).
- Utilities that export lightweight artifacts (demo CSVs + JSON model weights) for the React frontend.

You can explore the latest frontend build here: https://main.d1z5il29z8aho1.amplifyapp.com – it is the exact artifact produced by the steps described below, so stakeholders can see the deployed experience without running the project locally.

The codebase was optimized for fast iteration on local datasets and easy deployment to lightweight environments (browser, mobile, embedded).

---

## How to Run

All commands below are verified on Windows 10 PowerShell. Other operating systems have not been tested with this repo, so the steps may require adaptation.

### 1. Bootstrap the environment
```powershell
cd <your-clone-path>\Battery_Classifier
python -m venv .venv
.\.venv\Scripts\activate
pip install -U pip
pip install -r requirements.txt
```

### 2. Get the data assets
- **Required (raw data)**: Download `raw_20251207.zip` (~14.09 GB) from https://drive.google.com/file/d/1sHScf_HNTzuAurPBTFqm3j2pkNYALomt/view?usp=sharing and extract to `assets/raw/` (folders such as CS2, CX2, Dataset_1_NCA_battery, INR, ISU, MIT, Oxford, PL, Stanford, TU_Finland). Without this, parsers and notebooks cannot run.
- **Optional but recommended**: Download the processed/images bundle (~4.46 GB) from https://drive.google.com/file/d/1Sg6yYnOG9Xf_9XegGZ_khLr2wJUaHR38/view?usp=sharing and extract into `assets/` to populate `processed/`, `images/`, and `images_clipped/`. This skips >2 hours of preprocessing; otherwise, run `src/01_run_all_parsers.py` (and helpers) to rebuild processed data from raw.

### 3. Explore / retrain in notebooks
```powershell
python -m jupyter lab
```
Open Jupyter Lab and start from `03_Background.ipynb` once the data assets are in place. When processed data changes, re‑run the final cell in `05_Logistic_Regression_Exploration.ipynb` to refresh frontend artifacts (see step 4).

### Notebook execution order
Run preprocessing scripts first, then proceed with notebooks in order:
1. `src/01_run_all_parsers.py` – parse raw vendor data into the normalized processed CSVs.
2. `src/02_plot_processed_voltage_time.py` – generate processed voltage/time plots (optional sanity check).
3. `notebooks/03_Background.ipynb` – high-level context and sanity checks on the processed tables.
4. `notebooks/04_EDA.ipynb` – exploratory plots/statistics on the feature set.
5. `notebooks/05_Logistic_Regression_Exploration.ipynb` – main tabular chemistry classifier (plus frontend export cell).
6. `notebooks/06_PCA_Exploration.ipynb` – dimensionality reduction experiments.
7. `notebooks/07_Random_Forest_Classification.ipynb` – alternative tree-based baseline.
8. `notebooks/08_1D_CNN.ipynb` – sequence-based CNN modeling for charge/discharge traces.
9. `notebooks/09_image_based_classification.ipynb` – computer-vision pipeline using processed microscopy images.

### 4. Refresh frontend demo data + JS model
In `05_Logistic_Regression_Exploration.ipynb`, run the final `%run` helper cell. It calls `create_demo_datasets.py` and `export_logreg_to_json.py` to regenerate demo CSVs and the JS model bundle whenever processed assets change.

### 5. (Optional) Run the React demo
```powershell
cd frontend\battery-best
npm install
npm start
```

---

## Repository Highlights
- `src/model_training/image/` – image classification experiments for future extensions (CNN).
- `notebooks/` – step‑by‑step experimentation for multiple model families.
- `frontend/battery-best/` – React single-page app for demoing chemistry prediction directly in the browser.

Notes:
- Downsampling for the React demo is handled entirely by `create_demo_datasets.py`.
- LCO/NCA currently have fewer real samples; future data collection should prioritize those chemistries.
- MATLAB helper `export_pl_data.m` must be aimed at the latest raw source before re-running the PL parser.
- Plotting conventions now live alongside the code in `src/parser/plotting/02_plotJ_processed_voltage_time.py`, so reference that script for the latest Matplotlib defaults.