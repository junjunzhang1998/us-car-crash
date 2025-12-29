# Processed Data (Model-Ready)

This folder contains cleaned, transformed datasets used directly for modeling and analysis.

These files are derived from the intermediate merged data after:
- filtering
- feature selection
- harmonization across years
- creation of modeling targets
- basic preprocessing

---

## Files

data/processed/
├── crss_panel_2019_2023.csv
├── crss_modeling_severe.csv

---

## File Descriptions

### `crss_panel_2019_2023.csv`

A consolidated panel dataset spanning multiple years (2019–2023).

Includes:
- Harmonized variable definitions across years
- Cleaned categorical codes
- Consistent feature naming
- One row per involved person
- Variables needed for exploratory analysis and modeling

---

### `crss_modeling_severe.csv`

Final modeling dataset used for machine learning.

Key characteristics:
- Binary target variable indicating **severe injury** (incapacitating or fatal)
- Cleaned and standardized columns

---

## Target Variable

**Severe injury indicator**:
- `1` → incapacitating or fatal injury  
- `0` → non-severe injury  

This label is derived from CRSS injury severity codes.

---

## Notes

- All transformations are reproducible from scripts/notebooks in the project.
- Processed files are suitable for training, evaluation, and deployment.
- Any additional feature engineering or thresholding is applied at model level, not here.

---

## Data Flow Summary
raw/
→ intermediate/
→ processed/
→ modeling + Streamlit app


This structure ensures reproducibility, transparency, and separation of concerns.

