# Processed Data (Model-Ready)

This folder contains cleaned, transformed datasets used directly for modeling and analysis.

---

## Files

```text
data/processed/
├── crss_panel_2019_2023.csv
├── crss_modeling_severe.csv
```
---

## File Descriptions

### `crss_panel_2019_2023.csv`

A consolidated panel dataset spanning multiple years (2019–2023), including:

- Harmonized variable definitions across years
- Cleaned categorical codes
- Consistent feature naming
- One row per involved person
- Variables needed for exploratory analysis and modeling

---

### `crss_modeling_severe.csv`

Final modeling dataset used for machine learning.Key characteristics:

- Binary target variable indicating **severe injury** (incapacitating or fatal)
  - `1` → incapacitating or fatal injury  
  - `0` → non-severe injury  
- Cleaned and standardized columns

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
→ modeling notebook + Streamlit app


This structure ensures reproducibility, transparency, and separation of concerns.

