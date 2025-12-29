# Intermediate Data

This folder contains **merged and lightly processed datasets** created from the raw CRSS files.  
These datasets combine information across crash, vehicle, and person levels to form a single table per year.

They are intended as an intermediate step between raw ingestion and final modeling datasets.

---

## Files
data/intermediate/
├── merged_2019.csv
├── merged_2020.csv
├── merged_2021.csv
├── merged_2022.csv
├── merged_2023.csv

---

## How These Files Are Created

Each `merged_YYYY.csv` file is created by:

1. Loading:
   - `accident.csv`
   - `vehicle.csv`
   - `person.csv`

2. Joining tables using CRSS identifiers:
   - `ST_CASE`
   - `VEH_NO`
   - `PER_NO`

3. Keeping one row per **person–vehicle–crash combination**

4. Selecting relevant variables related to:
   - Crash context (time, weather, roadway)
   - Vehicle characteristics
   - Occupant attributes
   - Injury severity
   - Safety equipment usage


They are **not yet fully cleaned or encoded** for machine learning.

---

## Notes

- Missing values may still be present
- Categorical variables remain coded numerically
- No modeling assumptions are applied at this stage
- Files may be large due to retained detail

These files are later consolidated and transformed into modeling-ready datasets stored in `data/processed/`.

