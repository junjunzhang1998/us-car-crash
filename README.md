# US Car Crash Injury Severity (CRSS)

This project analyzes U.S. crash data from the NHTSA Crash Report Sampling System (CRSS)
to predict driver injury severity and identify high-risk crash factors.

**Data:** NHTSA Crash Report Sampling System (2019–2023)  
**Unit of analysis:** Driver-level  
**Scope:** National-level crash injury severity analysis

### Overview
**Data Overview**
This project analyzes U.S. motor vehicle crash data from the NHTSA Crash Report Sampling System (CRSS) covering 2019–2023. CRSS provides nationally representative crash, vehicle, and person-level data. Raw data files are not included in this repository due to size and licensing constraints. 

**Data Cleaning & Preparation**
CRSS schemas vary across years. To ensure consistency and avoid schema drift, the data pipeline follows these principles:
- Yearly CRSS tables were merged into single annual datasets.
- A strict multi-year panel was constructed by retaining only variables that appear in all years (2019–2023). Variables introduced, renamed, or unused in certain years were excluded.
- The final dataset contains 468,311 observations and 316 consistently defined variables.

**Key Variables & Design Choices**
- Injury outcomes are measured using injury severity codes (INJSEV_IM). 
- Geographic identifiers are not consistently available across all years; analysis is therefore conducted at the national level.
- Derived variables (e.g., any injury, severe injury, speeding involvement) are created during analysis as needed.

### Data paths used in notebooks
All notebooks assume they are run from the `notebooks/` directory and define paths relative to the project root:
- Raw data: `data/raw/`
- Intermediate merged data: `data/intermediate/`
- Processed/model-ready data: `data/processed/`

Raw CRSS CSV files are intentionally excluded from version control via
`.gitignore`. To reproduce the analysis:

1. Download CRSS PERSON.csv, VEHICLE.csv, and ACCIDENT.csv files for years 2019–2023
2. Place them under:
   `data/raw/<year>/`
3. Run notebooks in numerical order from the `notebooks/` directory
