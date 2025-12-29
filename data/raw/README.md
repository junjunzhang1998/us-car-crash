## Folder Structure

Each subfolder corresponds to a calendar year and contains three CSV files:

```text
data/raw/
├── 2019/
│ ├── accident.csv
│ ├── vehicle.csv
│ └── person.csv
├── 2020/
├── 2021/
├── 2022/
├── 2023/
```
---

## File Descriptions

### `accident.csv`
Crash-level information (one row per crash), including:

- Crash time and date
- Location and roadway characteristics
- Weather and lighting conditions
- Road relation variables
- Crash severity indicators

### `vehicle.csv`
Vehicle-level information (one row per vehicle involved in a crash), including:

- Vehicle body type
- Vehicle maneuver / travel direction
- Speed-related indicators
- Airbag deployment
- Vehicle damage and deformation indicators

### `person.csv`
Person-level information (one row per involved person), including:

- Age and sex
- Injury severity
- Restraint use and misuse
- Alcohol / drug involvement
- Seating position

---

## Notes

- Files are stored **exactly as provided by NHTSA**, without cleaning or transformation.
- Column codes follow the official CRSS documentation.
- A copy of the official CRSS data dictionary is included in this folder for reference.
- These raw files should never be edited directly.

---

## Source/Website Link

National Highway Traffic Safety Administration (NHTSA) 
Crash Report Sampling System (CRSS)  
https://www.nhtsa.gov/file-downloads?p=nhtsa/downloads/CRSS/

