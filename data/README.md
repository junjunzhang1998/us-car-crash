Important:
---
The dataset itself is not committed to GitHub due to size and licensing considerations. This repository focuses on:
- modeling code
- preprocessing logic
- evaluation
- deployment via Streamlit

The files in this directory represent the original source data used in this project.

---

Reproducibility:
---
Running the notebooks in `/notebooks` will regenerate the processed dataset, assuming raw CRSS data is available locally.

---

## Data Directory Structure

```text
data/
├── raw/
│   ├── 2019/
│   ├── 2020/
│   ├── 2021/
│   ├── 2022/
│   ├── 2023/
│   └── README.md
│
├── intermediate/
│   ├── merged_2019.csv
│   ├── merged_2020.csv
│   ├── merged_2021.csv
│   ├── merged_2022.csv
│   ├── merged_2023.csv
│   └── README.md
│
└── processed/
    ├── crss_modeling_severe.csv
    ├── crss_panel_2019_2023.csv
    └── README.md
```
