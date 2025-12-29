# Severe Injury Risk Prediction (CRSS)

Predicts whether a traffic crash results in severe injury (incapacitating or fatal) using U.S. crash data from the NHTSA Crash Report Sampling System (CRSS).

**Live app**: https://us-car-crash-prediction.streamlit.app/  

**Model development notebook**:  
https://github.com/junjunzhang1998/us-car-crash/blob/main/notebooks/04_modeling.ipynb

---

## Overview

This project builds an end-to-end machine learning pipeline using U.S. crash data from 2019 - 2023.  
The goal is to model rare but high-impact severe injuries using interpretable and deployable models.

The workflow covers data preparation, modeling, evaluation, and deployment through a Streamlit web application.

---

## Data

Source: **NHTSA Crash Report Sampling System (CRSS)**

Raw, intermediate, and processed datasets are documented in the `data/` subfolders.  
Each folder contains a README describing its contents and preprocessing steps.

---

## Target

Binary outcome:

- `1` -> Severe injury (incapacitating or fatal)  
- `0` -> Non-severe injury  

This is a highly imbalanced classification problem.

---

## Models Tested

- Logistic Regression (baseline, interpretable)
- Random Forest
- Gradient Boosting (final deployed model)

Evaluation focuses on PR-AUC, which is more informative than accuracy for rare outcomes.

---

## Features Used

- Driver demographics  
- Vehicle type  
- Restraint and airbag indicators  
- Roadway relationship  
- Weather and time variables  

Categorical variables are encoded through preprocessing pipelines.

---

## Streamlit Application

The app allows users to:

- Input crash characteristics  
- Adjust the classification threshold  
- View predicted probability of severe injury  
- See HIGH / LOW risk classification  
- Inspect SHAP-based local explanations  

Live demo: https://us-car-crash-prediction.streamlit.app/

---

## Tech Stack

- Python 3.12  
- pandas, numpy  
- scikit-learn  
- matplotlib  
- SHAP  
- Streamlit  

---

## Author

**JJ Zhang**  
M.S. in Data Science, Columbia University  
GitHub: https://github.com/junjunzhang1998
