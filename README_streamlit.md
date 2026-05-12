# Streamlit Prediction Platform

This project provides a lightweight **Streamlit-based visual prediction platform** for copper-based catalyst performance prediction in methanol synthesis from CO2 hydrogenation.

## Functions

- Batch prediction from Excel or CSV
- Single-sample manual prediction
- Downloadable prediction table
- Clear display of the fixed model configuration
- SHAP-based single-sample feature impact charts

## Fixed Models

- `Conversion/%`: `XGBoost`
- `CH3OH Selectivity/%`: `CatBoost`

The model files are loaded from:

- `artifacts/final_app_models/conversion_xgboost.json`
- `artifacts/final_app_models/ch3oh_catboost.cbm`
- `artifacts/final_app_models/feature_schema.json`
- `artifacts/final_app_models/model_metadata.json`

## Installation

```powershell
cd E:\codex\催化
pip install -r requirements-streamlit.txt
```

## Build Models

If the fixed model files are missing, rebuild them first:

```powershell
python build_final_models.py
```

## Run the App

```powershell
streamlit run streamlit_app.py
```

Or simply double-click:

- `run_streamlit_app.bat`

After startup, Streamlit will open a local browser page automatically or print a local URL such as:

- `http://localhost:8501`

## Input Rules

- Use the fixed feature schema stored in `feature_schema.json`
- All `loading(wt%)` columns are automatically filled with `0` if missing
- Missing non-loading columns are not allowed
- Extra columns are ignored during prediction

## Suggested Paper Description

> A lightweight Streamlit-based visual prediction platform was developed to facilitate user-friendly performance prediction for Cu-based methanol synthesis catalysts. The platform supports both batch prediction from spreadsheet files and interactive single-sample prediction, enabling experimental researchers to rapidly estimate CO2 conversion and CH3OH selectivity without writing code.

## Suggested Figures for the Paper

1. Home page screenshot
2. Batch prediction page with uploaded table and output preview
3. Single-sample prediction page with predicted values

## Template File

Use the provided template file:

- `sample_input_template.xlsx`

It contains all required input feature columns and one example row for editing.

## SHAP Explanation

The single-sample prediction page includes local SHAP-based feature impact charts for:

- `Conversion/%`
- `CH3OH Selectivity/%`

Only the top 10 feature contributions are displayed for cleaner paper-ready visualization.
