from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from catboost import CatBoostRegressor


MODEL_DIR = Path("artifacts") / "final_app_models"
SCHEMA_PATH = MODEL_DIR / "feature_schema.json"
METADATA_PATH = MODEL_DIR / "model_metadata.json"


def load_schema() -> dict[str, Any]:
    return json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))


def load_metadata() -> dict[str, Any]:
    return json.loads(METADATA_PATH.read_text(encoding="utf-8"))


def _prepare_features(df: pd.DataFrame, schema: dict[str, Any]) -> pd.DataFrame:
    feature_cols = schema["feature_cols"]
    loading_cols = set(schema["loading_cols"])

    missing_non_loading = [col for col in feature_cols if col not in df.columns and col not in loading_cols]
    if missing_non_loading:
        raise ValueError(f"缺少必需的非 loading 特征列: {missing_non_loading}")

    features = df.copy()
    for col in feature_cols:
        if col not in features.columns:
            features[col] = np.nan

    x = features[feature_cols].copy()
    for col in schema["loading_cols"]:
        if col in x.columns:
            x[col] = x[col].fillna(0)
    for col in feature_cols:
        x[col] = pd.to_numeric(x[col], errors="coerce")
    return x


def load_models() -> dict[str, Any]:
    metadata = load_metadata()
    models: dict[str, Any] = {}

    conversion_model = xgb.Booster()
    conversion_model.load_model(str(MODEL_DIR / metadata["targets"]["Conversion/%"]["model_file"]))
    models["Conversion/%"] = conversion_model

    ch3oh_model = CatBoostRegressor()
    ch3oh_model.load_model(str(MODEL_DIR / metadata["targets"]["CH3OH Selectivity/%"]["model_file"]))
    models["CH3OH Selectivity/%"] = ch3oh_model
    return models


def _predict_target(model: Any, features: pd.DataFrame) -> np.ndarray:
    if isinstance(model, xgb.Booster):
        return model.predict(xgb.DMatrix(features))
    return model.predict(features)


def _build_single_explanation(
    target: str,
    model: Any,
    sample_features: pd.DataFrame,
    predicted_value: float,
) -> dict[str, Any]:
    explainer = shap.TreeExplainer(model)
    explanation_input = xgb.DMatrix(sample_features) if isinstance(model, xgb.Booster) else sample_features
    explanation = explainer(explanation_input)

    if hasattr(explanation, "values"):
        values = np.array(explanation.values)
        base_values = np.array(explanation.base_values)
        shap_values = values[0]
        base_value = float(np.ravel(base_values)[0])
    else:
        shap_values = np.array(explainer.shap_values(explanation_input))[0]
        expected_value = explainer.expected_value
        base_value = float(np.ravel(np.array(expected_value))[0])

    contributions = []
    feature_values = sample_features.iloc[0].to_dict()
    for feature_name, feature_value, shap_value in zip(sample_features.columns, feature_values.values(), shap_values):
        contributions.append(
            {
                "feature": feature_name,
                "feature_value": None if pd.isna(feature_value) else float(feature_value),
                "shap_value": float(shap_value),
                "abs_shap_value": float(abs(shap_value)),
                "direction": "Positive" if shap_value >= 0 else "Negative",
            }
        )

    contributions.sort(key=lambda item: item["abs_shap_value"], reverse=True)

    return {
        "target": target,
        "predicted_value": float(predicted_value),
        "base_value": base_value,
        "top_contributions": contributions[:10],
        "all_contributions": contributions,
    }


def predict_batch(df: pd.DataFrame) -> pd.DataFrame:
    schema = load_schema()
    models = load_models()
    x = _prepare_features(df, schema)

    scored = df.copy()
    scored["Predicted Conversion/%"] = _predict_target(models["Conversion/%"], x)
    scored["Predicted CH3OH Selectivity/%"] = _predict_target(models["CH3OH Selectivity/%"], x)
    return scored


def predict_single(sample: dict[str, Any]) -> dict[str, Any]:
    schema = load_schema()
    metadata = load_metadata()
    models = load_models()

    input_df = pd.DataFrame([sample])
    x = _prepare_features(input_df, schema)

    conversion_pred = float(_predict_target(models["Conversion/%"], x)[0])
    ch3oh_pred = float(_predict_target(models["CH3OH Selectivity/%"], x)[0])

    explanations = {
        "Conversion/%": _build_single_explanation("Conversion/%", models["Conversion/%"], x, conversion_pred),
        "CH3OH Selectivity/%": _build_single_explanation(
            "CH3OH Selectivity/%", models["CH3OH Selectivity/%"], x, ch3oh_pred
        ),
    }

    return {
        "predictions": {
            "Conversion/%": conversion_pred,
            "CH3OH Selectivity/%": ch3oh_pred,
        },
        "explanations": explanations,
        "metadata": metadata["targets"],
        "prepared_features": x.iloc[0].to_dict(),
    }
