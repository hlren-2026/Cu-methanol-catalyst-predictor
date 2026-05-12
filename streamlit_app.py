from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Any

import altair as alt
import pandas as pd
import streamlit as st

from app_predictor_lite import load_metadata, load_schema, predict_batch, predict_single


APP_TITLE = "Cu-Based Catalyst Methanol Performance Predictor"
MODEL_DIR = Path("artifacts") / "final_app_models"


def build_template_dataframe(schema: dict[str, Any]) -> pd.DataFrame:
    sample_row = {col: 0 if col in schema["loading_cols"] else None for col in schema["feature_cols"]}
    return pd.DataFrame([sample_row])


def to_excel_bytes(df: pd.DataFrame) -> bytes:
    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="predictions")
    buffer.seek(0)
    return buffer.getvalue()


def validate_uploaded_columns(df: pd.DataFrame, schema: dict[str, Any]) -> tuple[list[str], list[str]]:
    feature_cols = schema["feature_cols"]
    loading_cols = set(schema["loading_cols"])
    missing_loading = [col for col in feature_cols if col in loading_cols and col not in df.columns]
    missing_non_loading = [col for col in feature_cols if col not in loading_cols and col not in df.columns]
    return missing_loading, missing_non_loading


def render_home_page(schema: dict[str, Any], metadata: dict[str, Any]) -> None:
    st.title(APP_TITLE)
    st.caption("A lightweight Streamlit-based visual prediction platform for paper presentation and laboratory use.")

    st.markdown(
        """
        This platform predicts two catalyst performance indicators:

        - **CO2 Conversion / Conversion(%)**
        - **Methanol Selectivity / CH3OH Selectivity(%)**

        The current version is designed for **local use**, **batch screening**, and **single-sample prediction**.
        """
    )

    st.subheader("Workflow")
    st.markdown(
        """
        1. Prepare catalyst composition and reaction-condition features.
        2. Upload an Excel/CSV file or enter one sample manually.
        3. The platform fills missing `loading(wt%)` features with `0`.
        4. The trained models return predicted conversion and selectivity values.
        """
    )

    st.subheader("Current Fixed Models")
    left, right = st.columns(2)
    with left:
        st.markdown("**Conversion/% model**")
        st.json(metadata["targets"]["Conversion/%"])
    with right:
        st.markdown("**CH3OH Selectivity/% model**")
        st.json(metadata["targets"]["CH3OH Selectivity/%"])

    st.subheader("Input Feature Schema")
    st.write(f"Total feature columns: **{len(schema['feature_cols'])}**")
    st.dataframe(pd.DataFrame({"Feature Column": schema["feature_cols"]}), use_container_width=True, height=420)

    st.info(
        "Missing-value policy: all loading(wt%) columns are automatically filled with 0; "
        "other numeric/process features remain missing and are handled by the models directly."
    )
    st.warning("This tool is intended for model-assisted prediction and does not replace experimental validation.")


def render_batch_page(schema: dict[str, Any]) -> None:
    st.title("Batch Prediction")
    st.write("Upload an Excel or CSV file, preview the input, run prediction, and download the result table.")

    uploaded = st.file_uploader("Upload input file", type=["xlsx", "csv"])
    if uploaded is None:
        st.info("Please upload an `.xlsx` or `.csv` file.")
        return

    suffix = Path(uploaded.name).suffix.lower()
    try:
        if suffix == ".csv":
            input_df = pd.read_csv(uploaded)
        else:
            input_df = pd.read_excel(uploaded)
    except Exception as exc:
        st.error(f"Failed to read the file: {exc}")
        return

    st.subheader("Input Preview")
    st.dataframe(input_df.head(10), use_container_width=True)

    missing_loading, missing_non_loading = validate_uploaded_columns(input_df, schema)
    status_col, info_col = st.columns([1, 2])
    with status_col:
        st.metric("Rows", len(input_df))
    with info_col:
        st.write(f"Detected columns: **{len(input_df.columns)}**")

    if missing_loading:
        st.info(f"Missing loading columns will be auto-filled with 0: {missing_loading}")
    if missing_non_loading:
        st.error(f"Missing required non-loading columns: {missing_non_loading}")
        return

    if st.button("Run Batch Prediction", type="primary"):
        try:
            result_df = predict_batch(input_df)
        except Exception as exc:
            st.error(f"Prediction failed: {exc}")
            return

        st.success("Prediction completed successfully.")
        st.subheader("Prediction Result")
        st.dataframe(result_df, use_container_width=True)

        csv_bytes = result_df.to_csv(index=False).encode("utf-8-sig")
        excel_bytes = to_excel_bytes(result_df)

        download_col1, download_col2 = st.columns(2)
        with download_col1:
            st.download_button(
                "Download CSV",
                data=csv_bytes,
                file_name="predictions.csv",
                mime="text/csv",
            )
        with download_col2:
            st.download_button(
                "Download Excel",
                data=excel_bytes,
                file_name="predictions.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )

        st.caption("Output columns include `Predicted Conversion/%` and `Predicted CH3OH Selectivity/%`.")


def render_single_page(schema: dict[str, Any], metadata: dict[str, Any]) -> None:
    st.title("Single Sample Prediction")
    st.write("Enter one sample manually and predict both target properties.")

    with st.form("single_sample_form"):
        form_values: dict[str, Any] = {}
        columns = st.columns(3)

        for idx, feature in enumerate(schema["feature_cols"]):
            target_col = columns[idx % 3]
            default_value = 0.0 if feature in schema["loading_cols"] else None
            with target_col:
                form_values[feature] = st.text_input(
                    feature,
                    value="" if default_value is None else str(default_value),
                    help="Loading columns default to 0; other fields may be left blank if unknown.",
                )

        submitted = st.form_submit_button("Run Single Prediction", type="primary")

    if not submitted:
        return

    normalized_sample: dict[str, Any] = {}
    for feature, raw_value in form_values.items():
        if raw_value is None or str(raw_value).strip() == "":
            normalized_sample[feature] = 0 if feature in schema["loading_cols"] else None
            continue
        try:
            normalized_sample[feature] = float(raw_value)
        except ValueError:
            st.error(f"Feature `{feature}` must be numeric.")
            return

    try:
        prediction = predict_single(normalized_sample)
    except Exception as exc:
        st.error(f"Prediction failed: {exc}")
        return

    preds = prediction["predictions"]
    st.success("Prediction completed successfully.")

    result_col1, result_col2 = st.columns(2)
    with result_col1:
        st.metric("Predicted CO2 Conversion (%)", f"{preds['Conversion/%']:.4f}")
    with result_col2:
        st.metric("Predicted CH3OH Selectivity (%)", f"{preds['CH3OH Selectivity/%']:.4f}")

    st.subheader("Model Information")
    st.json(metadata["targets"])

    st.subheader("Prepared Input Summary")
    prepared_df = pd.DataFrame([prediction["prepared_features"]])
    st.dataframe(prepared_df, use_container_width=True)

    st.subheader("Single-Sample Feature Impact (SHAP)")
    st.caption("The charts below show the top 10 SHAP contributions for the current sample.")

    for target_name in ["Conversion/%", "CH3OH Selectivity/%"]:
        explanation = prediction["explanations"][target_name]
        shap_df = pd.DataFrame(explanation["top_contributions"]).copy()
        shap_df = shap_df.sort_values("shap_value", ascending=True)

        st.markdown(f"**{target_name}**")
        info_col1, info_col2 = st.columns(2)
        with info_col1:
            st.metric("Base value", f"{explanation['base_value']:.4f}")
        with info_col2:
            st.metric("Predicted value", f"{explanation['predicted_value']:.4f}")

        chart = (
            alt.Chart(shap_df)
            .mark_bar()
            .encode(
                x=alt.X("shap_value:Q", title="SHAP value"),
                y=alt.Y("feature:N", sort=None, title="Feature"),
                color=alt.Color(
                    "direction:N",
                    scale=alt.Scale(domain=["Positive", "Negative"], range=["#d62728", "#1f77b4"]),
                    legend=alt.Legend(title="Impact"),
                ),
                tooltip=[
                    alt.Tooltip("feature:N", title="Feature"),
                    alt.Tooltip("feature_value:Q", title="Feature value"),
                    alt.Tooltip("shap_value:Q", title="SHAP value", format=".4f"),
                ],
            )
            .properties(height=320)
        )
        st.altair_chart(chart, use_container_width=True)
        st.dataframe(
            shap_df[["feature", "feature_value", "shap_value", "abs_shap_value", "direction"]],
            use_container_width=True,
            hide_index=True,
        )


def main() -> None:
    st.set_page_config(
        page_title=APP_TITLE,
        page_icon="🧪",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    if not MODEL_DIR.exists():
        st.error("Model directory `artifacts/final_app_models` was not found. Please run `python build_final_models.py` first.")
        st.stop()

    schema = load_schema()
    metadata = load_metadata()

    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Select a page",
        ["Home", "Batch Prediction", "Single Sample Prediction"],
    )

    st.sidebar.markdown("---")
    st.sidebar.caption("For paper presentation and laboratory screening.")

    if page == "Home":
        render_home_page(schema, metadata)
    elif page == "Batch Prediction":
        render_batch_page(schema)
    else:
        render_single_page(schema, metadata)


if __name__ == "__main__":
    main()
