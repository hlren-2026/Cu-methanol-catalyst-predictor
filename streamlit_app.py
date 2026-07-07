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
FEATURE_SHORT_LABELS = {
    "Al-loading(wt%)": "Al",
    "Al2O3-loading(wt%)": "Al2O3",
    "Ba-loading(wt%)": "Ba",
    "CNTs-loading(wt%)": "CNTs",
    "Ce-loading(wt%)": "Ce",
    "CeO2-loading(wt%)": "CeO2",
    "Cr-loading(wt%)": "Cr",
    "Cu-loading(wt%)": "Cu",
    "GNS(石墨烯)-loading(wt%)": "GNS",
    "GNS(鐭冲ⅷ鐑?-loading(wt%)": "GNS",
    "Ga-loading(wt%)": "Ga",
    "HT（水滑石）-loading(wt%)": "HT",
    "HT锛堟按婊戠煶锛?loading(wt%)": "HT",
    "In-loading(wt%)": "In",
    "K-loading(wt%)": "K",
    "La-loading(wt%)": "La",
    "Mg-loading(wt%)": "Mg",
    "MgO-loading(wt%)": "MgO",
    "Mn-loading(wt%)": "Mn",
    "Mo-loading(wt%)": "Mo",
    "Nd-loading(wt%)": "Nd",
    "Ni-loading(wt%)": "Ni",
    "Pd-loading(wt%)": "Pd",
    "Pr-loading(wt%)": "Pr",
    "Si-loading(wt%)": "Si",
    "SiO2-loading(wt%)": "SiO2",
    "TNTs-loading(wt%)": "TNTs",
    "Ti-loading(wt%)": "Ti",
    "TiO2-loading(wt%)": "TiO2",
    "W-loading(wt%)": "W",
    "Y-loading(wt%)": "Y",
    "Zn-loading(wt%)": "Zn",
    "ZnO-loading(wt%)": "ZnO",
    "Zr-loading(wt%)": "Zr",
    "ZrO2-loading(wt%)": "ZrO2",
    "om-SiO2-loading(wt%)": "om-SiO2",
    "Calcination Temperature": "T_cal",
    "Calcination duration": "t_cal",
    "drying Temperature": "T_dry",
    "drying during": "t_dry",
    "Activation Temperature": "T_act",
    "Activation duration": "t_act",
    "流速（ml/min)": "F_gas",
    "娴侀€燂紙ml/min)": "F_gas",
    "H2流速（ml/min)": "F_H2",
    "H2娴侀€燂紙ml/min)": "F_H2",
    "BET": "S_BET",
    "Pv": "V_p",
    "Ps": "D_p",
    "SCu": "S_Cu",
    "DCu": "D_Cu",
    "Cryst.size of CuO": "d_CuO",
    "Cryst.size of Cu": "d_Cu",
    "Temp/K": "T",
    "Pressure/Mpa": "P",
    "H2/CO2": "H/C",
    "GHSV(ml/(g*h))": "GHSV",
}

FEATURE_FORM_LABELS = {
    "Al-loading(wt%)": "Al loading (wt%)",
    "Al2O3-loading(wt%)": "Al2O3 loading (wt%)",
    "Ba-loading(wt%)": "Ba loading (wt%)",
    "CNTs-loading(wt%)": "CNTs loading (wt%)",
    "Ce-loading(wt%)": "Ce loading (wt%)",
    "CeO2-loading(wt%)": "CeO2 loading (wt%)",
    "Cr-loading(wt%)": "Cr loading (wt%)",
    "Cu-loading(wt%)": "Cu loading (wt%)",
    "GNS(石墨烯)-loading(wt%)": "GNS loading (wt%)",
    "GNS(鐭冲ⅷ鐑?-loading(wt%)": "GNS loading (wt%)",
    "Ga-loading(wt%)": "Ga loading (wt%)",
    "HT（水滑石）-loading(wt%)": "HT loading (wt%)",
    "HT锛堟按婊戠煶锛?loading(wt%)": "HT loading (wt%)",
    "In-loading(wt%)": "In loading (wt%)",
    "K-loading(wt%)": "K loading (wt%)",
    "La-loading(wt%)": "La loading (wt%)",
    "Mg-loading(wt%)": "Mg loading (wt%)",
    "MgO-loading(wt%)": "MgO loading (wt%)",
    "Mn-loading(wt%)": "Mn loading (wt%)",
    "Mo-loading(wt%)": "Mo loading (wt%)",
    "Nd-loading(wt%)": "Nd loading (wt%)",
    "Ni-loading(wt%)": "Ni loading (wt%)",
    "Pd-loading(wt%)": "Pd loading (wt%)",
    "Pr-loading(wt%)": "Pr loading (wt%)",
    "Si-loading(wt%)": "Si loading (wt%)",
    "SiO2-loading(wt%)": "SiO2 loading (wt%)",
    "TNTs-loading(wt%)": "TNTs loading (wt%)",
    "Ti-loading(wt%)": "Ti loading (wt%)",
    "TiO2-loading(wt%)": "TiO2 loading (wt%)",
    "W-loading(wt%)": "W loading (wt%)",
    "Y-loading(wt%)": "Y loading (wt%)",
    "Zn-loading(wt%)": "Zn loading (wt%)",
    "ZnO-loading(wt%)": "ZnO loading (wt%)",
    "Zr-loading(wt%)": "Zr loading (wt%)",
    "ZrO2-loading(wt%)": "ZrO2 loading (wt%)",
    "om-SiO2-loading(wt%)": "om-SiO2 loading (wt%)",
    "Calcination Temperature": "Calcination temperature",
    "Calcination duration": "Calcination duration",
    "drying Temperature": "Drying temperature",
    "drying during": "Drying duration",
    "Activation Temperature": "Activation temperature",
    "Activation duration": "Activation duration",
    "流速（ml/min)": "Flow rate (mL/min)",
    "娴侀€燂紙ml/min)": "Flow rate (mL/min)",
    "H2流速（ml/min)": "H2 flow rate (mL/min)",
    "H2娴侀€燂紙ml/min)": "H2 flow rate (mL/min)",
    "BET": "BET surface area",
    "Pv": "Pore volume",
    "Ps": "Pore size",
    "SCu": "Cu surface area",
    "DCu": "Cu dispersion",
    "Cryst.size of CuO": "Crystal size of CuO",
    "Cryst.size of Cu": "Crystal size of Cu",
    "Temp/K": "Reaction temperature (K)",
    "Pressure/Mpa": "Pressure (MPa)",
    "H2/CO2": "H2/CO2 ratio",
    "GHSV(ml/(g*h))": "GHSV (mL/(g·h))",
}


def get_short_label(feature_name: str) -> str:
    if feature_name in FEATURE_SHORT_LABELS:
        return FEATURE_SHORT_LABELS[feature_name]
    return feature_name


def get_form_label(feature_name: str) -> str:
    if feature_name in FEATURE_FORM_LABELS:
        return FEATURE_FORM_LABELS[feature_name]
    return feature_name


def build_display_schema_table(schema: dict[str, Any]) -> pd.DataFrame:
    loading_cols = set(schema["loading_cols"])
    return pd.DataFrame(
        {
            "Display name": [get_form_label(col) for col in schema["feature_cols"]],
            "Short label": [get_short_label(col) for col in schema["feature_cols"]],
            "Internal column name": schema["feature_cols"],
            "Type": ["Loading" if col in loading_cols else "Process / characterization" for col in schema["feature_cols"]],
        }
    )


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
    st.dataframe(build_display_schema_table(schema), use_container_width=True, height=420)

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

    with st.expander("Input field mapping for uploaded files"):
        st.write("Uploaded spreadsheets must use the internal training column names shown below.")
        st.dataframe(build_display_schema_table(schema), use_container_width=True, height=360)

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
                    get_form_label(feature),
                    value="" if default_value is None else str(default_value),
                    help=f"Internal column: {feature}. Loading columns default to 0; other fields may be left blank if unknown.",
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
    prepared_df = prepared_df.rename(columns={col: get_form_label(col) for col in prepared_df.columns})
    st.dataframe(prepared_df, use_container_width=True)

    st.subheader("Single-Sample Feature Impact (SHAP)")
    st.caption("The charts below show the top 10 SHAP contributions for the current sample.")

    for target_name in ["Conversion/%", "CH3OH Selectivity/%"]:
        explanation = prediction["explanations"][target_name]
        shap_df = pd.DataFrame(explanation["top_contributions"]).copy()
        shap_df["display_feature"] = shap_df["feature"].map(get_short_label)
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
                y=alt.Y("display_feature:N", sort=None, title="Feature"),
                color=alt.Color(
                    "direction:N",
                    scale=alt.Scale(domain=["Positive", "Negative"], range=["#d62728", "#1f77b4"]),
                    legend=alt.Legend(title="Impact"),
                ),
                tooltip=[
                    alt.Tooltip("display_feature:N", title="Feature"),
                    alt.Tooltip("feature:N", title="Internal column"),
                    alt.Tooltip("feature_value:Q", title="Feature value"),
                    alt.Tooltip("shap_value:Q", title="SHAP value", format=".4f"),
                ],
            )
            .properties(height=320)
        )
        st.altair_chart(chart, use_container_width=True)
        st.dataframe(
            shap_df[["display_feature", "feature", "feature_value", "shap_value", "abs_shap_value", "direction"]].rename(
                columns={"display_feature": "Feature", "feature": "Internal column"}
            ),
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
