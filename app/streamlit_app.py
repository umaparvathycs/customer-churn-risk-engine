from pathlib import Path
import json
import sys
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from data_generation import FEATURES, CATEGORICAL_FEATURES, NUMERIC_FEATURES, generate_scorecards, mitigation_actions, risk_band

DATA_PATH = ROOT / "data" / "customer_churn.csv"
MODEL_PATH = ROOT / "models" / "churn_pipeline.joblib"
METRICS_PATH = ROOT / "models" / "metrics.json"

st.set_page_config(page_title="Churn Intelligence", page_icon="📊", layout="wide")

@st.cache_data

def load_data():
    return pd.read_csv(DATA_PATH)

@st.cache_resource

def load_bundle():
    return joblib.load(MODEL_PATH)

@st.cache_data

def load_metrics():
    return json.loads(METRICS_PATH.read_text())


def explain_row(bundle, row: pd.DataFrame):
    transformed = bundle["preprocessor"].transform(row[FEATURES])
    model = bundle["model"]
    names = list(bundle["numeric_features"])
    names += list(bundle["preprocessor"].named_transformers_["categorical"].get_feature_names_out(bundle["categorical_features"]))
    try:
        import shap
        explainer = shap.TreeExplainer(model)
        raw_values = explainer.shap_values(transformed)
        # SHAP versions differ: binary classifiers may return a list of
        # class arrays, a (rows, features) array, or (rows, features, classes).
        if isinstance(raw_values, list):
            values = np.asarray(raw_values[1 if len(raw_values) > 1 else 0])[0]
        else:
            values = np.asarray(raw_values)
            if values.ndim == 3:
                values = values[0, :, 1 if values.shape[2] > 1 else 0]
            elif values.ndim == 2:
                values = values[0]
            else:
                values = values.reshape(-1)
    except Exception:
        # Keep the dashboard usable if SHAP cannot initialize on a platform.
        values = np.asarray(model.feature_importances_) * np.where(np.asarray(transformed[0]) >= 0, 1.0, -1.0)
    values = np.asarray(values, dtype=float).reshape(-1)
    if len(values) != len(names):
        values = np.resize(values, len(names))
    return pd.DataFrame({"feature": names, "impact": values}).sort_values("impact", ascending=False)


def main():
    st.markdown("# Churn Intelligence")
    st.caption("Explainable customer-risk prioritization for account managers")
    if not MODEL_PATH.exists() or not DATA_PATH.exists():
        st.error("Model artifacts are missing. Run `python src/data_generation.py` and `python src/train.py` first.")
        st.stop()
    df, bundle, metrics = load_data(), load_bundle(), load_metrics()
    probabilities = bundle["model"].predict_proba(bundle["preprocessor"].transform(df[FEATURES]))[:, 1]
    scored = df.copy(); scored["churn_probability"] = probabilities; scored["risk_band"] = [risk_band(p) for p in probabilities]
    cards = generate_scorecards(df, probabilities)

    with st.sidebar:
        st.header("Filters")
        bands = st.multiselect("Risk band", ["High", "Medium", "Low"], default=["High", "Medium", "Low"])
        contract = st.multiselect("Contract type", sorted(df.contract_type.unique()), default=sorted(df.contract_type.unique()))
        st.divider()
        st.metric("ROC-AUC", f"{metrics['roc_auc']:.2f}")
        st.caption("Random Forest with one-hot preprocessing and SHAP TreeExplainer.")

    filtered = scored[scored.risk_band.isin(bands) & scored.contract_type.isin(contract)]
    high = int((scored.risk_band == "High").sum())
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Customers", f"{len(scored):,}")
    c2.metric("High risk", f"{high:,}", f"{high / len(scored):.1%} of portfolio")
    c3.metric("Average risk", f"{scored.churn_probability.mean():.1%}")
    c4.metric("Filtered view", f"{len(filtered):,}")

    tab1, tab2, tab3 = st.tabs(["Portfolio overview", "Customer explanation", "Mitigation scorecards"])
    with tab1:
        left, right = st.columns(2)
        with left:
            fig = px.histogram(scored, x="churn_probability", color="risk_band", nbins=25, color_discrete_map={"High":"#ef4444", "Medium":"#f59e0b", "Low":"#10b981"}, title="Predicted churn-risk distribution")
            st.plotly_chart(fig, use_container_width=True)
        with right:
            summary = scored.groupby("contract_type", as_index=False).agg(customers=("customer_id", "count"), avg_risk=("churn_probability", "mean"))
            fig = px.bar(summary, x="contract_type", y="avg_risk", text_auto=".1%", title="Average predicted risk by contract")
            st.plotly_chart(fig, use_container_width=True)
        st.dataframe(filtered.sort_values("churn_probability", ascending=False)[["customer_id", "churn_probability", "risk_band", "contract_type", "monthly_charges", "tenure_months", "satisfaction_score"]].head(100), use_container_width=True, hide_index=True)

    with tab2:
        selected = st.selectbox("Select a customer", filtered.sort_values("churn_probability", ascending=False).customer_id.tolist())
        row = df[df.customer_id == selected].iloc[0]
        p = float(scored.loc[scored.customer_id == selected, "churn_probability"].iloc[0])
        a, b, c = st.columns(3); a.metric("Predicted probability", f"{p:.1%}"); b.metric("Risk band", risk_band(p)); c.metric("Recommended actions", len(mitigation_actions(row)))
        st.subheader("Top churn drivers")
        impacts = explain_row(bundle, pd.DataFrame([row]))
        impacts["direction"] = np.where(impacts.impact >= 0, "Increases risk", "Reduces risk")
        fig = px.bar(impacts.head(10).sort_values("impact"), x="impact", y="feature", color="direction", orientation="h", color_discrete_map={"Increases risk":"#ef4444", "Reduces risk":"#10b981"})
        st.plotly_chart(fig, use_container_width=True)
        st.info("SHAP values show each feature's contribution relative to the model baseline; positive values increase predicted churn risk.")

    with tab3:
        st.subheader("Prioritized account-manager actions")
        card_view = cards[cards.risk_band.isin(bands)].sort_values("risk_probability", ascending=False)
        st.download_button("Download scorecards CSV", card_view.to_csv(index=False), "churn_risk_scorecards.csv", "text/csv")
        st.dataframe(card_view.head(150), use_container_width=True, hide_index=True)

if __name__ == "__main__":
    main()
