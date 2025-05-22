# app.py  ──────────────────────────────────────────────────────────────
# Streamlit front-end for Diabetes Onset Prediction
# -------------------------------------------------
# • Loads a scikit-learn–style classifier (TabPFN, XGBoost, etc.).
# • Collects feature inputs from the user.
# • Returns probability + class prediction.
# • Optionally plots ROC curve if model + probs provided.

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

# ───────────────────────────────
# 1. Config & utility functions
# ───────────────────────────────
st.set_page_config(
    page_title="Diabetes Prediction",
    page_icon="🩺",
    layout="centered",
)

FEATURES = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
]

DEFAULTS = {
    "Pregnancies": 1,
    "Glucose": 120,
    "BloodPressure": 70,
    "SkinThickness": 20,
    "Insulin": 80,
    "BMI": 28.0,
    "DiabetesPedigreeFunction": 0.47,
    "Age": 33,
}

@st.cache_resource(show_spinner=False)
def load_model(pkl_path: str):
    """Load a pickled scikit-learn model."""
    if not Path(pkl_path).exists():
        st.error(f"Model file not found: {pkl_path}")
        st.stop()
    return joblib.load(pkl_path)

def predict(model, input_df: pd.DataFrame):
    """Return predicted class & probability for class 1."""
    proba = model.predict_proba(input_df)[:, 1]
    pred = (proba >= 0.5).astype(int)
    return pred[0], proba[0]

# ───────────────────────────────
# 2. Sidebar – model selection
# ───────────────────────────────
st.sidebar.header("⚙️ Model")
model_choice = st.sidebar.selectbox(
    "Choose model",
    [
        "TabPFN (Transformer)",
        "XGBoost",
        "Random Forest",
        "Logistic Regression",
    ],
)

MODEL_PATHS = {
    "TabPFN (Transformer)": "models/tabpfn_diabetes.pkl",
    "XGBoost": "models/xgb_diabetes.pkl",
    "Random Forest": "models/rf_diabetes.pkl",
    "Logistic Regression": "models/lr_diabetes.pkl",
}

model = load_model(MODEL_PATHS[model_choice])

st.sidebar.markdown("🛈 **Threshold** (probability ≥ threshold → class 1)")
threshold = st.sidebar.slider("Decision threshold", 0.05, 0.95, 0.50, 0.01)

# ───────────────────────────────
# 3. Main – feature inputs
# ───────────────────────────────
st.title("Diabetes Onset Prediction")
st.write(
    """
    Enter patient diagnostic values below.  
    Click **Predict** to see the risk score.
    """
)

cols = st.columns(2)
user_input = {}
for i, feat in enumerate(FEATURES):
    with cols[i % 2]:
        if feat in {"Pregnancies", "Age"}:
            user_input[feat] = st.number_input(
                feat, min_value=0, max_value=30, value=DEFAULTS[feat], step=1
            )
        elif feat in {"Glucose"}:
            user_input[feat] = st.number_input(
                feat, min_value=0, max_value=300, value=DEFAULTS[feat]
            )
        elif feat in {"BloodPressure"}:
            user_input[feat] = st.number_input(
                feat, min_value=0, max_value=200, value=DEFAULTS[feat]
            )
        elif feat in {"SkinThickness", "Insulin"}:
            user_input[feat] = st.number_input(
                feat, min_value=0, max_value=600, value=DEFAULTS[feat]
            )
        elif feat in {"BMI"}:
            user_input[feat] = st.number_input(
                feat, min_value=0.0, max_value=70.0, value=DEFAULTS[feat], step=0.1
            )
        else:  # DiabetesPedigreeFunction
            user_input[feat] = st.number_input(
                feat, min_value=0.0, max_value=3.0, value=DEFAULTS[feat], step=0.01
            )

input_df = pd.DataFrame([user_input])

# ───────────────────────────────
# 4. Prediction
# ───────────────────────────────
if st.button("🔮 Predict"):
    proba = model.predict_proba(input_df)[:, 1][0]
    pred = int(proba >= threshold)

    st.markdown(
        f"### Probability of Diabetes: **{proba:.3f}**\n"
        f"### Prediction (threshold {threshold:.2f}): "
        f"**{'Positive (1)' if pred else 'Negative (0)'}**"
    )

# ───────────────────────────────
# 5. Show ROC curve (if file exists)
# ───────────────────────────────
roc_path = Path("results/roc_curve.png")
if roc_path.exists():
    with st.expander("Model ROC Curve"):
        st.image(str(roc_path), caption="ROC curve on hold-out test set", use_column_width=True)

# ───────────────────────────────
# 6. Footer
# ───────────────────────────────
st.caption("Built with ❤️ by Ozodbek • Streamlit app ver. 1.0")
# End of file
