# app.py
# ---------------------------------------------------------
# HyperK-ML v0.1 – Educational Toy Model for Hyperkalemia
# Built on synthetic data. NOT for clinical decision support.
# ---------------------------------------------------------

import numpy as np
import pandas as pd
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score


# -----------------------------
# 1. Synthetic data + model
# -----------------------------
@st.cache_resource
def train_synthetic_model(n_samples: int = 2000):
    np.random.seed(42)

    # --- Generate synthetic features ---
    K = np.round(np.random.normal(loc=6.1, scale=0.4, size=n_samples), 1)
    K = np.clip(K, 5.5, 8.0)

    ecg_change = np.random.choice([0, 1, 2], size=n_samples, p=[0.5, 0.3, 0.2])
    hemodynamics = np.random.choice([0, 1, 2], size=n_samples, p=[0.6, 0.25, 0.15])
    symptoms = np.random.choice([0, 1, 2], size=n_samples, p=[0.4, 0.4, 0.2])
    kidney = np.random.choice([0, 1, 2], size=n_samples, p=[0.5, 0.3, 0.2])
    chronicity = np.random.choice([0, 1], size=n_samples, p=[0.6, 0.4])

    data = pd.DataFrame({
        "K": K,
        "ecg_change": ecg_change,
        "hemodynamics": hemodynamics,
        "symptoms": symptoms,
        "kidney": kidney,
        "chronicity": chronicity,
    })

    # --- Define hidden "true" risk model (log-odds) ---
    intercept = -6.0
    coef_K = 0.9
    coef_ecg = 0.8
    coef_hemo = 0.7
    coef_symptoms = 0.6
    coef_kidney = 0.4
    coef_chronicity = 0.5

    log_odds = (
        intercept
        + coef_K * (data["K"] - 6.0)
        + coef_ecg * data["ecg_change"]
        + coef_hemo * data["hemodynamics"]
        + coef_symptoms * data["symptoms"]
        + coef_kidney * data["kidney"]
        + coef_chronicity * data["chronicity"]
    )

    true_prob = 1 / (1 + np.exp(-log_odds))
    labels = np.random.binomial(1, true_prob, size=n_samples)

    data["need_emergent_shift"] = labels

    # --- Train logistic regression on synthetic data ---
    feature_cols = ["K", "ecg_change", "hemodynamics", "symptoms", "kidney", "chronicity"]
    X = data[feature_cols].values
    y = data["need_emergent_shift"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    y_pred_prob = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_pred_prob)

    return model, auc


def predict_risk(model, K, ecg_change, hemodynamics, symptoms, kidney, chronicity):
    X_new = np.array([[K, ecg_change, hemodynamics, symptoms, kidney, chronicity]])
    prob = model.predict_proba(X_new)[0, 1]

    if prob < 0.15:
        category = "Low"
    elif prob < 0.40:
        category = "Moderate"
    else:
        category = "High"

    return prob, category


# -----------------------------
# 2. Streamlit UI
# -----------------------------
def main():
    st.title("HyperK-ML v0.1 (Educational Toy Model)")
    st.write(
        """
        This app demonstrates how a simple machine learning model could estimate the
        likelihood that a patient with hyperkalemia needs **emergent potassium-shifting**.

        **Important:**  
        - Built entirely on synthetic data  
        - For education and demonstration only  
        - **Not** for clinical decision support or patient care
        """
    )

    # Train or load the synthetic model
    with st.spinner("Training synthetic model..."):
        model, auc = train_synthetic_model()
    st.success(f"Synthetic model trained. Test AUC ≈ {auc:.2f}")

    st.markdown("---")
    st.header("Enter Patient Features")

    # Potassium level
    K = st.number_input(
        "Potassium level (mmol/L)",
        min_value=5.0,
        max_value=8.5,
        value=6.2,
        step=0.1,
        format="%.1f",
    )

    # ECG changes
    ecg_map = {
        "None": 0,
        "Peaked T waves only": 1,
        "Significant changes (QRS widening / arrhythmia / sine wave)": 2,
    }
    ecg_choice = st.selectbox("ECG changes", list(ecg_map.keys()))
    ecg_encoded = ecg_map[ecg_choice]

    # Hemodynamics
    hemo_map = {
        "Stable (SBP ≥ 100, HR 50–110, no shock signs)": 0,
        "Borderline": 1,
        "Unstable (SBP < 90 or on pressors)": 2,
    }
    hemo_choice = st.selectbox("Hemodynamic status", list(hemo_map.keys()))
    hemo_encoded = hemo_map[hemo_choice]

    # Symptoms
    symp_map = {
        "None": 0,
        "Mild (weakness, palpitations)": 1,
        "Severe (syncope, chest pain, arrhythmia, arrest)": 2,
    }
    symp_choice = st.selectbox("Symptoms / events", list(symp_map.keys()))
    symp_encoded = symp_map[symp_choice]

    # Kidney function
    kidney_map = {
        "eGFR ≥ 30": 0,
        "eGFR < 30 (not on dialysis)": 1,
        "On dialysis": 2,
    }
    kidney_choice = st.selectbox("Kidney function / dialysis", list(kidney_map.keys()))
    kidney_encoded = kidney_map[kidney_choice]

    # Chronicity
    chronic_map = {
        "Likely chronic / baseline high": 0,
        "Likely acute / new rise": 1,
    }
    chronic_choice = st.selectbox("Chronicity of hyperkalemia", list(chronic_map.keys()))
    chronic_encoded = chronic_map[chronic_choice]

    st.markdown("---")

    if st.button("Estimate risk (educational only)"):
        prob, category = predict_risk(
            model,
            K=K,
            ecg_change=ecg_encoded,
            hemodynamics=hemo_encoded,
            symptoms=symp_encoded,
            kidney=kidney_encoded,
            chronicity=chronic_encoded,
        )

        st.subheader("Estimated Risk (Synthetic Model)")
        st.write(f"**Probability of needing emergent shifting:** {prob:.1%}")
        st.write(f"**Risk category:** {category}")

        st.info(
            "This estimate comes from a logistic regression model trained on synthetic "
            "data, meant only to illustrate how machine learning models can be built "
            "and used for clinical questions. It should **never** be used to guide real patient care."
        )


if __name__ == "__main__":
    main()
