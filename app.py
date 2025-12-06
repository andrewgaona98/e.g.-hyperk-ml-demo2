# app.py
# ---------------------------------------------------------
# HyperK-ML (Guideline-Trained Toy Model)
# Supervised learning on 48 guideline-based scenarios.
# Educational only – NOT for clinical decision support.
# ---------------------------------------------------------

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.linear_model import LogisticRegression


# ---------------------------------------------------------
# 1. Guideline rule: shift vs lower slowly
# ---------------------------------------------------------
def guideline_shift_decision(K_cat, symptoms, ekg, esrd, breakdown):
    """
    Implement the flowchart directly.

    K_cat: "5.0-5.5", "5.5-6.5", ">6.5"
    symptoms, ekg, esrd, breakdown: 0/1
    Returns: 1 = shift rapidly, 0 = lower slowly
    """

    # Pathway 1: clinical manifestations (symptoms or EKG changes)
    if symptoms == 1 or ekg == 1:
        return 1

    # Pathway 2: K > 6.5
    if K_cat == ">6.5":
        return 1

    # Pathway 3/4: K 5.5–6.5 with ESRD/oliguria
    if K_cat == "5.5-6.5" and esrd == 1:
        return 1

    # Otherwise: lower slowly
    return 0


# ---------------------------------------------------------
# 2. Build 48-patient dataset from the guideline
# ---------------------------------------------------------
def build_dataset():
    rows = []
    K_cats = ["5.0-5.5", "5.5-6.5", ">6.5"]
    id_counter = 1

    for K_cat in K_cats:
        for symptoms in [0, 1]:
            for ekg in [0, 1]:
                for esrd in [0, 1]:
                    for breakdown in [0, 1]:
                        shift = guideline_shift_decision(K_cat, symptoms, ekg, esrd, breakdown)
                        rows.append({
                            "ID": id_counter,
                            "K_cat": K_cat,
                            "Symptoms": symptoms,
                            "EKG_changes": ekg,
                            "ESRD_oliguria": esrd,
                            "Tissue_breakdown_bleed": breakdown,
                            "Shift_now": shift  # 1 = shift rapidly, 0 = lower slowly
                        })
                        id_counter += 1

    return pd.DataFrame(rows)


# ---------------------------------------------------------
# 3. Train logistic regression on that dataset (once)
# ---------------------------------------------------------
@st.cache_resource
def train_rule_model():
    df = build_dataset()

    # One-hot encode K_cat; 5.0–5.5 is reference
    X = pd.get_dummies(
        df[["K_cat", "Symptoms", "EKG_changes", "ESRD_oliguria", "Tissue_breakdown_bleed"]],
        columns=["K_cat"],
        drop_first=True
    )
    y = df["Shift_now"].values

    model = LogisticRegression(max_iter=1000, C=1e6)  # high C = very weak regularization
    model.fit(X, y)

    coeffs = pd.DataFrame({
        "feature": X.columns,
        "coefficient": model.coef_[0]
    })
    coeffs["odds_ratio"] = np.exp(coeffs["coefficient"])

    return model, X.columns.tolist(), coeffs


def model_probability(model, feature_cols, K_cat, symptoms, ekg, esrd, breakdown):
    """Return the model's probability of 'Shift_now = 1'."""
    row = pd.DataFrame([{
        "K_cat": K_cat,
        "Symptoms": symptoms,
        "EKG_changes": ekg,
        "ESRD_oliguria": esrd,
        "Tissue_breakdown_bleed": breakdown,
    }])

    row_enc = pd.get_dummies(row, columns=["K_cat"])
    row_enc = row_enc.reindex(columns=feature_cols, fill_value=0)

    prob = model.predict_proba(row_enc)[0, 1]
    return prob


# ---------------------------------------------------------
# 4. Streamlit UI
# ---------------------------------------------------------
def main():
    st.title("HyperK-ML (Guideline-Trained Toy Model)")
    st.write(
        """
        This app demonstrates **supervised learning** using a tiny dataset of
        48 synthetic patients built from a published hyperkalemia algorithm.

        For each combination of:
        - Potassium category  
        - Presence/absence of symptoms  
        - Presence/absence of EKG changes  
        - ESRD or oliguria  
        - Active tissue breakdown / bleed  

        we label the case as either:
        - **Shift now (rapid lowering)**, or  
        - **Lower slowly**,  

        according to the flowchart. A logistic regression model is then trained
        on these examples and learns **weights** for each factor.

        **Important:**  
        - This is an educational toy model only  
        - Not calibrated on real patient data  
        - Absolutely **not** for clinical decision support
        """
    )

    with st.spinner("Training logistic regression on guideline-based data..."):
        model, feature_cols, coeffs = train_rule_model()
    st.success("Model trained on 48 rule-based scenarios.")

    st.markdown("### Learned Weights (Logistic Regression Coefficients)")
    st.write(
        "Odds ratio > 1 means the feature pushes the model toward **shifting now**."
    )
    st.dataframe(coeffs.style.format({"coefficient": "{:.2f}", "odds_ratio": "{:.2f}"}))

    st.markdown("---")
    st.header("Try a Patient")

    # Potassium category
    K_choice = st.radio(
        "Serum potassium category",
        ["5.0-5.5", "5.5-6.5", ">6.5"],
        index=1
    )

    symp_choice = st.checkbox("Clinical manifestations (weakness, paralysis, arrhythmia symptoms)?")
    ekg_choice = st.checkbox("EKG changes concerning for hyperkalemia?")
    esrd_choice = st.checkbox("ESRD or oliguria?")
    breakdown_choice = st.checkbox("Active tissue breakdown / ongoing potassium load (eg, rhabdo, TLS, major bleed)?")

    if st.button("Estimate decision (educational only)"):
        symptoms = int(symp_choice)
        ekg = int(ekg_choice)
        esrd = int(esrd_choice)
        breakdown = int(breakdown_choice)

        # Guideline decision (authoritative)
        rule_decision = guideline_shift_decision(K_choice, symptoms, ekg, esrd, breakdown)

        # Model probability (what the ML model learned from those examples)
        prob = model_probability(
            model,
            feature_cols,
            K_cat=K_choice,
            symptoms=symptoms,
            ekg=ekg,
            esrd=esrd,
            breakdown=breakdown,
        )

        st.subheader("Guideline-Based Decision")
        if rule_decision == 1:
            st.success("**Guideline decision: SHIFT NOW (rapid lowering)**")
        else:
            st.info("**Guideline decision: LOWER SLOWLY**")

        st.subheader("Supervised Model View (Toy)")
        st.write(
            f"Estimated probability (from logistic regression) that the decision is "
            f"**'Shift now'**: {prob:.1%}"
        )
        st.caption(
            "The probability is derived from a logistic regression model trained on the "
            "48 guideline-based examples. It approximates the algorithm but does not "
            "replace it. Educational use only."
        )


if __name__ == "__main__":
    main()
