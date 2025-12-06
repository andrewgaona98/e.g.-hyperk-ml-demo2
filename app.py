# app.py
# ---------------------------------------------------------
# HyperK-ML (Rule-Trained Toy Model)
# Supervised learning on 48 guideline-based scenarios.
# Educational only – NOT for clinical decision support.
# ---------------------------------------------------------

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.linear_model import LogisticRegression


# ---------------------------------------------------------
# 1. Build the 48-patient dataset based on the flowchart
# ---------------------------------------------------------
def build_dataset():
    rows = []
    K_cats = ["5.0-5.5", "5.5-6.5", ">6.5"]
    id_counter = 1

    for K_cat in K_cats:
        for symptoms in [0, 1]:        # 0 = No, 1 = Yes
            for ekg in [0, 1]:         # 0 = No, 1 = Yes
                for esrd in [0, 1]:    # 0 = No, 1 = Yes
                    for breakdown in [0, 1]:  # 0 = No, 1 = Yes

                        # ------------------------------
                        # Apply the guideline algorithm
                        # ------------------------------
                        shift = False

                        # Pathway 1: clinical manifestations (symptoms or EKG)
                        if symptoms == 1 or ekg == 1:
                            shift = True

                        # Pathway 2: K > 6.5
                        if K_cat == ">6.5":
                            shift = True

                        # Pathway 3/4: K 5.5–6.5 with ESRD/oliguria
                        if K_cat == "5.5-6.5" and esrd == 1:
                            shift = True

                        rows.append({
                            "ID": id_counter,
                            "K_cat": K_cat,
                            "Symptoms": symptoms,
                            "EKG_changes": ekg,
                            "ESRD_oliguria": esrd,
                            "Tissue_breakdown_bleed": breakdown,
                            "Shift_now": 1 if shift else 0  # 1 = shift rapidly, 0 = lower slowly
                        })

                        id_counter += 1

    return pd.DataFrame(rows)


# ---------------------------------------------------------
# 2. Train logistic regression on that dataset (once)
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

    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)

    coeffs = pd.DataFrame({
        "feature": X.columns,
        "coefficient": model.coef_[0]
    })
    coeffs["odds_ratio"] = np.exp(coeffs["coefficient"])

    return model, X.columns.tolist(), coeffs


def predict_patient(model, feature_cols, K_cat, symptoms, ekg, esrd, breakdown):
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
    pred = model.predict(row_enc)[0]  # 1 = shift, 0 = slow

    return prob, int(pred)


# ---------------------------------------------------------
# 3. Streamlit UI
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
        prob, pred = predict_patient(
            model,
            feature_cols,
            K_cat=K_choice,
            symptoms=int(symp_choice),
            ekg=int(ekg_choice),
            esrd=int(esrd_choice),
            breakdown=int(breakdown_choice),
        )

        st.subheader("Model Output (Toy)")
        st.write(f"**Estimated probability that guideline-consistent decision is to SHIFT now:** {prob:.1%}")

        if pred == 1:
            st.success("**Model decision: SHIFT NOW (rapid lowering)**")
        else:
            st.info("**Model decision: LOWER SLOWLY**")

        st.caption(
            "This mirrors what the model inferred from the 48 rule-based examples, "
            "not real-world outcomes. Educational use only."
        )


if __name__ == "__main__":
    main()
