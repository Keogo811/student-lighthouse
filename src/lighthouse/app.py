"""Streamlit application for Student Lighthouse."""
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import roc_curve, precision_recall_curve, auc
from lighthouse.data import DatasetLoader
from lighthouse.config import load_config

# Configuration de la page
st.set_page_config(page_title="Student Lighthouse", page_icon="🕯️", layout="wide")

# On charge le modèle et les données une seule fois (pour que ce soit rapide)
@st.cache_resource
def load_model():
    config = load_config()
    model_path = Path(config["paths"]["models_dir"]) / "lighthouse_model.joblib"
    return joblib.load(model_path)

@st.cache_data
def load_data():
    loader = DatasetLoader()
    df = loader.load_and_clean()
    return loader.split(df)

# On récupère les éléments
model = load_model()
X_train, X_test, y_train, y_test = load_data()

# --- EN-TÊTE ---
st.title("🕯️ Student Lighthouse")
st.markdown("*Guiding students before they drift away.*")

# On crée les 3 onglets
tab1, tab2, tab3 = st.tabs(["🎯 Triage", "📊 Model Lab", "📖 Method"])

# ==========================================
# ONGLET 1 : TRIAGE (La carte étudiant)
# ==========================================
with tab1:
    st.header("Student Card Simulation")
    st.markdown("Select a scenario to see the model's decision and the main signals.")
    
    # Menu déroulant pour choisir un scénario
    scenario = st.radio("Choose a student profile:", ["Low Risk Student", "High Risk Student"])
    
    # On charge le bon fichier JSON
    if scenario == "Low Risk Student":
        with open("new_case.json", "r") as f:
            student_data = json.load(f)
    else:
        with open("high_risk_case.json", "r") as f:
            student_data = json.load(f)
            
    # On fait la prédiction
    probs = model.predict_proba(pd.DataFrame([student_data]))[0, 1]
    threshold = 0.9927
    
    # Affichage de la carte
    col1, col2 = st.columns(2)
    with col1:
        st.metric(label="Risk Score", value=f"{probs:.2%}")
        if probs >= threshold:
            st.error("**Status: HIGH RISK**\n\nAction: Contact this student this week.")
        else:
            st.success("**Status: LOW RISK**\n\nAction: No immediate action required.")
            
    with col2:
        st.info("**Main Signals**")
        # On récupère les signaux depuis predict.py (on simplifie l'appel)
        from lighthouse.predict import get_top_signals
        signals = get_top_signals(model, pd.DataFrame([student_data]))
        for s in signals:
            st.write(f"- {s}")

    st.divider()
    st.markdown("**Why this matters:** The model does not just say yes/no. It tells the advisor *why* this student needs help, making the decision actionable and fair.")

# ==========================================
# ONGLET 2 : MODEL LAB (Pour le prof)
# ==========================================
with tab2:
    st.header("Model Performance & Interpretation")
    st.markdown("This tab is for the data scientists. It shows the model's quality and how it makes decisions.")
    
    # Courbes ROC et PR
    col1, col2 = st.columns(2)
    
    probs_test = model.predict_proba(X_test)[:, 1]
    
    with col1:
        st.subheader("ROC Curve")
        fpr, tpr, _ = roc_curve(y_test, probs_test)
        roc_auc = auc(fpr, tpr)
        fig, ax = plt.subplots()
        ax.plot(fpr, tpr, color="darkorange", lw=2, label=f"AUC = {roc_auc:.3f}")
        ax.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.legend(loc="lower right")
        st.pyplot(fig)
        
    with col2:
        st.subheader("Precision-Recall Curve")
        precision, recall, _ = precision_recall_curve(y_test, probs_test)
        fig2, ax2 = plt.subplots()
        ax2.plot(recall, precision, color="green", lw=2)
        ax2.set_xlabel("Recall")
        ax2.set_ylabel("Precision")
        ax2.set_title("Average Precision (PR-AUC)")
        st.pyplot(fig2)
        
    st.divider()
    
    # Coefficients du modèle
    st.subheader("Top 10 Most Influential Features")
    st.markdown("Positive values increase dropout risk. Negative values decrease it.")
    
    preprocessor = model.named_steps["preprocessor"]
    classifier = model.named_steps["model"]
    feature_names = preprocessor.get_feature_names_out()
    
    # On nettoie les noms pour l'affichage
    clean_names = [name.replace("num__", "").replace("cat__", "") for name in feature_names]
    
    # On crée un DataFrame pour le graphique
    coef_df = pd.DataFrame({
        "Feature": clean_names,
        "Coefficient": classifier.coef_[0]
    })
    # On trie par valeur absolue (les plus influents en premier)
    coef_df["Abs"] = coef_df["Coefficient"].abs()
    coef_df = coef_df.sort_values("Abs", ascending=False).head(10)
    
    st.bar_chart(data=coef_df, x="Feature", y="Coefficient", color="Feature")

# ==========================================
# ONGLET 3 : METHOD (Explications)
# ==========================================
with tab3:
    st.header("Methodology & Design Choices")
    
    st.markdown("""
    ### 1. The Problem
    This tool supports the **Student Success Service**. It does not automate decisions. It prioritizes which students to call first, given a limited capacity of **200 appointments per semester**.
    
    ### 2. Data & Leakage
    We used the UCI dataset *Predict Students' Dropout and Academic Success*.
    **Crucial choice:** We removed all columns from the **2nd semester**. Why? Because at the moment we want to act (end of 1st semester), these data do not exist yet. Using them would be cheating (leakage) and make the tool useless in production.
    
    ### 3. The Metric
    We do not use Accuracy. A model that says "nobody drops out" is 68% accurate but catches zero students.
    We use **Average Precision (PR-AUC)**, which measures how well we rank the most at-risk students. We also look at **Precision @ Top 200**: if we call 200 students, how many are truly at risk?
    
    ### 4. The Model
    We compared Logistic Regression, k-NN, and Linear SVM under the same cross-validation. Logistic Regression won, and we chose it because it is **interpretable**: we can read the coefficients to show the advisor *why* a student is flagged.
    """)