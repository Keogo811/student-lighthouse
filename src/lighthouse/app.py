"""Interactive dashboard for Student Lighthouse."""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve, auc
from sklearn.model_selection import cross_val_predict
from lighthouse.data import DatasetLoader
from lighthouse.config import load_config
from lighthouse.predict import get_top_signals

# --- CONFIGURATION DE LA PAGE ---
st.set_page_config(page_title="Student Lighthouse", page_icon="🕯️", layout="wide")

# --- CSS PERSONNALISÉ (Adapté au thème clair/sombre) ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Space Grotesk', sans-serif;
    }
    
    /* Effet de verre dépoli qui fonctionne sur fond clair ET sombre */
    .glass-card {
        background: rgba(128, 128, 128, 0.1);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid rgba(128, 128, 128, 0.2);
        border-radius: 20px;
        padding: 25px;
        margin: 10px 0;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.1);
    }
    
    .glass-card-high { border-left: 6px solid #ff4b4b; }
    .glass-card-low { border-left: 6px solid #00cc96; }
    
    .risk-title { margin-top: 0; font-size: 1.5rem; font-weight: 700; }
    .risk-score { font-size: 3.5rem; font-weight: 700; margin: 0; line-height: 1.1; }
    
    .signal-badge {
        display: inline-block;
        background: rgba(128, 128, 128, 0.15);
        border: 1px solid rgba(128, 128, 128, 0.3);
        padding: 8px 12px;
        border-radius: 12px;
        font-size: 0.9rem;
        margin-right: 8px;
        margin-bottom: 8px;
    }
</style>
""", unsafe_allow_html=True)

# --- CHARGEMENT DES DONNÉES ---
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

model = load_model()
X_train, X_test, y_train, y_test = load_data()

# --- EN-TÊTE ---
st.title("🕯️ Student Lighthouse")
st.markdown("*Guiding students before they drift away.*")
st.divider()

# ==========================================
# BARRE LATÉRALE (Contrôles)
# ==========================================
with st.sidebar:
    st.header("⚙️ Settings")
    
    st.markdown("**Service Capacity**")
    st.caption("The maximum number of students the success service can contact this semester.")
    capacity = st.slider("Number of students we can contact:", min_value=50, max_value=500, value=200, step=50)
    
    # Calcul dynamique du seuil
    probs_train = cross_val_predict(model, X_train, y_train, cv=5, method="predict_proba", n_jobs=-1)[:, 1]
    sorted_probs = sorted(probs_train, reverse=True)
    threshold = sorted_probs[capacity - 1]
    
    st.info(f"Current Threshold: **{threshold:.4f}**")
    
    st.markdown("---")
    st.markdown("**Student Profile**")
    scenario = st.radio("Choose a profile to simulate:", ["Low Risk Profile", "High Risk Profile"])
    file_name = "new_case.json" if scenario == "Low Risk Profile" else "high_risk_case.json"
    
    st.markdown("---")
    st.caption("Built for the ML × Python final project (aivancity, PGE 2).")

# ==========================================
# ONGLETS
# ==========================================
tab1, tab2, tab3 = st.tabs(["🎯 Triage Simulator", "📊 Model Lab", "📖 Methodology"])

# ==========================================
# ONGLET 1 : TRIAGE
# ==========================================
with tab1:
    st.header("How does this tool help?")
    st.markdown("""
    **Student Lighthouse is a decision-support tool.** It does not replace the human advisor. 
    Instead, it helps the university's student success service know **who to call first**. 
    
    Because the service has limited time and resources, they cannot call every student. 
    This tool ranks students by their risk of dropping out and provides a **Student Card** that explains 
    *why* a student might need help (e.g., missing credits, unpaid tuition). The advisor uses this 
    information to start a constructive conversation.
    """)
    
    st.divider()
    st.header("Student Card")
    
    # Chargement du profil
    with open(file_name, "r") as f:
        student_data = json.load(f)
        
    # Prédiction
    prob = model.predict_proba(pd.DataFrame([student_data]))[0, 1]
    is_risk = prob >= threshold
    signals = get_top_signals(model, pd.DataFrame([student_data]))
    
    # Rendu HTML de la carte
    card_class = "glass-card-high" if is_risk else "glass-card-low"
    status_text = "🚨 HIGH RISK" if is_risk else "✅ LOW RISK"
    action_text = "Contact this student this week." if is_risk else "No immediate action required."
    
    # Construction des badges HTML
    badges_html = "".join([f'<span class="signal-badge">🔹 {sig}</span>' for sig in signals])
    
    st.markdown(f"""
    <div class="glass-card {card_class}">
        <h2 class="risk-title">{status_text}</h2>
        <p class="risk-score">{prob:.1%}</p>
        <p style="color: #888; margin-top: 0;">Risk Score</p>
        <hr style="border-color: rgba(128,128,128,0.2);">
        <p><b>Action:</b> {action_text}</p>
        <p><b>Main Signals:</b></p>
        <div>{badges_html}</div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()
    st.subheader(f"📋 Triage List (Top {capacity} students to call)")
    st.markdown("Here is the actual list of students the advisor should contact, ranked by risk.")
    
    # On crée un DataFrame avec les prédictions sur le test set (qui simule une nouvelle cohorte)
    probs_test = model.predict_proba(X_test)[:, 1]
    
    # On crée un tableau temporaire
    triage_df = pd.DataFrame({
        "Student ID": X_test.index,
        "Risk Score": probs_test,
        "Actual Dropout": y_test
    })
    
    # On trie par risque décroissant et on garde le top (capacity)
    triage_df = triage_df.sort_values("Risk Score", ascending=False).head(capacity)
    
    # On formate la colonne Risk en pourcentage
    triage_df["Risk Score"] = triage_df["Risk Score"].apply(lambda x: f"{x:.1%}")
    
    # On affiche le tableau interactif
    st.dataframe(triage_df, use_container_width=True, hide_index=True)

# ==========================================
# ONGLET 2 : MODEL LAB
# ==========================================
with tab2:
    st.header("Model Performance & Interpretation")
    st.markdown("For the data scientists: statistical quality and global feature importance.")
    
    probs_test = model.predict_proba(X_test)[:, 1]
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("ROC Curve")
        fpr, tpr, _ = roc_curve(y_test, probs_test)
        roc_auc = auc(fpr, tpr)
        fig, ax = plt.subplots(figsize=(6, 4))
        fig.patch.set_alpha(0) # Fond transparent
        ax.patch.set_alpha(0)
        ax.plot(fpr, tpr, color="#ff4b4b", lw=3, label=f"AUC = {roc_auc:.3f}")
        ax.plot([0, 1], [0, 1], color="gray", lw=1, linestyle="--")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.legend(loc="lower right")
        st.pyplot(fig)
        
    with col2:
        st.subheader("Precision-Recall Curve")
        precision, recall, _ = precision_recall_curve(y_test, probs_test)
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        fig2.patch.set_alpha(0)
        ax2.patch.set_alpha(0)
        ax2.plot(recall, precision, color="#00cc96", lw=3)
        ax2.set_xlabel("Recall")
        ax2.set_ylabel("Precision")
        ax2.set_title("Average Precision (PR-AUC)")
        st.pyplot(fig2)
        
    st.divider()
    st.subheader("Top 10 Most Influential Features")
    st.markdown("Positive values increase dropout risk. Negative values decrease it.")
    
    preprocessor = model.named_steps["preprocessor"]
    classifier = model.named_steps["model"]
    feature_names = preprocessor.get_feature_names_out()
    
    clean_names = [name.replace("num__", "").replace("cat__", "") for name in feature_names]
    coef_df = pd.DataFrame({"Feature": clean_names, "Coefficient": classifier.coef_[0]})
    coef_df["Abs"] = coef_df["Coefficient"].abs()
    coef_df = coef_df.sort_values("Abs", ascending=False).head(10)
    
    st.bar_chart(data=coef_df, x="Feature", y="Coefficient", color="#ff4b4b")

# ==========================================
# ONGLET 3 : METHODOLOGY (La technique)
# ==========================================
with tab3:
    st.header("Methodology & Design Choices")
    
    st.markdown("""
    ### 1. The Problem
    The university's student success service can only contact a limited number of students per semester. The goal is to prioritize who to call first to maximize the impact of the advisor's time, while avoiding the stigmatization of students.
    
    ### 2. Temporal Leakage (The most important choice)
    The dataset contains 2nd-semester features. We strictly removed them. Why? Because at the moment we want to act (end of the 1st semester), these data do not exist yet. Keeping them would be cheating (temporal leakage) and would make the tool useless in real-world production.
    
    ### 3. The Metric
    We do not use Accuracy. A model that predicts "nobody drops out" is 68% accurate but catches zero at-risk students. We use **Average Precision (PR-AUC)** and **Precision@K**, which measure how well the model ranks the most at-risk students at the top of the list.
    
    ### 4. The Model
    We compared Logistic Regression, k-NN, and Linear SVM under the same cross-validation. Logistic Regression won (Average Precision: 0.8556) and was chosen because it is **interpretable**. We can read its coefficients to generate the "Student Card" and explain to the advisor *why* a student is flagged.
    
    ### 5. The Decision Threshold
    We do not use the default 0.5 threshold. Instead, we calculate a dynamic threshold based on the **Service Capacity**. For example, if the service can contact 200 students, the threshold is set exactly at the risk score of the 200th most at-risk student (0.9927).
    """)