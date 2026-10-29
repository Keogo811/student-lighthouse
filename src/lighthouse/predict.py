"""Prediction logic and Student Card generation."""
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline

def get_top_signals(pipeline: Pipeline, X_single: pd.DataFrame, top_n: int = 3) -> list[str]:
    """Extract the top features that increased the risk for this student."""
    preprocessor = pipeline.named_steps["preprocessor"]
    model = pipeline.named_steps["model"]
    
    # On transforme l'étudiant avec le préprocesseur (scaling + encoding)
    X_trans = preprocessor.transform(X_single)
    
    # Si c'est une matrice creuse, on la convertit
    if hasattr(X_trans, "toarray"):
        X_trans = X_trans.toarray()
        
    coefs = model.coef_[0]
    # Calcul de la contribution : valeur de la feature * coefficient
    contributions = X_trans[0] * coefs
    
    # On récupère les noms de colonnes après transformation
    feature_names = preprocessor.get_feature_names_out()
    
    # On trie par contribution positive (qui augmente le risque)
    top_indices = contributions.argsort()[-top_n:][::-1]
    
    signals = []
    for idx in top_indices:
        name = feature_names[idx].replace("num__", "").replace("cat__", "")
        signals.append(name)
    return signals

def predict_student(pipeline: Pipeline, student_data: dict, threshold: float) -> dict:
    """Generate the Student Card for a single student."""
    # On transforme le dictionnaire en DataFrame d'une seule ligne
    X_single = pd.DataFrame([student_data])
    
    # Prédiction de la probabilité
    prob = pipeline.predict_proba(X_single)[0, 1]
    
    # Décision selon le seuil
    is_risk = prob >= threshold
    
    # On récupère les signaux
    signals = get_top_signals(pipeline, X_single)
    
    # On génère le texte de la carte
    if is_risk:
        status = "HIGH RISK (RED)"
        action = "Contact this student this week."
    else:
        status = "LOW RISK (GREEN)"
        action = "No immediate action required."
        
    card = {
        "probability": float(prob),
        "status": status,
        "action": action,
        "signals": signals
    }
    
    print("\n--- STUDENT CARD ---")
    print(f"Risk Score : {prob:.2%}")
    print(f"Status     : {status}")
    print(f"Main Signals: {', '.join(signals)}")
    print(f"Action     : {action}")
    print("--------------------")
    
    return card