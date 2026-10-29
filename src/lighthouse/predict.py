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
    if hasattr(X_trans, "toarray"):
        X_trans = X_trans.toarray()
        
    coefs = model.coef_[0]
    # Calcul de la contribution : valeur de la feature * coefficient
    contributions = X_trans[0] * coefs
    feature_names = preprocessor.get_feature_names_out()
    
    # On sépare les features numériques (lisibles) des catégorielles (codées)
    num_mask = np.array([name.startswith("num__") for name in feature_names])
    
    # On trie les contributions de la plus forte à la plus faible
    sorted_indices = contributions.argsort()[::-1]
    
    # On priorise les 3 numériques les plus fortes (bien plus lisibles)
    num_indices = [i for i in sorted_indices if num_mask[i]][:top_n]
    cat_indices = [i for i in sorted_indices if not num_mask[i]]
    selected = num_indices + cat_indices[:top_n - len(num_indices)]
    
    signals = []
    for idx in selected[:top_n]:
        name = feature_names[idx]
        if name.startswith("num__"):
            # Affiche "Nom de la feature : valeur" (ex: "Curricular units 1st sem (approved): 0")
            base = name.replace("num__", "")
            value = X_single[base].iloc[0]
            signals.append(f"{base}: {value}")
        else:
            # Pour les catégorielles, enlève "cat__" et le code chiffré final
            base = name.replace("cat__", "").rsplit("_", 1)[0]
            signals.append(base)
    return signals

def predict_student(pipeline: Pipeline, student_data: dict, threshold: float) -> dict:
    """Generate the Student Card for a single student."""
    # On transforme le dictionnaire en DataFrame d'une seule ligne
    X_single = pd.DataFrame([student_data])
    
    # Prédiction de la probabilité
    prob = pipeline.predict_proba(X_single)[0, 1]
    is_risk = prob >= threshold
    signals = get_top_signals(pipeline, X_single)
    
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
    print(f"Risk Score  : {prob:.2%}")
    print(f"Status      : {status}")
    print(f"Main Signals: {', '.join(signals)}")
    print(f"Action      : {action}")
    print("--------------------")
    
    return card