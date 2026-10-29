"""Calculate metrics and baseline."""
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, f1_score, average_precision_score, roc_auc_score, precision_score, confusion_matrix
from sklearn.pipeline import Pipeline

def get_baseline_scores(X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    """Train a dummy model and return its scores."""
    dummy = DummyClassifier(strategy="most_frequent")
    dummy.fit(X_train, y_train)
    preds = dummy.predict(X_test)
    
    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds, zero_division=0)
    print(f"--- BASELINE (Dummy) ---")
    print(f"Accuracy : {acc:.4f} | F1-score : {f1:.4f}")
    return {"accuracy": acc, "f1": f1}

def evaluate_final_model(pipeline: Pipeline, X_test: pd.DataFrame, y_test: pd.Series, threshold: float, capacity: int = 200) -> dict:
    """Evaluate the final pipeline on the test set using the chosen threshold."""
    # 1. Prédictions de probabilités
    probs = pipeline.predict_proba(X_test)[:, 1]
    
    # 2. Prédictions binaires basées sur le seuil
    preds = (probs >= threshold).astype(int)
    
    # 3. Calcul des métriques globales
    ap = average_precision_score(y_test, probs)
    roc_auc = roc_auc_score(y_test, probs)
    
    # 4. Calcul des métriques au seuil (le fameux top 200)
    # Attention: le test set fait 885 lignes. 200/885 = 22.6% de capacité sur le test.
    # On va donc mesurer la précision sur les X% les plus à risque.
    test_capacity = int(len(X_test) * (capacity / 3539)) # On garde la même proportion
    top_k_indices = probs.argsort()[-test_capacity:][::-1]
    top_k_true = y_test.iloc[top_k_indices]
    
    precision_at_k = top_k_true.mean()
    
    # 5. Matrice de confusion
    cm = confusion_matrix(y_test, preds)
    
    print("\n--- FINAL MODEL EVALUATION (TEST SET) ---")
    print(f"Average Precision (PR-AUC) : {ap:.4f}")
    print(f"ROC-AUC                    : {roc_auc:.4f}")
    print(f"Precision @ Top {test_capacity}     : {precision_at_k:.4f}")
    print(f"Confusion Matrix           :\n{cm}")
    
    return {
        "average_precision": ap,
        "roc_auc": roc_auc,
        "precision_at_k": precision_at_k,
        "confusion_matrix": cm
    }