"""Calculate metrics and baseline."""
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, f1_score

def get_baseline_scores(X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    """Train a dummy model and return its scores."""
    
    # Le DummyClassifier prédit toujours "Pas de décrochage" (0)
    dummy = DummyClassifier(strategy="most_frequent")
    dummy.fit(X_train, y_train)
    
    preds = dummy.predict(X_test)
    
    # On calcule l'accuracy et le F1
    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds, zero_division=0)
    
    print(f"--- BASELINE (Dummy) ---")
    print(f"Accuracy : {acc:.4f}")
    print(f"F1-score : {f1:.4f}")
    
    return {"accuracy": acc, "f1": f1}