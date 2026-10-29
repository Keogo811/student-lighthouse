"""Calculate metrics, baseline, and handle predictions."""
import pandas as pd
import json
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, f1_score, average_precision_score, roc_auc_score, confusion_matrix
from sklearn.pipeline import Pipeline
from lighthouse.predict import predict_student

def get_baseline_scores(X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    """Train a dummy model and return its scores."""
    # Le modèle "dummy" prédit toujours la classe majoritaire (Pas de décrochage)
    dummy = DummyClassifier(strategy="most_frequent")
    dummy.fit(X_train, y_train)
    preds = dummy.predict(X_test)
    
    # On calcule l'accuracy et le F1 pour avoir une référence
    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds, zero_division=0)
    return {"accuracy": acc, "f1": f1}

def evaluate_final_model(pipeline: Pipeline, X_test: pd.DataFrame, y_test: pd.Series, threshold: float, capacity: int = 200) -> dict:
    """Evaluate the final pipeline on the test set using the chosen threshold."""
    # 1. On récupère les probabilités prédites pour la classe 1 (décrocheur)
    probs = pipeline.predict_proba(X_test)[:, 1]
    
    # 2. On transforme en 0 ou 1 en utilisant le seuil calculé à l'Arc 8
    preds = (probs >= threshold).astype(int)
    
    # 3. Calcul des métriques globales (qui ne dépendent pas du seuil)
    ap = average_precision_score(y_test, probs)
    roc_auc = roc_auc_score(y_test, probs)
    
    # 4. Calcul de la précision sur le Top K (les 200 places du service)
    # On calcule la proportion de 200 par rapport au train set (3539)
    test_capacity = int(len(X_test) * (capacity / 3539))
    # On trie les indices des probabilités les plus hautes
    top_k_indices = probs.argsort()[-test_capacity:][::-1]
    top_k_true = y_test.iloc[top_k_indices]
    precision_at_k = top_k_true.mean()
    
    # 5. Matrice de confusion pour voir les vrais/faux positifs au seuil
    cm = confusion_matrix(y_test, preds)
    
    print("\n--- FINAL MODEL EVALUATION ---")
    print(f"Average Precision : {ap:.4f}")
    print(f"ROC-AUC           : {roc_auc:.4f}")
    print(f"Precision @ Top {test_capacity} : {precision_at_k:.4f}")
    print(f"Confusion Matrix  :\n{cm}")
    
    # On oublie pas de renvoyer la matrice pour les tests
    return {
        "average_precision": ap, 
        "roc_auc": roc_auc, 
        "precision_at_k": precision_at_k,
        "confusion_matrix": cm
    }

def predict_from_json(pipeline: Pipeline, json_path: str, threshold: float) -> dict:
    """Load a JSON file and generate a Student Card."""
    # On ouvre le fichier JSON contenant les infos de l'étudiant
    with open(json_path, "r", encoding="utf-8") as f:
        student_data = json.load(f)
        
    # On appelle la fonction de l'Arc 10 pour générer la carte
    return predict_student(pipeline, student_data, threshold)