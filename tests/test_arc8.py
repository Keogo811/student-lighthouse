"""Tests for the decision threshold calculation."""
from lighthouse.data import DatasetLoader
from lighthouse.models import Trainer

def test_threshold_capacity() -> None:
    """Check that the threshold selects roughly the target capacity."""
    loader = DatasetLoader()
    df = loader.load_and_clean()
    X_train, X_test, y_train, y_test = loader.split(df)
    
    trainer = Trainer()
    # On calcule le seuil pour 200 étudiants
    threshold = trainer.calculate_threshold(X_train, y_train, capacity=200)
    
    # Vérification : si on applique ce seuil aux prédictions, on doit avoir ~200 positifs
    # (On utilise le test set juste pour vérifier la logique mathématique, pas pour l'entraînement)
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_predict
    
    best_model = LogisticRegression(random_state=42, class_weight="balanced", max_iter=1000, C=1.0)
    pipeline = trainer.build_pipeline(best_model)
    probs = cross_val_predict(pipeline, X_train, y_train, cv=5, method="predict_proba", n_jobs=-1)[:, 1]
    
    # On compte combien de probabilités dépassent le seuil
    predicted_positives = sum(1 for p in probs if p >= threshold)
    
    # À cause des égalités (ties), on peut avoir 199, 200 ou 201
    assert 195 <= predicted_positives <= 205