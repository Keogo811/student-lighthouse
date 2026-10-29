"""Tests for final model evaluation."""
from lighthouse.data import DatasetLoader
from lighthouse.models import Trainer
from lighthouse.evaluate import evaluate_final_model

def test_final_evaluation() -> None:
    """Check that final evaluation runs and returns valid metrics."""
    loader = DatasetLoader()
    df = loader.load_and_clean()
    X_train, X_test, y_train, y_test = loader.split(df)
    
    trainer = Trainer()
    # On entraîne le modèle final sur tout le train
    pipeline = trainer.fit(X_train, y_train)
    
    # On évalue sur le test (avec le seuil qu'on a trouvé)
    results = evaluate_final_model(pipeline, X_test, y_test, threshold=0.9927)
    
    # Vérifications
    assert 0.0 <= results["average_precision"] <= 1.0
    assert 0.0 <= results["roc_auc"] <= 1.0
    assert 0.0 <= results["precision_at_k"] <= 1.0
    assert results["confusion_matrix"].shape == (2, 2)