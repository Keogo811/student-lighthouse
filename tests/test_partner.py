"""Tests to verify model outputs."""
from lighthouse.data import DatasetLoader
from lighthouse.models import Trainer

def test_probabilities_are_valid() -> None:
    """Check that predicted probabilities are between 0 and 1."""
    loader = DatasetLoader()
    df = loader.load_and_clean()
    X_train, X_test, y_train, y_test = loader.split(df)
    
    trainer = Trainer()
    pipeline = trainer.fit(X_train, y_train)
    
    # On prend 10 étudiants au hasard
    probs = pipeline.predict_proba(X_test[:10])[:, 1]
    
    # Vérifie que chaque probabilité est bien un pourcentage valide
    for p in probs:
        assert 0.0 <= p <= 1.0