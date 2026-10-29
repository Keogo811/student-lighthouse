"""Tests for the model comparison."""
from lighthouse.data import DatasetLoader
from lighthouse.models import Trainer

def test_model_comparison() -> None:
    """Check that cross-validation runs for all 3 models."""
    loader = DatasetLoader()
    df = loader.load_and_clean()
    
    # On prend un sous-échantillon pour que le test soit rapide (moins de 2 min)
    df_sample = df.sample(n=1000, random_state=42)
    X = df_sample.drop(columns=["target"])
    y = df_sample["target"]
    
    trainer = Trainer()
    results = trainer.compare_models(X, y)
    
    # Vérifie qu'on a bien 3 modèles dans les résultats
    assert len(results) == 3
    # Vérifie que le score est une probabilité (entre 0 et 1)
    for name in results:
        assert 0.0 <= results[name]["mean"] <= 1.0