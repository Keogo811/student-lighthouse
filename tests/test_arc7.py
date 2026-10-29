"""Tests for hyperparameter tuning and model selection."""
from lighthouse.data import DatasetLoader
from lighthouse.models import Trainer

def test_tune_and_select() -> None:
    """Check that GridSearchCV runs and returns a valid best model."""
    loader = DatasetLoader()
    df = loader.load_and_clean()
    
    # On prend 1500 lignes pour garder le test sous les 2 minutes
    df_sample = df.sample(n=1500, random_state=42)
    X = df_sample.drop(columns=["target"])
    y = df_sample["target"]
    
    trainer = Trainer()
    result = trainer.tune_and_select(X, y)
    
    # Vérifie qu'on a bien un modèle sélectionné et des paramètres trouvés
    assert "best_name" in result
    assert "best_params" in result
    assert len(result["best_params"]) > 0
    assert result["best_score"] > 0.0