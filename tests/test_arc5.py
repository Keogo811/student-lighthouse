"""Tests for the preprocessing pipeline and no leakage."""
import numpy as np
from lighthouse.data import DatasetLoader
from lighthouse.models import Trainer

def test_pipeline_no_leakage() -> None:
    """Check that the scaler is fitted only on the training set."""
    loader = DatasetLoader()
    df = loader.load_and_clean()
    X_train, X_test, y_train, y_test = loader.split(df)
    
    trainer = Trainer()
    pipeline = trainer.fit(X_train, y_train)
    
    # On récupère le scaler qui a été entraîné
    scaler = pipeline.named_steps["preprocessor"].named_transformers_["num"]
    
    # On vérifie que la moyenne apprise par le scaler est celle du TRAIN uniquement
    # et pas celle de tout le dataset.
    train_mean = X_train["Age at enrollment"].mean()
    full_mean = df["Age at enrollment"].mean()
    
    # np.isclose compare deux nombres à virgule
    assert np.isclose(scaler.mean_[2], train_mean, rtol=1e-3)
    assert not np.isclose(scaler.mean_[2], full_mean, rtol=1e-5)