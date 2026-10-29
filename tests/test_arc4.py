"""Tests pour le split et la baseline."""
from lighthouse.data import DatasetLoader
from lighthouse.evaluate import get_baseline_scores

def test_split_no_leakage() -> None:
    """Vérifie que le train et le test sont bien séparés."""
    loader = DatasetLoader()
    df = loader.load_and_clean()
    X_train, X_test, y_train, y_test = loader.split(df)
    
    # Le total doit faire 4424
    assert len(X_train) + len(X_test) == 4424
    # Le test doit faire 20% (environ 885)
    assert len(X_test) == 885
    
    # Vérifie qu'aucune ligne du test n'est dans le train
    train_indices = set(X_train.index)
    test_indices = set(X_test.index)
    assert len(train_indices.intersection(test_indices)) == 0

def test_baseline_runs() -> None:
    """Vérifie que la baseline s'exécute sans erreur."""
    loader = DatasetLoader()
    df = loader.load_and_clean()
    X_train, X_test, y_train, y_test = loader.split(df)
    
    scores = get_baseline_scores(X_train, y_train, X_test, y_test)
    
    # L'accuracy doit être autour de 68% (car 32% de décrocheurs)
    assert 0.60 < scores["accuracy"] < 0.75