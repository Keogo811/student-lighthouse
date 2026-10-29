"""Tests for the dataset loader."""
from lighthouse.data import DatasetLoader

def test_dataset_loads() -> None:
    """Checks that the file loads and has no 2nd semester columns."""
    loader = DatasetLoader()
    df = loader.load_and_clean()
    
    assert len(df) > 0
    # Vérifie qu'aucune colonne du S2 n'est présente
    has_s2 = any("2nd sem" in col for col in df.columns)
    assert not has_s2
    # Vérifie que la cible est binaire (0 ou 1)
    assert set(df["target"].unique()) == {0, 1}