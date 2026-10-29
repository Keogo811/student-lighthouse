"""Tests for the Student Card prediction."""
from lighthouse.data import DatasetLoader
from lighthouse.models import Trainer
from lighthouse.predict import predict_student

def test_student_card() -> None:
    """Check that the student card is generated correctly."""
    loader = DatasetLoader()
    df = loader.load_and_clean()
    X_train, X_test, y_train, y_test = loader.split(df)
    
    trainer = Trainer()
    pipeline = trainer.fit(X_train, y_train)
    
    # On prend le premier étudiant du test set (converti en dictionnaire)
    sample_student = X_test.iloc[0].to_dict()
    
    # On génère la carte
    card = predict_student(pipeline, sample_student, threshold=0.9927)
    
    # Vérifications
    assert "probability" in card
    assert "status" in card
    assert "action" in card
    assert len(card["signals"]) == 3
    assert 0.0 <= card["probability"] <= 1.0