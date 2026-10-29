"""Tests for the Student Card prediction."""
from lighthouse.data import DatasetLoader
from lighthouse.models import Trainer
from lighthouse.predict import predict_student

def test_student_card() -> None:
    """Check that the student card is generated with human-readable signals."""
    loader = DatasetLoader()
    df = loader.load_and_clean()
    X_train, X_test, y_train, y_test = loader.split(df)
    
    trainer = Trainer()
    pipeline = trainer.fit(X_train, y_train)
    
    sample_student = X_test.iloc[0].to_dict()
    card = predict_student(pipeline, sample_student, threshold=0.9927)
    
    # Vérifications
    assert len(card["signals"]) == 3
    assert 0.0 <= card["probability"] <= 1.0
    # Au moins un signal doit être numérique (avec la valeur affichée)
    assert any(":" in s for s in card["signals"])