"""Model trainer and pipeline."""
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from lighthouse.features import build_preprocessor

class Trainer:
    """Builds, trains, and evaluates scikit-learn pipelines."""

    def __init__(self) -> None:
        # On prépare le transformateur de colonnes
        self.preprocessor = build_preprocessor()

    def build_pipeline(self, model) -> Pipeline:
        """Create a full pipeline with preprocessing and a model."""
        # Le pipeline garantit qu'il n'y a pas de fuite de données
        return Pipeline(steps=[
            ("preprocessor", self.preprocessor),
            ("model", model)
        ])

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series, model=None) -> Pipeline:
        """Fit the pipeline on the training data."""
        # Si aucun modèle n'est fourni, on utilise la Régression Logistique par défaut
        if model is None:
            model = LogisticRegression(random_state=42, class_weight="balanced")
            
        pipeline = self.build_pipeline(model)
        pipeline.fit(X_train, y_train)
        return pipeline