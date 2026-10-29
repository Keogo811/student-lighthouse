"""Model trainer and pipeline."""
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_validate
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

    def compare_models(self, X: pd.DataFrame, y: pd.Series) -> dict:
        """Compare 3 models using cross-validation and return their scores."""
        # On définit nos 3 modèles de base
        models = {
            "Logistic Regression": LogisticRegression(random_state=42, class_weight="balanced", max_iter=1000),
            "k-NN": KNeighborsClassifier(n_neighbors=5),
            "Linear SVM": LinearSVC(random_state=42, class_weight="balanced")
        }
        
        results = {}
        
        # On teste chaque modèle avec une validation croisée (5 folds)
        for name, model in models.items():
            pipeline = self.build_pipeline(model)
            
            # On utilise l'Average Precision (gère bien les classes déséquilibrées)
            cv_results = cross_validate(
                pipeline, X, y, 
                cv=5, 
                scoring="average_precision",
                n_jobs=-1 # Utilise tous les cœurs du processeur
            )
            
            # On garde la moyenne et l'écart-type
            mean_score = cv_results["test_score"].mean()
            std_score = cv_results["test_score"].std()
            
            results[name] = {"mean": mean_score, "std": std_score}
            print(f"{name}: {mean_score:.4f} (+/- {std_score:.4f})")
            
        return results