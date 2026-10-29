"""Model trainer and pipeline."""
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_validate, GridSearchCV
from lighthouse.features import build_preprocessor

class Trainer:
    """Builds, trains, and evaluates scikit-learn pipelines."""

    def __init__(self) -> None:
        self.preprocessor = build_preprocessor()

    def build_pipeline(self, model) -> Pipeline:
        """Create a full pipeline with preprocessing and a model."""
        return Pipeline(steps=[
            ("preprocessor", self.preprocessor),
            ("model", model)
        ])

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series, model=None) -> Pipeline:
        """Fit the pipeline on the training data."""
        if model is None:
            model = LogisticRegression(random_state=42, class_weight="balanced", max_iter=1000)
        pipeline = self.build_pipeline(model)
        pipeline.fit(X_train, y_train)
        return pipeline

    def compare_models(self, X: pd.DataFrame, y: pd.Series) -> dict:
        """Compare 3 models using cross-validation and return their scores."""
        models = {
            "Logistic Regression": LogisticRegression(random_state=42, class_weight="balanced", max_iter=1000),
            "k-NN": KNeighborsClassifier(n_neighbors=5),
            "Linear SVM": LinearSVC(random_state=42, class_weight="balanced")
        }
        
        results = {}
        for name, model in models.items():
            pipeline = self.build_pipeline(model)
            cv_results = cross_validate(pipeline, X, y, cv=5, scoring="average_precision", n_jobs=-1)
            results[name] = {"mean": cv_results["test_score"].mean(), "std": cv_results["test_score"].std()}
        return results

    def tune_and_select(self, X: pd.DataFrame, y: pd.Series) -> dict:
        """Tune hyperparameters with GridSearchCV and select the best model."""
        # 1. On définit les modèles et leurs grilles d'hyperparamètres
        model_grids = {
            "Logistic Regression": {
                "model": LogisticRegression(random_state=42, class_weight="balanced", max_iter=1000),
                "params": {"model__C": [0.1, 1.0, 10.0]} # C = force de régularisation
            },
            "k-NN": {
                "model": KNeighborsClassifier(),
                "params": {"model__n_neighbors": [3, 5, 7, 11]} # Nombre de voisins
            },
            "Linear SVM": {
                "model": LinearSVC(random_state=42, class_weight="balanced"),
                "params": {"model__C": [0.1, 1.0, 10.0]}
            }
        }
        
        best_name = ""
        best_score = -1.0
        best_params = {}
        
        # 2. On fait le GridSearch pour chaque modèle
        for name, config in model_grids.items():
            pipeline = self.build_pipeline(config["model"])
            
            grid = GridSearchCV(
                pipeline, 
                config["params"], 
                cv=5, 
                scoring="average_precision", 
                n_jobs=-1
            )
            grid.fit(X, y)
            
            print(f"{name} : Meilleur score = {grid.best_score_:.4f} | Params = {grid.best_params_}")
            
            # 3. On garde le meilleur modèle global
            if grid.best_score_ > best_score:
                best_score = grid.best_score_
                best_name = name
                best_params = grid.best_params_
                
        print(f"\n--> Modèle retenu : {best_name} (Score : {best_score:.4f})")
        
        return {
            "best_name": best_name,
            "best_score": best_score,
            "best_params": best_params,
            "best_estimator": grid.best_estimator_
        }
        
    def calculate_threshold(self, X: pd.DataFrame, y: pd.Series, capacity: int = 200) -> float:
        """Calculate the decision threshold to select the top N students."""
        from sklearn.model_selection import cross_val_predict
        
        # On utilise le meilleur modèle trouvé (LogReg avec C=1.0)
        best_model = LogisticRegression(random_state=42, class_weight="balanced", max_iter=1000, C=1.0)
        pipeline = self.build_pipeline(best_model)
        
        # On récupère les probabilités hors-pli (out-of-fold) pour le train set
        # Cela simule ce qui se passera sur de nouvelles données sans toucher au test set
        probs = cross_val_predict(pipeline, X, y, cv=5, method="predict_proba", n_jobs=-1)[:, 1]
        
        # On trie les probabilités de la plus haute à la plus basse
        sorted_probs = sorted(probs, reverse=True)
        
        # Le seuil est la probabilité du 200ème étudiant
        if len(sorted_probs) >= capacity:
            threshold = float(sorted_probs[capacity - 1])
        else:
            threshold = 0.5 # Sécurité si le dataset est trop petit
            
        print(f"Seuil pour contacter {capacity} étudiants : {threshold:.4f}")
        return threshold