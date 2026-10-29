"""Command Line Interface for Lighthouse."""
import argparse
import joblib
from pathlib import Path
from lighthouse.data import DatasetLoader
from lighthouse.models import Trainer
from lighthouse.evaluate import evaluate_final_model, predict_from_json
from lighthouse.config import load_config

def main() -> None:
    """Parse arguments and route to the correct command."""
    # On prépare le gestionnaire de commandes (ex: lighthouse train)
    parser = argparse.ArgumentParser(description="Lighthouse: Student Dropout Prediction Tool")
    # On crée des sous-commandes pour chaque action
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Déclaration de la commande "train"
    subparsers.add_parser("train", help="Train the model and save it")

    # Déclaration de la commande "evaluate"
    subparsers.add_parser("evaluate", help="Evaluate the saved model on the test set")

    # Déclaration de la commande "predict" avec son argument obligatoire --input
    predict_parser = subparsers.add_parser("predict", help="Predict for one student from a JSON file")
    predict_parser.add_argument("--input", required=True, type=str, help="Path to the JSON file")

    # On lit ce que l'utilisateur a tapé dans le terminal
    args = parser.parse_args()
    
    # On charge la configuration et on prépare les chemins
    config = load_config()
    models_dir = Path(config["paths"]["models_dir"])
    model_path = models_dir / "lighthouse_model.joblib"
    threshold = 0.9927 # Le seuil qu'on a calculé à l'Arc 8

    # --- BRANCHE 1 : TRAIN ---
    if args.command == "train":
        print("Loading data...")
        loader = DatasetLoader()
        df = loader.load_and_clean()
        X_train, X_test, y_train, y_test = loader.split(df)
        
        print("Training final model...")
        trainer = Trainer()
        # On entraîne le pipeline sur le train set
        pipeline = trainer.fit(X_train, y_train)
        
        # On crée le dossier models/ si besoin et on sauvegarde le modèle
        models_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(pipeline, model_path)
        print(f"Model saved to {model_path}")

    # --- BRANCHE 2 : EVALUATE ---
    elif args.command == "evaluate":
        print("Loading model and data...")
        # On recharge le modèle sauvegardé
        pipeline = joblib.load(model_path)
        loader = DatasetLoader()
        df = loader.load_and_clean()
        X_train, X_test, y_train, y_test = loader.split(df)
        
        # On affiche les métriques
        evaluate_final_model(pipeline, X_test, y_test, threshold)

    # --- BRANCHE 3 : PREDICT ---
    elif args.command == "predict":
        print("Loading model...")
        # On recharge le modèle et on prédit sur un fichier JSON
        pipeline = joblib.load(model_path)
        predict_from_json(pipeline, args.input, threshold)