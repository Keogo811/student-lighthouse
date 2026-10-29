"""Load and clean the raw dataset."""
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from lighthouse.config import load_config

class DatasetLoader:
    """Loads the CSV, drops the semester 2 columns, and maps the target."""

    def __init__(self) -> None:
        self.config = load_config()
        self.raw_path = Path(self.config["paths"]["raw_data"])

    def load_and_clean(self) -> pd.DataFrame:
        """Read CSV, remove leakage, and convert target to binary."""
        # Le fichier de l'UCI utilise des points-virgules
        df = pd.read_csv(self.raw_path, sep=";")
        
        # 1. Supprimer les colonnes du semestre 2 (fuite temporelle)
        cols_to_drop = [col for col in df.columns if "2nd sem" in col]
        df = df.drop(columns=cols_to_drop)
        
        # 2. Mapping de la cible : Dropout = 1, le reste = 0
        df["target"] = df["Target"].apply(lambda x: 1 if x == "Dropout" else 0)
        
        # 3. Supprimer l'ancienne colonne cible (texte)
        df = df.drop(columns=["Target"])
        
        return df

    def split(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Split data into train and test sets with stratification."""
        X = df.drop(columns=["target"])
        y = df["target"]
        
        # On utilise les valeurs de config.yaml
        test_size = self.config["split"]["test_size"]
        seed = self.config["seed"]
        
        return train_test_split(
            X, y, 
            test_size=test_size, 
            stratify=y, 
            random_state=seed
        )