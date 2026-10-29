"""Load and clean the raw dataset."""
import pandas as pd
from pathlib import Path
from lighthouse.config import load_config

class DatasetLoader:
    """Loads the CSV, drops the semester 2 columns, and maps the target."""

    def __init__(self) -> None:
        self.config = load_config()
        self.raw_path = Path(self.config["paths"]["raw_data"])

    def load_and_clean(self) -> pd.DataFrame:
        """Read CSV, remove leakage, and convert target to binary."""
        # Le fichier de l'UCI utilise des points-virgules, pas des virgules
        df = pd.read_csv(self.raw_path, sep=";")
        
        # 1. Supprimer les colonnes du semestre 2 (fuite temporelle)
        cols_to_drop = [col for col in df.columns if "2nd sem" in col]
        df = df.drop(columns=cols_to_drop)
        
        # 2. Mapping de la cible : Dropout = 1, le reste = 0
        df["target"] = df["Target"].apply(lambda x: 1 if x == "Dropout" else 0)
        
        # 3. Supprimer l'ancienne colonne cible (texte)
        df = df.drop(columns=["Target"])
        
        return df