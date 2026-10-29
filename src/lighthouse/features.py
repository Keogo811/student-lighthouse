"""Feature definitions and preprocessing pipeline."""
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# Colonnes qui sont de vrais nombres (notes, âge, unités, etc.)
NUMERICAL_COLS = [
    "Previous qualification (grade)", "Admission grade", "Age at enrollment",
    "Curricular units 1st sem (credited)", "Curricular units 1st sem (enrolled)",
    "Curricular units 1st sem (evaluations)", "Curricular units 1st sem (approved)",
    "Curricular units 1st sem (grade)", "Curricular units 1st sem (without evaluations)",
    "Unemployment rate", "Inflation rate", "GDP"
]

# Colonnes qui sont des codes (filière, statut marital, etc.)
CATEGORICAL_COLS = [
    "Marital status", "Application mode", "Course", "Daytime/evening attendance",
    "Previous qualification", "Nacionality", "Mother's qualification",
    "Father's qualification", "Mother's occupation", "Father's occupation",
    "Displaced", "Educational special needs", "Debtor", "Tuition fees up to date",
    "Gender", "Scholarship holder", "International"
]

def build_preprocessor() -> ColumnTransformer:
    """Build the preprocessing pipeline for numeric and categorical data."""
    # On met à l'échelle les nombres, et on fait du One-Hot pour les catégories
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERICAL_COLS),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_COLS)
        ],
        remainder="drop" # On ignore les autres colonnes (ex: l'ID)
    )
    return preprocessor