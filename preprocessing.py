import pandas as pd
import numpy as np
import joblib

# Charger le modèle pour obtenir la vraie liste des colonnes attendues
model = joblib.load("modele_random_forest.joblib")
MODEL_FEATURES = list(model.feature_names_in_)

CATEGORICAL = ["Month", "VisitorType"]


def preprocess_input(data: pd.DataFrame) -> pd.DataFrame:
    # Encodage one-hot uniquement sur Month et VisitorType (drop_first=True)
    data = pd.get_dummies(data, columns=CATEGORICAL, drop_first=True)
    # S'assurer que Weekend est bien booléen et existe
    if 'Weekend' in data.columns:
        data['Weekend'] = data['Weekend'].astype(bool)
    else:
        data['Weekend'] = False
    # Ajouter les colonnes manquantes attendues par le modèle
    for col in MODEL_FEATURES:
        if col not in data.columns:
            data[col] = 0
    # Supprimer toutes les colonnes inattendues
    data = data[MODEL_FEATURES]
    return data 