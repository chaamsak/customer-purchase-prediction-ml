from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import pandas as pd
import joblib
from preprocessing import preprocess_input

app = FastAPI()

# Chargement du modèle
model = joblib.load("modele_random_forest.joblib")

# Exemple de schéma pour la saisie manuelle
class InputData(BaseModel):
    Administrative: int
    Administrative_Duration: float
    Informational: int
    Informational_Duration: float
    ProductRelated: int
    ProductRelated_Duration: float
    BounceRates: float
    ExitRates: float
    PageValues: float
    SpecialDay: float
    Month: str
    OperatingSystems: int
    Browser: int
    Region: int
    TrafficType: int
    VisitorType: str
    Weekend: bool

@app.post("/predict_file/")
async def predict_file(file: UploadFile = File(...)):
    df = pd.read_csv(file.file)
    X = preprocess_input(df)
    preds = model.predict(X)
    proba = model.predict_proba(X)[:, 1]
    return JSONResponse({
        "predictions": preds.tolist(),
        "probabilities": proba.tolist()
    })

@app.post("/predict_manual/")
async def predict_manual(data: InputData):
    df = pd.DataFrame([data.dict()])
    X = preprocess_input(df)
    pred = model.predict(X)[0]
    proba = model.predict_proba(X)[0, 1]
    return {"prediction": int(pred), "probability": float(proba)} 