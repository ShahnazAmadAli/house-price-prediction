import pickle
from pathlib import Path

import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="House Price Prediction API")

model_path = Path("model") / "linear_regression_model.pkl"
with open(model_path, "rb") as f:
    model = pickle.load(f)


class HouseFeatures(BaseModel):
    crim: float
    zn: float
    indus: float
    chas: int
    nox: float
    rm: float
    age: float
    dis: float
    rad: int
    tax: float
    ptratio: float
    b: float
    lstat: float


@app.get("/")
def root():
    return {"message": "API is running"}


@app.post("/predict")
def predict(features: HouseFeatures):
    df = pd.DataFrame([features.model_dump()])
    prediction = model.predict(df)[0]
    return {"predicted_price": round(float(prediction), 2)}