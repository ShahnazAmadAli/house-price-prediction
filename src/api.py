import pickle
from pathlib import Path

import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

app = FastAPI(title="House Price Prediction API")

model_path = Path("model") / "linear_regression_model.pkl"
with open(model_path, "rb") as f:
    model = pickle.load(f)


class HouseFeatures(BaseModel):
    crim: float = Field(..., ge=0)
    zn: float = Field(..., ge=0)
    indus: float = Field(..., ge=0)
    chas: int = Field(..., ge=0, le=1)
    nox: float = Field(..., ge=0)
    rm: float = Field(..., gt=0)
    age: float = Field(..., ge=0, le=100)
    dis: float = Field(..., gt=0)
    rad: int = Field(..., ge=1)
    tax: float = Field(..., ge=0)
    ptratio: float = Field(..., gt=0)
    b: float = Field(..., ge=0)
    lstat: float = Field(..., ge=0)


@app.get("/")
def root():
    return {"message": "API is running"}


@app.post("/predict")
def predict(features: HouseFeatures):
    df = pd.DataFrame([features.model_dump()])
    prediction = model.predict(df)[0]
    return {"predicted_price": round(float(prediction), 2)}