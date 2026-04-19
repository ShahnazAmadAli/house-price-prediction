import pickle
import pandas as pd
from pathlib import Path

# Load saved model
model_path = Path("model") / "linear_regression_model.pkl"
with open(model_path, "rb") as f:
    model = pickle.load(f)

# Example input
sample = pd.DataFrame([{
    "crim": 0.1,
    "zn": 18.0,
    "indus": 2.31,
    "chas": 0,
    "nox": 0.538,
    "rm": 6.575,
    "age": 65.2,
    "dis": 4.09,
    "rad": 1,
    "tax": 296,
    "ptratio": 15.3,
    "b": 396.9,
    "lstat": 4.98
}])

# Predict
prediction = model.predict(sample)

print(f"Predicted house price: {prediction[0]:.2f}")