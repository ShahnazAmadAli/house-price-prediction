import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from fastapi.testclient import TestClient
from src.api import app

client = TestClient(app)


def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "API is running"}


def test_predict():
    payload = {
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
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 200
    assert "predicted_price" in response.json()


def test_predict_invalid_input():
    payload = {
        "crim": -1,
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
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 422