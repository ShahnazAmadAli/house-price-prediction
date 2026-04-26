# House Price Prediction

An end-to-end machine learning project for predicting house prices using scikit-learn, with model serving through FastAPI.

## Features
- Train a linear regression model on the Boston Housing dataset
- Evaluate model performance using MSE and R²
- Save the trained model with pickle
- Load the saved model for inference
- Serve predictions through a FastAPI API
- Interactive API testing via Swagger UI (`/docs`)

## Project Structure

```text
house-price-prediction/
├── data/
│   └── BostonHousing.csv
├── model/
│   └── linear_regression_model.pkl
├── notebooks/
├── src/
│   ├── api.py
│   ├── predict.py
│   └── train.py
├── .gitignore
├── README.md
└── requirements.txt