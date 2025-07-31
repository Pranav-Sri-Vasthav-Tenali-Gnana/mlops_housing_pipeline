# src/utils.py

from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import joblib
import os

def load_data():
    return fetch_california_housing(return_X_y=True)

def train_model(X, y):
    model = LinearRegression()
    model.fit(X, y)
    return model

def evaluate_model(model, X, y):
    preds = model.predict(X)
    r2 = r2_score(y, preds)
    mse = mean_squared_error(y, preds)
    return r2, mse

def save_model(model, filepath=None):
    if filepath is None:
        filepath = os.path.join(os.path.dirname(__file__), "model.joblib")
    joblib.dump(model, filepath)
