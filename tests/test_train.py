# tests/test_train.py

import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
import joblib
import pytest
import utils
from sklearn.linear_model import LinearRegression

def test_data_loading():
    X, y = utils.load_data()
    assert X.shape[0] == y.shape[0]
    assert X.shape[1] > 0

def test_model_training():
    X, y = utils.load_data()
    model = utils.train_model(X, y)
    assert isinstance(model, LinearRegression)

def test_model_evaluation():
    X, y = utils.load_data()
    model = utils.train_model(X, y)
    r2, _ = utils.evaluate_model(model, X, y)
    assert r2 > 0.5  # Expect reasonable R²

def test_model_saving_and_loading():
    X, y = utils.load_data()
    model = utils.train_model(X, y)
    utils.save_model(model, "test_model.joblib")
    path = os.path.join("models", "test_model.joblib")
    assert os.path.exists(path)

    loaded_model = joblib.load(path)
    assert isinstance(loaded_model, LinearRegression)

    os.remove(path)  # Cleanup test model
