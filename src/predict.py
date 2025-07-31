# src/predict.py

import os
import joblib
import numpy as np
from utils import dequantize, load_data

# Load quantized parameters
quant_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'quant_params.joblib')
params = joblib.load(quant_path)

q_coef = params['q_coef']
q_intercept = params['q_intercept']
scale = params['scale']

# Dequantize
coef = dequantize(q_coef, scale)
intercept = dequantize(q_intercept, scale)[0]

# Inference
X, _ = load_data()
preds = np.dot(X, coef) + intercept

print("Predictions from Docker container:", preds[:5])
