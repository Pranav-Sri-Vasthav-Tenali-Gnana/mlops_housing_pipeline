# src/quantize.py

import os
import numpy as np
import joblib
from utils import load_data, dequantize
from sklearn.metrics import r2_score, mean_squared_error

model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'model.joblib')
model = joblib.load(model_path)

coef = model.coef_
intercept = model.intercept_

unquant_params = {
    'coef': coef,
    'intercept': intercept
}
unquant_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'unquant_params.joblib')
joblib.dump(unquant_params, unquant_path)

def quantize(array, scale=0.01):
    q_array = np.clip(np.round(array / scale), 0, 255).astype(np.uint8)
    return q_array, scale

q_coef, scale = quantize(coef)
q_intercept, _ = quantize(np.array([intercept]))

quant_params = {
    'q_coef': q_coef,
    'q_intercept': q_intercept,
    'scale': scale
}
quant_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'quant_params.joblib')
joblib.dump(quant_params, quant_path)

X, _ = load_data()

dq_coef = dequantize(q_coef, scale)
dq_intercept = dequantize(q_intercept, scale)[0]

preds = np.dot(X, dq_coef) + dq_intercept

print("Sample predictions (dequantized):", preds[:5])

_, y = load_data()

r2 = r2_score(y, preds)
mse = mean_squared_error(y, preds)

print(f"Quantized Model - R² Score: {r2:.4f}")
print(f"Quantized Model - MSE: {mse:.4f}")
