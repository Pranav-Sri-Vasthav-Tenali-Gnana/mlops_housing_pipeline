# MLOps Pipeline: California Housing Price Prediction

An end-to-end MLOps workflow around a scikit-learn Linear Regression model on the California Housing dataset: modular training code, manual `uint8` quantization, unit tests, a Docker image for inference, and a GitHub Actions CI/CD pipeline that runs on every push.

**Stack:** Python · scikit-learn · NumPy · joblib · pytest · Docker · GitHub Actions

## Features

- Modular code: shared helpers in `src/utils.py`, separate train / quantize / predict scripts
- Linear Regression as the only model (assignment constraint)
- Manual quantization of coefficients and intercept to `uint8`, plus dequantization for inference
- Unit tests with pytest
- Dockerized prediction pipeline
- CI/CD with GitHub Actions: test → train → quantize → build the Docker image → run inference

## Repository layout

```
mlops_housing_pipeline/
├── models/                  # trained model + quantized / unquantized parameters
├── src/
│   ├── train.py             # train and save the Linear Regression model
│   ├── quantize.py          # uint8 quantization, dequantization and evaluation
│   ├── predict.py           # inference (used inside the Docker image)
│   └── utils.py             # data loading and shared helpers
├── tests/test_train.py      # pytest unit tests
├── .github/workflows/ci.yml # CI/CD pipeline
├── Dockerfile
└── requirements.txt
```

## Run it

```bash
python -m venv venv
source venv/bin/activate            # Windows: venv\Scripts\activate
pip install -r requirements.txt

python src/train.py                 # train and save the model
python src/quantize.py              # quantize parameters and compare
PYTHONPATH=src pytest tests/        # run unit tests

docker build -t mlops-housing .
docker run mlops-housing            # run inference in the container
```

## Results: original vs quantized

| Metric   | Original model | Quantized model (`uint8`) |
|----------|----------------|---------------------------|
| R²       | 0.6062         | 0.1328                    |
| MSE      | 0.5243         | 1.1548                    |
| Size     | 697 bytes      | 349 bytes                 |

Quantization halves the parameter size, but costs a lot of accuracy here. The reason is a design constraint worth knowing about:

`uint8` can only hold values from 0 to 255, so the quantizer clips every **negative** coefficient of the regression to 0. Several California Housing features have negative weights, so the quantized model loses their contribution entirely, and R² drops from 0.61 to 0.13.

**How I would fix it outside the assignment constraints:** use signed `int8`, or keep `uint8` with an asymmetric scheme (a per-tensor scale and zero-point, as in PyTorch and ONNX Runtime quantization) so negative values map into the 0–255 range instead of being clipped. My [ALLM-1](https://github.com/Pranav-Sri-Vasthav-Tenali-Gnana/ALLM-1) project uses exactly that approach for transformer models and keeps accuracy within 0.1 points.

## CI/CD

`.github/workflows/ci.yml` runs on every push to `main`:

1. Unit tests with pytest
2. Model training and quantization
3. Docker image build and a containerised inference run
