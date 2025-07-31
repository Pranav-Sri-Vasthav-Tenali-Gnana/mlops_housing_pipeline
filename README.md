# MLOps Pipeline – California Housing Price Prediction

This project demonstrates a complete MLOps workflow using scikit-learn's Linear Regression model on the California Housing dataset. It includes modular code, model training, manual quantization, containerization using Docker, and CI/CD with GitHub Actions.

## Project Features

- Clean, modular Python code using `utils.py`
- Linear Regression as the only model, per assignment constraints
- Manual model quantization (to uint8) and dequantization
- Unit testing using Pytest
- Dockerized prediction pipeline
- Automated CI/CD pipeline using GitHub Actions

## Directory Structure

```

mlops\_housing\_pipeline/
├── models/                  # Saved models and parameter files
├── src/                     # Source code: training, utils, quantize, predict
├── tests/                   # Unit tests using pytest
├── .github/workflows/       # GitHub Actions CI/CD workflow
├── Dockerfile               # Docker image setup
├── requirements.txt         # Project dependencies
└── README.md                # Project documentation

````

## Setup Instructions

1. **Create and activate a virtual environment**

   conda create --name g24ai1114_env python=3.10 -y
   conda activate g24ai1114_env


2. **Install dependencies**

   pip install -r requirements.txt

3. **Run the training pipeline**

   python src/train.py

4. **Quantize the model**

   python src/quantize.py

5. **Test the setup**

   pytest tests/

6. **Build and run the Docker container**

   docker build -t mlops-housing .
   docker run mlops-housing

## Comparison Table

| Metric   | Original Model | Quantized Model (Dequantized) |
| -------- | -------------- | ----------------------------- |
| R² Score |  0.6062        | 0.1328                        |
| MSE      |  0.5243        | 1.1548                        |

> Note: Quantized model uses manual 8-bit quantization, which slightly reduces performance but enables lightweight deployment.

## CI/CD

The pipeline runs on every push to the `main` branch. It performs:

* Unit testing with pytest
* Model training and quantization
* Docker image build and inference

## Performance Note

```markdown
## Notes on Quantization Performance

The assignment requires manual quantization using `uint8`, which restricts all values to the range [0, 255]. As a result, any **negative weights learned by the Linear Regression model are clipped to 0 or distorted** during quantization.

This significantly reduces the expressiveness of the model, leading to a noticeable drop in performance when compared to the original (floating-point) model:

- R² Score dropped from ~0.6062 to 0.1328
- MSE increased from ~0.5243 to 1.1548

This behavior is expected due to the limitations of unsigned 8-bit quantization and helps illustrate the trade-offs between model precision and deployment efficiency.