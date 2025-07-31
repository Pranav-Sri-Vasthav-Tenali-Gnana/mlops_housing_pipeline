# src/train.py

from utils import load_data, train_model, evaluate_model, save_model

# Step 1: Load dataset
X, y = load_data()

# Step 2: Train model
model = train_model(X, y)

# Step 3: Evaluate
r2, mse = evaluate_model(model, X, y)
print(f"R² Score: {r2:.4f}")
print(f"MSE: {mse:.4f}")

# Step 4: Save model
save_model(model)
