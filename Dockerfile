# Use lightweight Python image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy only what’s needed
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy code and models
COPY src/ src/
COPY models/ models/

# Run inference
CMD ["python", "src/predict.py"]
