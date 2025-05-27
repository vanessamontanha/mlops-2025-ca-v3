FROM python:3.8

# Set working directory
WORKDIR /opt/app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application and training files
COPY ModelServing/app.py .
COPY monitor.py .
COPY synthetic_drift.csv .
COPY ModelTraining ModelTraining/

# Retrain the model inside the container to avoid numpy pickle issues
RUN python ModelTraining/train_model.py

# Expose Flask port
EXPOSE 5000

# Start the Flask app
CMD ["python", "app.py"]

