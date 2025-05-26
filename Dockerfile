# Base image
FROM python:3.8

# Set work directory
WORKDIR /opt/app

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy scripts and training data
COPY ModelServing/app.py .
COPY ModelTraining/train_model.py .
COPY ModelTraining/burnout_data.csv .

# Retrain the model inside Docker
RUN python train_model.py

# Expose Flask port
EXPOSE 5000

# Run app
CMD ["python", "app.py"]