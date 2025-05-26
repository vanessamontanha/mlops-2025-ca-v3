# Use official Python 3.8 image
FROM python:3.8

# Set the working directory
WORKDIR /opt/app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all required files
COPY ModelServing/app.py .
COPY ModelTraining/train_model.py .
COPY ModelTraining/burnout_data.csv .

# Train the model inside this container
RUN python train_model.py

# Expose the Flask port
EXPOSE 5000

# Run the Flask app
CMD ["python", "app.py"]