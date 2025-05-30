# monitor.py — Detects data drift and retrains model if accuracy drops below threshold

import pandas as pd
import pickle
import os
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from datetime import datetime
import logging
import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)

# Force output to appear immediately
sys.stdout.reconfigure(line_buffering=True)

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

print("monitor.py is running...")  # Basic heartbeat check

# Threshold below which retraining is triggered
ACCURACY_THRESHOLD = 0.90

# Load new data for monitoring (simulate drift)
drift_data_path = "synthetic_drift.csv"
if not os.path.exists(drift_data_path):
    logging.error(f"{drift_data_path} not found. Aborting monitoring.")
    exit(1)

df = pd.read_csv(drift_data_path)
X = df[["hours_worked", "sleep_hours", "mood_score"]]
y = df["burnout"]

# Load current model
try:
    with open("burnout_model.pkl", "rb") as f:
        model = pickle.load(f)
except FileNotFoundError:
    logging.error("burnout_model.pkl not found. Cannot evaluate or retrain.")
    exit(1)

# Evaluate model accuracy on new data
accuracy = model.score(X, y)
logging.info(f"Model accuracy on new data: {accuracy:.2f}")

# Check if accuracy is below threshold
if accuracy < ACCURACY_THRESHOLD:
    logging.warning("Model accuracy below threshold — data drift detected. Retraining model...")

    # Retrain model
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    new_model = LogisticRegression()
    new_model.fit(X_train, y_train)

    # Save retrained model with versioning
    version = datetime.now().strftime("%Y%m%d%H%M%S")
    retrained_model_name = f"burnout_model_v{version}.pkl"
    with open(retrained_model_name, "wb") as f:
        pickle.dump(new_model, f)
    logging.info(f"New model retrained and saved as {retrained_model_name}")

    # Update live model
    with open("burnout_model.pkl", "wb") as f:
        pickle.dump(new_model, f)
    logging.info("burnout_model.pkl updated with new retrained model.")
else:
    logging.info("Model performance is stable. No retraining needed.")
