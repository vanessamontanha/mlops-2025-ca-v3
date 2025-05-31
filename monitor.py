import pandas as pd
import os
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import datetime
import sys
import numpy as np

# Compatibility guard for older Python versions (like 3.6)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)

# Load the latest trained model from disk
def load_latest_model():
    if not os.path.exists("burnout_model.pkl"):
        return None
    with open("burnout_model.pkl", "rb") as f:
        return pickle.load(f)

# Check if model performance on new data is below threshold (indicating drift)
def detect_drift(model, X_drift, y_drift, threshold=0.75):
    preds = model.predict(X_drift)
    acc = accuracy_score(y_drift, preds)
    print(f"[INFO] Drift check accuracy: {acc:.2f}")
    return acc < threshold

# Retrain model from scratch using the drifted dataset
def retrain_model(X, y):
    model = LogisticRegression()
    model.fit(X, y)
    return model

# Save retrained model with timestamp versioning
def save_versioned_model(model):
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    versioned_filename = f"burnout_model_v{timestamp}.pkl"
    with open(versioned_filename, "wb") as f:
        pickle.dump(model, f)
    # Also overwrite the main model for serving
    with open("burnout_model.pkl", "wb") as f:
        pickle.dump(model, f)
    print(f"[INFO] New model saved as {versioned_filename} and updated burnout_model.pkl")

# Main function logic: monitor, detect drift, retrain if needed
def main():
    # Load new data to check for drift
    df = pd.read_csv("synthetic_drift.csv")
    X_drift = df[["hours_worked", "sleep_hours", "mood_score"]]
    y_drift = df["burnout"]

    # Load latest model
    model = load_latest_model()
    if model is None:
        print("[WARNING] No existing model found. Skipping drift check.")
        return

    # Check for drift and retrain if needed
    if detect_drift(model, X_drift, y_drift):
        print("[INFO] Drift detected. Retraining model...")
        model = retrain_model(X_drift, y_drift)
        save_versioned_model(model)
    else:
        print("[INFO] No drift detected. Model is still valid.")

# Entry point
if __name__ == "__main__":
    main()
