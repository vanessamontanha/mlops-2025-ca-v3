import pandas as pd
import os
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import datetime
import sys

# Ensure stdout doesn't crash in Python 3.6
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)

# Load existing model from disk
def load_latest_model():
    if not os.path.exists("burnout_model.pkl"):
        return None
    with open("burnout_model.pkl", "rb") as f:
        return pickle.load(f)

# Check if current model accuracy on drift data is below threshold
def detect_drift(model, X_drift, y_drift, threshold=0.75):
    preds = model.predict(X_drift)
    acc = accuracy_score(y_drift, preds)
    print(f"[INFO] Drift check accuracy: {acc:.2f}")
    return acc < threshold  # Trigger retraining if accuracy is too low

# Retrain logistic regression model from new data
def retrain_model(X, y):
    model = LogisticRegression()
    model.fit(X, y)
    return model

# Save retrained model with versioned filename
def save_versioned_model(model):
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    versioned_filename = f"burnout_model_v{timestamp}.pkl"
    with open(versioned_filename, "wb") as f:
        pickle.dump(model, f)
    print(f"[INFO] New version saved: {versioned_filename}")

# Main monitoring logic
def main():
    try:
        # Load drift dataset
        df = pd.read_csv("synthetic_drift.csv")
        X_drift = df[["hours_worked", "sleep_hours", "mood_score"]]
        y_drift = df["burnout"]

        model = load_latest_model()

        if model is None:
            print("[WARNING] No existing model found. Skipping drift detection.")
            return

        # If drift is detected, retrain and version the model
        if detect_drift(model, X_drift, y_drift):
            print("[INFO] Drift detected. Retraining...")
            model = retrain_model(X_drift, y_drift)
            save_versioned_model(model)
        else:
            print("[INFO] No drift detected. No action needed.")

    except Exception as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        sys.exit(1)

# Allow running monitor.py directly
if __name__ == "__main__":
    main()
