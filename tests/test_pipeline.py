import sys
import os

# Ensure parent directory is on import path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import subprocess
import glob

# Test 1: Check model trains without error
def test_model_training_runs():
    try:
        from ModelTraining.train_model import train_model
        model, X_test, y_test = train_model()
    except Exception as e:
        assert False, f"Model training failed with error: {e}"

# Test 2: Check model meets minimum accuracy threshold
def test_model_accuracy_threshold():
    from ModelTraining.train_model import train_model
    model, X_test, y_test = train_model()
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    assert acc >= 0.8, f"Model accuracy too low: {acc}"

# Test 3: Ensure model file is saved
def test_model_file_saved():
    model_path = "burnout_model.pkl"
    if not os.path.exists(model_path):
        import ModelTraining.train_model as tm
        tm.main()  # Call the wrapper to train and save model
    assert os.path.exists(model_path), f"Model file not found at {model_path}"

# Test 4: Check that monitor.py triggers retraining and saves a new version
def test_monitor_triggers_retraining():
    # Create a small "drifted" dataset to trigger retraining
    with open("synthetic_drift.csv", "w") as f:
        f.write("hours_worked,sleep_hours,mood_score,burnout\n")
        f.write("1,2,1,1\n")
        f.write("1.5,2.5,1,1\n")
        f.write("2,3,1,1\n")
        f.write("3,3,2,0\n")

    # Run monitor script
    retrain_script = "monitor.py"
    subprocess.run(["python3", retrain_script], check=True)

    # Check for versioned model output
    versioned_models = glob.glob("burnout_model_v*.pkl")
    assert len(versioned_models) > 0, "No retrained model file found — drift retraining may have failed."



