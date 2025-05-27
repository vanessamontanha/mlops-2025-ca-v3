import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import subprocess
import glob

# Test 1: Model training runs without error
def test_model_training_runs():
    try:
        from ModelTraining.train_model import train_model
        model, X_test, y_test = train_model()
    except Exception as e:
        assert False, f"Model training failed with error: {e}"

# Test 2: Model achieves acceptable accuracy on training data
def test_model_accuracy_threshold():
    from ModelTraining.train_model import train_model
    model, X_test, y_test = train_model()
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    assert acc >= 0.8, f"Model accuracy too low: {acc}"

# Test 3: Model file is saved after training
def test_model_file_saved():
    model_path = "ModelTraining/burnout_model.pkl"
    assert os.path.exists(model_path), f"Model file not found at {model_path}"

# Test 4: Drifted data triggers retraining and creates a new versioned model file
def test_monitor_triggers_retraining():
    # Overwrite the drift file with low quality data to force drift
    with open("synthetic_drift.csv", "w") as f:
        f.write("hours_worked,sleep_hours,mood_score,burnout\n")
        f.write("1,2,1,1\n")
        f.write("1.5,2.5,1,1\n")
        f.write("2,3,1,1\n")
        f.write("3,3,2,0\n")

    retrain_script = "monitor.py"
    subprocess.run(["python3", retrain_script], check=True)

    # Look for versioned retrained model file in root
    versioned_models = glob.glob("burnout_model_v*.pkl")
    assert len(versioned_models) > 0, "No retrained model file found — drift retraining may have failed."



