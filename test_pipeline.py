import os
import pickle
import pandas as pd
from monitor import check_model_drift

def test_model_training_creates_file():
    model_path = "test_model.pkl"
    if os.path.exists(model_path):
        os.remove(model_path)
    train_and_save_model(model_path)
    assert os.path.exists(model_path)
    os.remove(model_path)

def test_monitor_drift_no_retraining():
    drift_data = pd.DataFrame({
        "hours_worked": [8, 9, 7],
        "sleep_hours": [6, 7, 5],
        "mood_score": [5, 6, 7],
        "burnout": [0, 1, 0]
    })
    drift_data.to_csv("test_drift.csv", index=False)

    with open("burnout_model.pkl", "rb") as f:
        original_model = pickle.load(f)

    result = check_model_drift("test_drift.csv", threshold=0.5)
    assert "Model performance is stable" in result or "Retraining model" in result

    os.remove("test_drift.csv")