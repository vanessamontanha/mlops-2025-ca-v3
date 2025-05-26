# monitor.py – triggers retraining if model accuracy drops

import pandas as pd
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import numpy as np

# Load synthetic drift data
df = pd.read_csv("synthetic_drift.csv")

X = df[["hours_worked", "sleep_hours", "mood_score"]]
y = df["burnout"]

# Split into test to simulate validation
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Load existing model
with open("burnout_model.pkl", "rb") as f:
    model = pickle.load(f)

# Evaluate on drifted data
accuracy = model.score(X_test, y_test)
print(f"Drift check accuracy: {accuracy:.2f}")

# If performance drops, retrain model and overwrite .pkl
if accuracy < 0.85:
    print("Accuracy below threshold. Retraining...")
    new_model = LogisticRegression()
    new_model.fit(X_train, y_train)
    with open("ModelTraining/burnout_model.pkl", "wb") as f:
        pickle.dump(new_model, f)
else:
    print("Model is stable. No retraining needed.")