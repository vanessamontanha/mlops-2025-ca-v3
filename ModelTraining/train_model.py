import os
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import pickle
import sys

try:
    # Log: start of training
    print("[INFO] Starting model training...")

    # Load dataset (must be in the same format and path as expected)
    df = pd.read_csv("ModelTraining/burnout_data.csv")

    # Select features (independent variables) and label (dependent variable)
    X = df[["hours_worked", "sleep_hours", "mood_score"]]  # Features
    y = df["burnout"]  # Target variable: 0 (no burnout) or 1 (burnout)

    # Split data into training and testing sets (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Initialize and train a logistic regression model
    model = LogisticRegression()
    model.fit(X_train, y_train)

    # Evaluate the trained model on the test set
    accuracy = model.score(X_test, y_test)
    print(f"[INFO] Training completed. Accuracy: {accuracy:.2f}")

    # Save the trained model to disk using pickle
    # This file will later be used by the Flask API to make predictions
    model_path = "burnout_model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    print(f"[INFO] Model saved to: {model_path}")

    # (Optional) Log the contents of the container/app directory for verification
    print("[INFO] Contents of /opt/app after training:")
    os.system("ls -lh /opt/app")

except Exception as e:
    # Catch any exceptions, log them, and exit with non-zero status
    print(f"[ERROR] {e}", file=sys.stderr)
    sys.exit(1)
