import os
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import pickle
import sys

def train_model():
    try:
        print("[INFO] Starting model training...")

        df = pd.read_csv("ModelTraining/burnout_data.csv")
        X = df[["hours_worked", "sleep_hours", "mood_score"]]
        y = df["burnout"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        model = LogisticRegression()
        model.fit(X_train, y_train)

        accuracy = model.score(X_test, y_test)
        print(f"[INFO] Training completed. Accuracy: {accuracy:.2f}")

        model_path = "burnout_model.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(model, f)

        print(f"[INFO] Model saved to: {model_path}")
        print("[INFO] Contents of /opt/app after training:")
        os.system("ls -lh /opt/app")

        return model, X_test, y_test

    except Exception as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        sys.exit(1)

# Optional: run the function if the script is executed directly
if __name__ == "__main__":
    train_model()
