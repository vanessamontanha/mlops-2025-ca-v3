import os
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import pickle
import sys

# Main training function
def train_model():
    try:
        print("[INFO] Starting model training...")

        # Load dataset from CSV
        df = pd.read_csv("ModelTraining/burnout_data.csv")
        X = df[["hours_worked", "sleep_hours", "mood_score"]]  # Feature matrix
        y = df["burnout"]  # Target variable

        # Split data into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Train logistic regression model
        model = LogisticRegression()
        model.fit(X_train, y_train)

        # Evaluate model on test data
        accuracy = model.score(X_test, y_test)
        print(f"[INFO] Training completed. Accuracy: {accuracy:.2f}")

        # Save model to disk
        model_path = "burnout_model.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(model, f)

        print(f"[INFO] Model saved to: {model_path}")

        # list contents of working directory for debug
        print("[INFO] Contents of /opt/app after training:")
        os.system("ls -lh /opt/app")

        # Return model and test data for evaluation
        return model, X_test, y_test

    except Exception as e:
        # Catch any error and exit
        print(f"[ERROR] {e}", file=sys.stderr)
        sys.exit(1)

# Wrapper so tests can call train_model via tm.main()
def main():
    train_model()

# Run training when script is executed directly
if __name__ == "__main__":
    main()
