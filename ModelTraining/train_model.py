import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib
import os

def train_model():
    # Load the burnout dataset
    df = pd.read_csv("ModelTraining/burnout_data.csv")

    # Separate features and target
    X = df[["hours_worked", "sleep_hours", "mood_score"]]
    y = df["burnout"]

    # Split into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize and train the model
    model = LogisticRegression()
    model.fit(X_train, y_train)

    # Evaluate performance
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model accuracy: {accuracy:.2f}")

    # Save model to ModelTraining/
    os.makedirs("ModelTraining", exist_ok=True)
    joblib.dump(model, "ModelTraining/burnout_model.pkl")

    return model, X_test, y_test
