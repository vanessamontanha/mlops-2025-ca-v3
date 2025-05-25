# train_model.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pickle

# Load the burnout dataset
# We're assuming the CSV has 3 features: hours_worked, sleep_hours, mood_score, and a binary target: burnout
df = pd.read_csv("ModelTraining/burnout_data.csv")

# Separate features and target
X = df[["hours_worked", "sleep_hours", "mood_score"]]  # independent variables
y = df["burnout"]  # target variable (0 or 1)

# Split into training and test sets — 80/20 split is standard
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the logistic regression model — simple and interpretable for binary classification
model = LogisticRegression()

# Fit model on training data
model.fit(X_train, y_train)

# Evaluate the model quickly to check overfitting or major issues
accuracy = model.score(X_test, y_test)
print(f"Model accuracy: {accuracy:.2f}")

# Save the trained model to a pickle file so the Flask app can load it later
with open("ModelTraining/burnout_model.pkl", "wb") as f:
    pickle.dump(model, f)
