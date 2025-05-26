# train_model.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pickle

# Load the burnout dataset
df = pd.read_csv("ModelTraining/burnout_data.csv")

# Separate features and target
X = df[["hours_worked", "sleep_hours", "mood_score"]]
y = df["burnout"]

# Split into training and test sets — 80/20 split is standard
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the logistic regression model
model = LogisticRegression()

# Train the model
model.fit(X_train, y_train)

# Evaluate performance
accuracy = model.score(X_test, y_test)
print(f"Model accuracy: {accuracy:.2f}")

# Save the trained model to the root so Docker/Flask can access it easily
with open("burnout_model.pkl", "wb") as f:
    pickle.dump(model, f)