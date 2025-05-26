# app.py – Flask app to serve predictions from the trained burnout model

from flask import Flask, request, jsonify
import pickle
import pandas as pd

# Initialize the Flask app
app = Flask(__name__)

# Load the trained model from disk
# This assumes the .pkl file is present in the expected path
with open("burnout_model.pkl", "rb") as f:
    model = pickle.load(f)

# Define the prediction route
@app.route("/predict", methods=["POST"])
def predict():
    """
    Expects JSON data with keys: hours_worked, sleep_hours, mood_score
    Returns predicted burnout risk (0 = no burnout, 1 = burnout)
    """
    data = request.get_json()

    # Validate input format
    if not all(k in data for k in ("hours_worked", "sleep_hours", "mood_score")):
        return jsonify({"error": "Missing fields in input"}), 400

    # Format input into a DataFrame (as the model expects)
    input_df = pd.DataFrame([{
        "hours_worked": data["hours_worked"],
        "sleep_hours": data["sleep_hours"],
        "mood_score": data["mood_score"]
    }])

    # Make prediction
    prediction = model.predict(input_df)[0]

    # Return result
    return jsonify({"burnout_risk": int(prediction)})

# Simple test route
@app.route("/", methods=["GET"])
def index():
    return "Burnout prediction API is live."

# Allow running the app directly
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
