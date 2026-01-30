from flask import Flask, request, jsonify
import joblib
import numpy as np
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

model = joblib.load("model/readmission_model.pkl")

@app.route("/")
def home():
    return "Healthcare Readmission Prediction API is running"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    features = np.array([[data['time_in_hospital'],
                           data['num_lab_procedures'],
                           data['num_medications'],
                           data['number_inpatient']]])
    prediction = model.predict(features)[0]
    return jsonify({
        "prediction": "High Risk" if prediction == 1 else "Low Risk"
    })

app.run(debug=True)