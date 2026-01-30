import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

print("=== Training started ===")

# Create clean dataset
data = {
    "time_in_hospital": [1,2,3,4,5,6,7,8,9,10],
    "num_lab_procedures": [10,20,30,40,50,60,70,80,90,100],
    "num_medications": [5,6,7,8,9,10,11,12,13,14],
    "number_inpatient": [0,0,1,1,2,2,3,3,4,4],
    "readmitted": [0,0,0,0,0,1,1,1,1,1]
}

df = pd.DataFrame(data)

X = df.drop("readmitted", axis=1)
y = df["readmitted"]

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X, y)

# Ensure folder exists
os.makedirs("model", exist_ok=True)

# Save model
joblib.dump(model, "model/readmission_model.pkl")

print("=== Model saved successfully ===")