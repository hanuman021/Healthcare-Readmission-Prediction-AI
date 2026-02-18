import pandas as pd
import numpy as np
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from sklearn.utils import class_weight
from xgboost import XGBClassifier

# =============================
# 1️⃣ Load Dataset
# =============================

df = pd.read_csv("diabetic_data.csv")

# =============================
# 2️⃣ Target Conversion
# =============================

# Convert readmission to binary
df["readmitted"] = df["readmitted"].apply(lambda x: 0 if x == "NO" else 1)

# =============================
# 3️⃣ Drop Unnecessary Columns
# =============================

df.drop(["encounter_id", "patient_nbr"], axis=1, inplace=True)

# =============================
# 4️⃣ Handle Missing Values
# =============================

# Handle missing values
df.replace("?", np.nan, inplace=True)
df = df.ffill()


# =============================
# 5️⃣ Convert Age to Numeric
# =============================

df["age"] = df["age"].str.extract(r'(\d+)').astype(int)

# =============================
# 6️⃣ Encode Categorical Columns
# =============================

for col in df.select_dtypes(include="object").columns:
    df[col] = LabelEncoder().fit_transform(df[col])

# =============================
# 7️⃣ Split Features & Target
# =============================

X = df.drop("readmitted", axis=1)
y = df["readmitted"]

# =============================
# 8️⃣ Train-Test Split
# =============================

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# =============================
# 9️⃣ Class Weight (Imbalance Fix)
# =============================

weights = class_weight.compute_class_weight(
    class_weight="balanced",
    classes=np.unique(y_train),
    y=y_train
)

weight_dict = {0: weights[0], 1: weights[1]}

sample_weights = y_train.map(weight_dict)

# =============================
# 🔟 XGBoost Model
# =============================

model = XGBClassifier(
    n_estimators=500,
    max_depth=8,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    eval_metric="logloss"
)

model.fit(X_train, y_train, sample_weight=sample_weights)

# =============================
# 1️⃣1️⃣ Evaluation
# =============================

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("🔥 Final Model Accuracy:", accuracy)

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# =============================
# 1️⃣2️⃣ Save Model
# =============================

os.makedirs("model", exist_ok=True)
joblib.dump(model, "model/readmission_model.pkl")

print("\n✅ Model saved successfully!")
