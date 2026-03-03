import pandas as pd
import numpy as np
import joblib
import os
import re

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier


#LOAD DATA


df = pd.read_csv("diabetic_data.csv")

#TARGET


df["readmitted"] = df["readmitted"].apply(lambda x: 0 if x == "NO" else 1)


#DROP ID COLUMNS


df.drop(["encounter_id", "patient_nbr"], axis=1, inplace=True)


#CLEAN DATA


df.replace("?", np.nan, inplace=True)

# Convert age range to numeric midpoint
df["age"] = df["age"].str.extract(r'(\d+)').astype(float) + 5

# Separate numeric and categorical
numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns
categorical_cols = df.select_dtypes(include=["object", "string"]).columns

df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
df[categorical_cols] = df[categorical_cols].fillna(df[categorical_cols].mode().iloc[0])


#ONE HOT ENCODING


df = pd.get_dummies(df, drop_first=True)


#  FULL SAFE COLUMN CLEANING 


def clean_column(name):
   
    name = re.sub(r'[^A-Za-z0-9_]', '_', name)
    return name

df.columns = [clean_column(col) for col in df.columns]


#SPLIT


X = df.drop("readmitted", axis=1)
y = df["readmitted"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


#HANDLE IMBALANCE


scale_pos_weight = len(y_train[y_train == 0]) / len(y_train[y_train == 1])


#TRAIN MODEL


model = XGBClassifier(
    n_estimators=1200,
    max_depth=5,
    learning_rate=0.02,
    subsample=0.8,
    colsample_bytree=0.8,
    gamma=1,
    min_child_weight=3,
    reg_alpha=1,
    reg_lambda=2,
    scale_pos_weight=scale_pos_weight,
    random_state=42,
    eval_metric="logloss"
)

model.fit(X_train, y_train)


#EVALUATE


y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("\n🔥 High Accuracy Model:", round(accuracy * 100, 2), "%")
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))


#  SAVE MODEL


os.makedirs("model", exist_ok=True)

joblib.dump(model, "model/readmission_model.pkl")
joblib.dump(X.columns.tolist(), "model/full_feature_list.pkl")
joblib.dump(df.mode().iloc[0], "model/default_values.pkl")

print("\n✅ Model saved successfully!")

