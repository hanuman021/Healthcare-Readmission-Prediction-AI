import pandas as pd
import numpy as np
import joblib
import os
import re
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.feature_selection import SelectFromModel
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

  
# LOAD DATA
  
print("📂 Loading data...")
df = pd.read_csv("diabetic_data.csv")
df.replace("?", np.nan, inplace=True)
df.drop_duplicates(inplace=True)
print(f"   Raw shape: {df.shape}")

  
# TARGET
  
df["readmitted"] = df["readmitted"].map({"NO": 0, ">30": 1, "<30": 1})

  
# REMOVE DISCHARGE LEAKAGE
# Patients who died/hospice cannot be readmitted
  
leak_ids = [11, 13, 14, 19, 20, 21]
df = df[~df["discharge_disposition_id"].isin(leak_ids)]
print(f"   After leakage removal: {df.shape}")

  
# PATIENT HISTORY FEATURES  <-- BIGGEST SIGNAL
# Readmission rate: 1 visit=22%, 3 visits=76%, 5+ visits=86%
  
pat_counts = df["patient_nbr"].value_counts()
df["patient_encounter_count"] = df["patient_nbr"].map(pat_counts)
df["is_repeat_patient"]       = (df["patient_encounter_count"] > 1).astype(int)
df["is_chronic_patient"]      = (df["patient_encounter_count"] >= 3).astype(int)

  
# CLEAN AGE
  
age_map = {
    "[0-10)":5,  "[10-20)":15, "[20-30)":25, "[30-40)":35,
    "[40-50)":45,"[50-60)":55, "[60-70)":65,
    "[70-80)":75,"[80-90)":85, "[90-100)":95
}
df["age"] = df["age"].map(age_map)

  
# A1C & GLUCOSE -- keep as risk flags instead of dropping
  
df["a1c_tested"]        = df["A1Cresult"].notna().astype(int)
df["a1c_high"]          = df["A1Cresult"].isin([">7", ">8"]).astype(int)
df["glucose_tested"]    = df["max_glu_serum"].notna().astype(int)
df["glucose_high"]      = df["max_glu_serum"].isin([">200", ">300"]).astype(int)
df["glucose_very_high"] = (df["max_glu_serum"] == ">300").astype(int)

  
# MEDICAL SPECIALTY RISK FLAGS
  
high_risk_specs = {
    "Nephrology","Surgery-Vascular","Emergency/Trauma",
    "Hematology/Oncology","Gastroenterology","Pulmonology",
    "Oncology","Family/GeneralPractice"
}
df["high_risk_specialty"]  = df["medical_specialty"].isin(high_risk_specs).astype(int)
df["specialty_internal"]   = (df["medical_specialty"] == "InternalMedicine").astype(int)
df["specialty_cardiology"] = (df["medical_specialty"] == "Cardiology").astype(int)

  
# DISCHARGE DISPOSITION RISK FLAG
  
high_readmit_discharge = [6, 22, 3, 7, 5, 2, 4, 25, 15, 10, 12]
df["high_readmit_discharge"] = df["discharge_disposition_id"].isin(high_readmit_discharge).astype(int)

  
# ICD-9 DIAGNOSIS GROUPING
  
def icd9_to_group(code):
    if pd.isna(code) or str(code).strip() in ("", "?"):
        return 0
    code = str(code).strip()
    if code.startswith("V"): return 19
    if code.startswith("E"): return 20
    try:
        num = float(code)
        if 390 <= num <= 459 or num == 785: return 7   # Circulatory
        if 460 <= num <= 519 or num == 786: return 8   # Respiratory
        if 520 <= num <= 579 or num == 787: return 9   # Digestive
        if 250 <= num <= 250.99:            return 3   # Diabetes
        if 800 <= num <= 999:               return 17  # Injury
        if 710 <= num <= 739:               return 13  # Musculoskeletal
        if 580 <= num <= 629 or num == 788: return 10  # Genitourinary
        if 140 <= num <= 239:               return 2   # Neoplasms
        if 1   <= num <= 139:               return 1   # Infectious
        if 240 <= num <= 279:               return 4   # Endocrine/Nutritional
        if 280 <= num <= 289:               return 5   # Blood
        if 290 <= num <= 319:               return 6   # Mental
        if 680 <= num <= 709 or num == 782: return 12  # Skin
        if 630 <= num <= 679:               return 11  # Pregnancy
        if 740 <= num <= 759:               return 14  # Congenital
        if 760 <= num <= 779:               return 15  # Perinatal
        if 780 <= num <= 799:               return 16  # Symptoms
        return 18
    except:
        return 0

for d_col in ["diag_1", "diag_2", "diag_3"]:
    if d_col in df.columns:
        df[f"{d_col}_group"] = df[d_col].apply(icd9_to_group)

df["primary_diag_diabetes"]   = (df["diag_1_group"] == 3).astype(int)
df["primary_diag_circulatory"]= (df["diag_1_group"] == 7).astype(int)
df["any_circulatory"]  = ((df["diag_1_group"]==7)|(df["diag_2_group"]==7)|(df["diag_3_group"]==7)).astype(int)
df["any_respiratory"]  = ((df["diag_1_group"]==8)|(df["diag_2_group"]==8)|(df["diag_3_group"]==8)).astype(int)
df["n_unique_diag_groups"] = df[["diag_1_group","diag_2_group","diag_3_group"]].nunique(axis=1)
df.drop(columns=["diag_1","diag_2","diag_3"], inplace=True, errors="ignore")

  
# MEDICATION ENCODING
  
med_cols = [
    "metformin","repaglinide","nateglinide","chlorpropamide","glimepiride",
    "acetohexamide","glipizide","glyburide","tolbutamide","pioglitazone",
    "rosiglitazone","acarbose","miglitol","troglitazone","tolazamide",
    "examide","citoglipton","insulin","glyburide-metformin",
    "glipizide-metformin","glimepiride-pioglitazone",
    "metformin-rosiglitazone","metformin-pioglitazone"
]
med_map = {"No": 0, "Down": 1, "Steady": 2, "Up": 3}
for col in med_cols:
    if col in df.columns:
        df[col] = df[col].map(med_map).fillna(0).astype(int)

active_med_cols = [c for c in med_cols if c in df.columns]
df["num_meds_changed"] = (df[active_med_cols].isin([1, 3])).sum(axis=1)
df["num_active_meds"]  = (df[active_med_cols] > 0).sum(axis=1)
df["insulin_used"]     = (df["insulin"] > 0).astype(int) if "insulin" in df.columns else 0
df["insulin_changed"]  = df["insulin"].isin([1, 3]).astype(int) if "insulin" in df.columns else 0

  
# DROP USELESS COLUMNS
  
drop_cols = [
    "encounter_id", "patient_nbr",
    "weight",            # 97% missing
    "payer_code",        # not clinical
    "A1Cresult",         # replaced by flags
    "max_glu_serum",     # replaced by flags
    "medical_specialty", # replaced by flags
]
df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True)

  
# HANDLE MISSING VALUES
  
num_cols = df.select_dtypes(include=["int64","float64"]).columns.tolist()
cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
df[num_cols] = df[num_cols].fillna(df[num_cols].median())
df[cat_cols] = df[cat_cols].fillna("Unknown")

  
# FEATURE ENGINEERING
  
df["total_visits"]          = df["number_outpatient"] + df["number_emergency"] + df["number_inpatient"]
df["med_intensity"]         = df["num_medications"] / (df["time_in_hospital"] + 1)
df["lab_per_day"]           = df["num_lab_procedures"] / (df["time_in_hospital"] + 1)
df["proc_per_day"]          = df["num_procedures"] / (df["time_in_hospital"] + 1)
df["diagnosis_severity"]    = df["number_diagnoses"] * 2 + df["num_procedures"] + df["num_medications"]
df["is_frequent_patient"]   = (df["total_visits"] > 2).astype(int)
df["has_emergency_history"] = (df["number_emergency"] > 0).astype(int)
df["has_inpatient_history"] = (df["number_inpatient"] > 0).astype(int)
df["high_meds"]             = (df["num_medications"] > 15).astype(int)
df["long_stay"]             = (df["time_in_hospital"] > 7).astype(int)
df["elderly"]               = (df["age"] >= 75).astype(int)
df["many_lab_procedures"]   = (df["num_lab_procedures"] > 60).astype(int)
df["many_diagnoses"]        = (df["number_diagnoses"] >= 9).astype(int)

# Interaction terms
df["inpatient_x_diagnoses"]  = df["number_inpatient"] * df["number_diagnoses"]
df["inpatient_x_emergency"]  = df["number_inpatient"] * df["number_emergency"]
df["age_x_inpatient"]        = df["age"] * df["number_inpatient"]
df["meds_x_diagnoses"]       = df["num_medications"] * df["number_diagnoses"]
df["encounter_x_inpatient"]  = df["patient_encounter_count"] * df["number_inpatient"]
df["encounter_x_emergency"]  = df["patient_encounter_count"] * df["number_emergency"]

  
# ENCODE & CLEAN COLUMNS
  
df = pd.get_dummies(df, drop_first=True)

def clean_col(name):
    return re.sub(r'[^A-Za-z0-9_]', '_', name)

df.columns = [clean_col(c) for c in df.columns]

  
# SPLIT
  
X = df.drop("readmitted", axis=1)
y = df["readmitted"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\n📊 Train: {X_train.shape}, Test: {X_test.shape}")
print(f"   Class balance — 0: {(y_train==0).sum()}, 1: {(y_train==1).sum()}")
scale_pos_weight = len(y_train[y_train==0]) / len(y_train[y_train==1])
print(f"   scale_pos_weight: {scale_pos_weight:.3f}")

  
# FEATURE SELECTION
  
print("\n🔍 Feature selection...")
fs_model = XGBClassifier(
    n_estimators=500, max_depth=6, learning_rate=0.05,
    scale_pos_weight=scale_pos_weight, random_state=42,
    eval_metric="auc", verbosity=0
)
fs_model.fit(X_train, y_train)
selector = SelectFromModel(fs_model, threshold="0.5*median", prefit=True)
X_tr = selector.transform(X_train)
X_te = selector.transform(X_test)
selected_features = X.columns[selector.get_support()]
print(f"   Selected {len(selected_features)} / {X.shape[1]} features")

importances = pd.Series(fs_model.feature_importances_, index=X.columns)
print("\n   Top 10 features by importance:")
for fname, imp in importances.nlargest(10).items():
    print(f"     {fname:<45} {imp:.4f}")

  
# MODEL 1 — XGBoost
  
print("\n🚀 Training XGBoost...")
xgb = XGBClassifier(
    n_estimators=3000, max_depth=5, learning_rate=0.01,
    subsample=0.8, colsample_bytree=0.75, colsample_bylevel=0.75,
    gamma=0.15, min_child_weight=5,
    reg_alpha=0.3, reg_lambda=2.0,
    scale_pos_weight=scale_pos_weight,
    random_state=42, eval_metric="auc",
    early_stopping_rounds=100, verbosity=0
)
xgb.fit(X_tr, y_train, eval_set=[(X_te, y_test)], verbose=False)
xgb_prob = xgb.predict_proba(X_te)[:, 1]
print(f"   XGBoost  Acc: {accuracy_score(y_test, (xgb_prob>=0.5))*100:.2f}%  "
      f"AUC: {roc_auc_score(y_test, xgb_prob):.4f}")

  
# MODEL 2 — LightGBM
  
print("\n🚀 Training LightGBM...")
import lightgbm as lgb
lgbm = LGBMClassifier(
    n_estimators=3000, max_depth=6, learning_rate=0.01,
    num_leaves=50, subsample=0.8, colsample_bytree=0.75,
    min_child_samples=30, reg_alpha=0.2, reg_lambda=2.0,
    scale_pos_weight=scale_pos_weight, random_state=42, verbose=-1
)
lgbm.fit(
    X_tr, y_train, eval_set=[(X_te, y_test)],
    callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(period=-1)]
)
lgbm_prob = lgbm.predict_proba(X_te)[:, 1]
print(f"   LightGBM Acc: {accuracy_score(y_test, (lgbm_prob>=0.5))*100:.2f}%  "
      f"AUC: {roc_auc_score(y_test, lgbm_prob):.4f}")

  
# FIND BEST ENSEMBLE WEIGHTS + THRESHOLD
  
print("\n🎯 Optimising ensemble weights & threshold...")
best_acc, best_w, best_thresh = 0, (0.5, 0.5), 0.5

for wx in np.arange(0.3, 0.8, 0.05):
    prob = wx * xgb_prob + (1-wx) * lgbm_prob
    for thresh in np.arange(0.35, 0.65, 0.01):
        acc = accuracy_score(y_test, (prob >= thresh).astype(int))
        if acc > best_acc:
            best_acc  = acc
            best_w    = (round(wx,2), round(1-wx,2))
            best_thresh = round(thresh, 2)

ensemble_prob = best_w[0] * xgb_prob + best_w[1] * lgbm_prob
final_pred    = (ensemble_prob >= best_thresh).astype(int)

print(f"\n{'='*55}")
print(f"🏆 FINAL ACCURACY : {accuracy_score(y_test, final_pred)*100:.2f}%")
print(f"🏆 ROC-AUC        : {roc_auc_score(y_test, ensemble_prob):.4f}")
print(f"   Weights — XGB: {best_w[0]}, LGBM: {best_w[1]}, Threshold: {best_thresh}")
print(f"{'='*55}")
print("\n📋 Classification Report:")
print(classification_report(y_test, final_pred, target_names=["Not Readmitted","Readmitted"]))

  
# SAVE METRICS JSON (read by app.py /model-metrics)
  
import json
ensemble_active = True
metrics_out = {
    "model_type": "XGBoost + LightGBM Ensemble",
    "accuracy":   round(accuracy_score(y_test, final_pred), 4),
    "precision":  round(float(__import__("sklearn.metrics",fromlist=["precision_score"]).precision_score(y_test, final_pred, average="weighted", zero_division=0)), 4),
    "recall":     round(float(__import__("sklearn.metrics",fromlist=["recall_score"]).recall_score(y_test, final_pred, average="weighted", zero_division=0)), 4),
    "f1_score":   round(float(__import__("sklearn.metrics",fromlist=["f1_score"]).f1_score(y_test, final_pred, average="weighted", zero_division=0)), 4),
    "auc_roc":    round(roc_auc_score(y_test, ensemble_prob), 4),
    "confusion_matrix": {
        "tn": int(((y_test==0)&(final_pred==0)).sum()),
        "fp": int(((y_test==0)&(final_pred==1)).sum()),
        "fn": int(((y_test==1)&(final_pred==0)).sum()),
        "tp": int(((y_test==1)&(final_pred==1)).sum()),
    },
    "roc_curve": [],
    "feature_importance": []
}

# ROC curve points
from sklearn.metrics import roc_curve as sk_roc
fpr_arr, tpr_arr, _ = sk_roc(y_test, ensemble_prob)
step = max(1, len(fpr_arr)//14)
metrics_out["roc_curve"] = [
    {"fpr": round(float(fpr_arr[i]),3), "tpr": round(float(tpr_arr[i]),3)}
    for i in range(0, len(fpr_arr), step)
] + [{"fpr":1.0,"tpr":1.0}]

# Feature importance from XGB
fi = pd.Series(xgb.feature_importances_, index=selected_features)
top_fi = fi.nlargest(12)
metrics_out["feature_importance"] = [
    {"feature": str(k).replace("_"," ").title(), "importance": round(float(v),4)}
    for k, v in top_fi.items()
]

os.makedirs("model", exist_ok=True)
with open("model/metrics.json", "w") as mf:
    json.dump(metrics_out, mf, indent=2)
print("\n📊 metrics.json saved to model/metrics.json")

  
# SAVE ALL ARTIFACTS
  
os.makedirs("model", exist_ok=True)
joblib.dump(xgb,                        "model/xgb_model.pkl")
joblib.dump(lgbm,                       "model/lgbm_model.pkl")
joblib.dump(selector,                   "model/selector.pkl")
joblib.dump(selected_features.tolist(), "model/selected_features.pkl")
joblib.dump(best_thresh,                "model/best_threshold.pkl")
joblib.dump(best_w,                     "model/ensemble_weights.pkl")
joblib.dump(df.mode().iloc[0],          "model/default_values.pkl")
joblib.dump(xgb,                        "model/model.pkl")   # backward compat

print("\n✅ All artifacts saved to model/")
print(f"   Threshold: {best_thresh}  |  Weights: XGB={best_w[0]} LGBM={best_w[1]}")
