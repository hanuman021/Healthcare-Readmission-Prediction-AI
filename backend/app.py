from flask import Flask, request, jsonify
import joblib
import numpy as np
import sqlite3
import os
import secrets
import random
import string
from datetime import datetime
from werkzeug.security import generate_password_hash, check_password_hash
from functools import wraps

app = Flask(__name__)
active_tokens = {}

FEATURES = [
    'age','time_in_hospital','num_lab_procedures','num_procedures',
    'num_medications','number_outpatient','number_emergency',
    'number_inpatient','number_diagnoses','insulin','change','diabetesMed',
]

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_paths = [
    os.path.join(BASE_DIR,"model","readmission_model_12f.pkl"),
    os.path.join(BASE_DIR,"model","readmission_model.pkl"),
    os.path.join(BASE_DIR,"readmission_model_12f.pkl"),
    os.path.join(BASE_DIR,"readmission_model.pkl"),
]
model = None
for path in model_paths:
    try:
        if os.path.exists(path):
            model = joblib.load(path)
            print(f"✅ Model loaded: {path} (expects {getattr(model,'n_features_in_','?')} features)")
            break
    except Exception as e:
        print(f"⚠️ Skip {path}: {e}")

if model is None:
    print("⚠️ Using clinical scoring fallback")

def clinical_score_predict(X_row):
    (age,time_hosp,num_lab,num_proc,num_med,
     n_out,n_er,n_inp,n_diag,insulin_enc,change_enc,diabmed_enc) = X_row
    score = 0.0
    if n_inp == 0: score += 0.0
    elif n_inp == 1: score += 0.18
    elif n_inp == 2: score += 0.28
    elif n_inp == 3: score += 0.36
    else: score += min(0.50, 0.36 + (n_inp-3)*0.04)
    score += 0.10 if n_diag >= 9 else (0.06 if n_diag >= 6 else 0.02)
    score += 0.09 if n_er >= 3 else (0.07 if n_er == 2 else (0.04 if n_er == 1 else 0))
    score += 0.08 if num_med >= 20 else (0.05 if num_med >= 12 else 0.01)
    score += 0.07 if num_lab >= 60 else (0.04 if num_lab >= 40 else 0.01)
    score += 0.06 if n_out >= 5 else (0.03 if n_out >= 2 else 0)
    score += 0.07 if age >= 75 else (0.05 if age >= 60 else (0.03 if age >= 45 else 0.01))
    score += 0.05 if time_hosp >= 10 else (0.03 if time_hosp >= 6 else 0.01)
    score += 0.04 if num_proc >= 4 else (0.02 if num_proc >= 2 else 0)
    score += 0.03 if insulin_enc in (0,3) else (0.01 if insulin_enc == 2 else 0)
    if diabmed_enc == 1: score += 0.02
    if change_enc == 0:  score += 0.01
    prob = max(0.05, min(0.95, score))
    return (1 if prob >= 0.50 else 0), round(prob*100,1)

def generate_patient_id():
    letters = ''.join(random.choices(string.ascii_uppercase, k=2))
    numbers = ''.join(random.choices(string.digits, k=4))
    return f"PAT-{letters}{numbers}"

def log_activity(user_id, username, action, details=""):
    try:
        ip = request.remote_addr if request else ""
        conn = sqlite3.connect("database.db"); c = conn.cursor()
        c.execute("INSERT INTO activity_log (user_id,username,action,details,ip) VALUES (?,?,?,?,?)",
                  (user_id,username,action,details,ip))
        conn.commit(); conn.close()
    except: pass

def init_db():
    conn = sqlite3.connect("database.db")
    c = conn.cursor()
    c.execute("""CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE, password TEXT, role TEXT,
        email TEXT, created_at DATETIME DEFAULT CURRENT_TIMESTAMP, last_login DATETIME)""")
    c.execute("""CREATE TABLE IF NOT EXISTS predictions (
        id INTEGER PRIMARY KEY AUTOINCREMENT, user_id INTEGER,
        patient_name TEXT, patient_id TEXT,
        age INTEGER, time_in_hospital INTEGER, num_lab_procedures INTEGER,
        num_procedures INTEGER, num_medications INTEGER,
        number_outpatient INTEGER, number_emergency INTEGER,
        number_inpatient INTEGER, number_diagnoses INTEGER,
        insulin TEXT, change_med TEXT, diabetes_med TEXT,
        result TEXT, probability REAL, notes TEXT,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY(user_id) REFERENCES users(id))""")
    c.execute("""CREATE TABLE IF NOT EXISTS activity_log (
        id INTEGER PRIMARY KEY AUTOINCREMENT, user_id INTEGER,
        username TEXT, action TEXT, details TEXT, ip TEXT,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP)""")
    for sql in [
        "ALTER TABLE predictions ADD COLUMN patient_name TEXT",
        "ALTER TABLE predictions ADD COLUMN patient_id TEXT",
        "ALTER TABLE predictions ADD COLUMN notes TEXT",
        "ALTER TABLE predictions ADD COLUMN probability REAL",
        "ALTER TABLE predictions ADD COLUMN age INTEGER",
        "ALTER TABLE predictions ADD COLUMN num_procedures INTEGER",
        "ALTER TABLE predictions ADD COLUMN number_outpatient INTEGER",
        "ALTER TABLE predictions ADD COLUMN number_emergency INTEGER",
        "ALTER TABLE predictions ADD COLUMN number_diagnoses INTEGER",
        "ALTER TABLE predictions ADD COLUMN insulin TEXT",
        "ALTER TABLE predictions ADD COLUMN change_med TEXT",
        "ALTER TABLE predictions ADD COLUMN diabetes_med TEXT",
        "ALTER TABLE users ADD COLUMN email TEXT",
        "ALTER TABLE users ADD COLUMN last_login DATETIME",
    ]:
        try: c.execute(sql)
        except: pass
    c.execute("SELECT id FROM users WHERE username='admin'")
    if not c.fetchone():
        c.execute("INSERT INTO users (username,password,role) VALUES (?,?,?)",
                  ("admin", generate_password_hash("admin123"), "admin"))
        print("✅ Default admin: admin / admin123")
    conn.commit(); conn.close()

@app.after_request
def add_cors(response):
    origin = request.headers.get("Origin","")
    response.headers["Access-Control-Allow-Origin"] = origin if origin else "*"
    response.headers["Access-Control-Allow-Credentials"] = "true"
    response.headers["Access-Control-Allow-Methods"] = "GET,POST,PUT,DELETE,OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type,Authorization,X-Auth-Token"
    return response

@app.before_request
def handle_preflight():
    if request.method == "OPTIONS":
        origin = request.headers.get("Origin","*")
        resp = app.make_response(""); resp.status_code = 200
        resp.headers["Access-Control-Allow-Origin"] = origin
        resp.headers["Access-Control-Allow-Credentials"] = "true"
        resp.headers["Access-Control-Allow-Methods"] = "GET,POST,PUT,DELETE,OPTIONS"
        resp.headers["Access-Control-Allow-Headers"] = "Content-Type,Authorization,X-Auth-Token"
        resp.headers["Access-Control-Max-Age"] = "86400"
        return resp

def require_auth(f):
    @wraps(f)
    def wrapped(*args,**kwargs):
        token = request.headers.get("X-Auth-Token")
        if not token or token not in active_tokens:
            return jsonify({"error":"Unauthorized"}), 401
        request.user_data = active_tokens[token]
        return f(*args,**kwargs)
    return wrapped

def require_admin(f):
    @wraps(f)
    def wrapped(*args,**kwargs):
        token = request.headers.get("X-Auth-Token")
        if not token or token not in active_tokens:
            return jsonify({"error":"Unauthorized"}), 401
        if active_tokens[token]["role"] != "admin":
            return jsonify({"error":"Admin only"}), 403
        request.user_data = active_tokens[token]
        return f(*args,**kwargs)
    return wrapped

@app.route("/")
def home():
    return jsonify({"status":"ok","model":"loaded" if model else "fallback","version":"3.0"})

@app.route("/register", methods=["POST"])
def register():
    d = request.get_json() or {}
    if not d.get("username") or not d.get("password"):
        return jsonify({"message":"Username and password required"}), 400
    conn = sqlite3.connect("database.db"); c = conn.cursor()
    try:
        c.execute("INSERT INTO users (username,password,role,email) VALUES (?,?,?,?)",
                  (d["username"],generate_password_hash(d["password"]),d.get("role","user"),d.get("email","")))
        conn.commit()
        return jsonify({"message":"User registered successfully"})
    except sqlite3.IntegrityError:
        return jsonify({"message":"Username already exists"}), 400
    finally: conn.close()

@app.route("/login", methods=["POST"])
def login():
    d = request.get_json() or {}
    username = d.get("username","").strip()
    password = d.get("password","")
    conn = sqlite3.connect("database.db"); c = conn.cursor()
    c.execute("SELECT id,password,role FROM users WHERE username=?",(username,))
    user = c.fetchone()
    if user and check_password_hash(user[1],password):
        token = secrets.token_hex(32)
        active_tokens[token] = {"user_id":user[0],"username":username,"role":user[2]}
        c.execute("UPDATE users SET last_login=? WHERE id=?",(datetime.now(),user[0]))
        conn.commit(); conn.close()
        log_activity(user[0],username,"LOGIN",f"Role:{user[2]}")
        return jsonify({"message":"Login successful","token":token,"role":user[2],"username":username})
    conn.close()
    return jsonify({"message":"Invalid credentials"}), 401

@app.route("/logout", methods=["POST"])
@require_auth
def logout():
    log_activity(request.user_data["user_id"],request.user_data["username"],"LOGOUT")
    active_tokens.pop(request.headers.get("X-Auth-Token"),None)
    return jsonify({"message":"Logged out"})

@app.route("/users")
@require_admin
def get_users():
    conn = sqlite3.connect("database.db"); c = conn.cursor()
    c.execute("""SELECT id,username,role,email,created_at,last_login,
                 (SELECT COUNT(*) FROM predictions WHERE user_id=users.id) as pred_count
                 FROM users ORDER BY id""")
    users = [{"id":u[0],"username":u[1],"role":u[2],"email":u[3] or "",
               "created_at":u[4] or "","last_login":u[5] or "","predictions":u[6]}
             for u in c.fetchall()]
    conn.close()
    return jsonify({"users":users})

@app.route("/delete-user/<int:uid>", methods=["DELETE"])
@require_admin
def delete_user(uid):
    conn = sqlite3.connect("database.db"); c = conn.cursor()
    c.execute("SELECT username FROM users WHERE id=?",(uid,))
    u = c.fetchone()
    if not u: conn.close(); return jsonify({"message":"Not found"}), 404
    if u[0] == "admin": conn.close(); return jsonify({"message":"Cannot delete admin"}), 400
    c.execute("DELETE FROM predictions WHERE user_id=?",(uid,))
    c.execute("DELETE FROM users WHERE id=?",(uid,))
    conn.commit(); conn.close()
    log_activity(request.user_data["user_id"],request.user_data["username"],"DELETE_USER",f"Deleted:{u[0]}")
    return jsonify({"message":"User deleted successfully"})

@app.route("/update-user/<int:uid>", methods=["PUT"])
@require_admin
def update_user(uid):
    d = request.get_json() or {}
    conn = sqlite3.connect("database.db"); c = conn.cursor()
    c.execute("SELECT username FROM users WHERE id=?",(uid,))
    u = c.fetchone()
    if not u: conn.close(); return jsonify({"message":"Not found"}), 404
    updates,params = [],[]
    if "role" in d: updates.append("role=?"); params.append(d["role"])
    if "email" in d: updates.append("email=?"); params.append(d["email"])
    if "password" in d and d["password"]:
        updates.append("password=?"); params.append(generate_password_hash(d["password"]))
    if not updates: conn.close(); return jsonify({"message":"Nothing to update"}), 400
    params.append(uid)
    c.execute(f"UPDATE users SET {', '.join(updates)} WHERE id=?",params)
    conn.commit(); conn.close()
    return jsonify({"message":"User updated successfully"})

@app.route("/predict", methods=["POST"])
@require_auth
def predict():
    d = request.get_json() or {}
    insulin_map  = {"Down":0,"No":1,"Steady":2,"Up":3}
    change_map   = {"Ch":0,"No":1}
    diabetes_map = {"No":0,"Yes":1}
    try:
        patient_name = d.get("patient_name","").strip() or "Unknown Patient"
        patient_id   = generate_patient_id()
        age_val      = float(d.get("age",55))
        insulin_str  = d.get("insulin","No")
        change_str   = d.get("change","No")
        diabetes_str = d.get("diabetesMed","Yes")
        notes        = d.get("notes","").strip()
        X_row = [
            age_val,
            float(d.get("time_in_hospital",3)),
            float(d.get("num_lab_procedures",40)),
            float(d.get("num_procedures",1)),
            float(d.get("num_medications",15)),
            float(d.get("number_outpatient",0)),
            float(d.get("number_emergency",0)),
            float(d.get("number_inpatient",0)),
            float(d.get("number_diagnoses",7)),
            float(insulin_map.get(insulin_str,1)),
            float(change_map.get(change_str,1)),
            float(diabetes_map.get(diabetes_str,1)),
        ]
        if model is not None:
            X = np.array([X_row])
            pred = int(model.predict(X)[0])
            result = "High Risk" if pred == 1 else "Low Risk"
            prob = None
            if hasattr(model,"predict_proba"):
                prob = round(float(max(model.predict_proba(X)[0]))*100,1)
        else:
            pred,prob = clinical_score_predict(X_row)
            result = "High Risk" if pred == 1 else "Low Risk"
        conn = sqlite3.connect("database.db"); c = conn.cursor()
        c.execute("""INSERT INTO predictions
            (user_id,patient_name,patient_id,age,time_in_hospital,num_lab_procedures,
             num_procedures,num_medications,number_outpatient,number_emergency,
             number_inpatient,number_diagnoses,insulin,change_med,diabetes_med,
             result,probability,notes)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (request.user_data["user_id"],patient_name,patient_id,age_val,
             d.get("time_in_hospital"),d.get("num_lab_procedures"),d.get("num_procedures"),
             d.get("num_medications"),d.get("number_outpatient"),d.get("number_emergency"),
             d.get("number_inpatient"),d.get("number_diagnoses"),
             insulin_str,change_str,diabetes_str,result,prob,notes))
        conn.commit(); conn.close()
        log_activity(request.user_data["user_id"],request.user_data["username"],
                     "PREDICT",f"{patient_name}({patient_id})→{result}")
        return jsonify({"prediction":result,"probability":prob,"patient_id":patient_id})
    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({"error":str(e)}), 500

@app.route("/history")
@require_auth
def history():
    conn = sqlite3.connect("database.db"); c = conn.cursor()
    uid = request.user_data["user_id"]
    search = request.args.get("search","").strip()
    filter_result = request.args.get("result","").strip()
    query = """SELECT patient_name,patient_id,age,time_in_hospital,num_lab_procedures,
                      num_procedures,num_medications,number_outpatient,number_emergency,
                      number_inpatient,number_diagnoses,insulin,change_med,
                      diabetes_med,result,probability,notes,created_at
               FROM predictions WHERE user_id=?"""
    params = [uid]
    if search:
        query += " AND (patient_name LIKE ? OR patient_id LIKE ?)"
        params += [f"%{search}%",f"%{search}%"]
    if filter_result:
        query += " AND result=?"; params.append(filter_result)
    query += " ORDER BY id DESC LIMIT 200"
    c.execute(query,params); rows = c.fetchall(); conn.close()
    keys = ["patient_name","patient_id","age","time_in_hospital","num_lab_procedures",
            "num_procedures","num_medications","number_outpatient","number_emergency",
            "number_inpatient","number_diagnoses","insulin","change","diabetesMed",
            "result","probability","notes","created_at"]
    return jsonify({"history":[dict(zip(keys,r)) for r in rows]})

@app.route("/all-patients")
@require_admin
def all_patients():
    conn = sqlite3.connect("database.db"); c = conn.cursor()
    search = request.args.get("search","").strip()
    filter_result = request.args.get("result","").strip()
    query = """SELECT p.patient_name,p.patient_id,p.age,p.result,p.probability,
                      p.created_at,u.username,p.notes,p.number_inpatient,p.number_emergency,p.number_diagnoses
               FROM predictions p JOIN users u ON p.user_id=u.id"""
    params = []; conditions = []
    if search:
        conditions.append("(p.patient_name LIKE ? OR p.patient_id LIKE ? OR u.username LIKE ?)")
        params += [f"%{search}%",f"%{search}%",f"%{search}%"]
    if filter_result:
        conditions.append("p.result=?"); params.append(filter_result)
    if conditions: query += " WHERE "+" AND ".join(conditions)
    query += " ORDER BY p.id DESC LIMIT 500"
    c.execute(query,params); rows = c.fetchall(); conn.close()
    keys = ["patient_name","patient_id","age","result","probability","created_at",
            "clinician","notes","inpatient","emergency","diagnoses"]
    return jsonify({"patients":[dict(zip(keys,r)) for r in rows]})

@app.route("/cases-by-risk")
@require_auth
def cases_by_risk():
    risk = request.args.get("risk","")
    if risk not in ("High Risk","Low Risk"):
        return jsonify({"error":"Invalid risk"}), 400
    conn = sqlite3.connect("database.db"); c = conn.cursor()
    uid = request.user_data["user_id"]
    is_admin = request.user_data["role"] == "admin"
    if is_admin:
        c.execute("""SELECT p.patient_name,p.patient_id,p.age,p.result,p.probability,
                            p.created_at,u.username
                     FROM predictions p JOIN users u ON p.user_id=u.id
                     WHERE p.result=? ORDER BY p.id DESC LIMIT 100""",(risk,))
    else:
        c.execute("""SELECT patient_name,patient_id,age,result,probability,
                            created_at,? as username
                     FROM predictions WHERE user_id=? AND result=?
                     ORDER BY id DESC LIMIT 100""",(request.user_data["username"],uid,risk))
    rows = c.fetchall(); conn.close()
    keys = ["patient_name","patient_id","age","result","probability","created_at","clinician"]
    return jsonify({"cases":[dict(zip(keys,r)) for r in rows]})

@app.route("/dashboard-data")
@require_auth
def dashboard_data():
    conn = sqlite3.connect("database.db"); c = conn.cursor()
    uid = request.user_data["user_id"]
    is_admin = request.user_data["role"] == "admin"
    if is_admin:
        c.execute("SELECT COUNT(*) FROM predictions"); total = c.fetchone()[0]
        c.execute("SELECT COUNT(*) FROM predictions WHERE result='High Risk'"); high = c.fetchone()[0]
        c.execute("SELECT COUNT(*) FROM predictions WHERE result='Low Risk'"); low = c.fetchone()[0]
        c.execute("SELECT COUNT(DISTINCT user_id) FROM predictions"); au = c.fetchone()[0]
        c.execute("SELECT COUNT(*) FROM users WHERE role='user'"); clinicians = c.fetchone()[0]
        c.execute("""SELECT DATE(created_at),COUNT(*) FROM predictions
                     WHERE created_at>=DATE('now','-7 days')
                     GROUP BY DATE(created_at) ORDER BY DATE(created_at)""")
    else:
        c.execute("SELECT COUNT(*) FROM predictions WHERE user_id=?",(uid,)); total = c.fetchone()[0]
        c.execute("SELECT COUNT(*) FROM predictions WHERE user_id=? AND result='High Risk'",(uid,)); high = c.fetchone()[0]
        c.execute("SELECT COUNT(*) FROM predictions WHERE user_id=? AND result='Low Risk'",(uid,)); low = c.fetchone()[0]
        au=1; clinicians=1
        c.execute("""SELECT DATE(created_at),COUNT(*) FROM predictions
                     WHERE user_id=? AND created_at>=DATE('now','-7 days')
                     GROUP BY DATE(created_at) ORDER BY DATE(created_at)""",(uid,))
    trend = [{"date":r[0],"count":r[1]} for r in c.fetchall()]
    conn.close()
    return jsonify({"total":total,"high":high,"low":low,"active_users":au,"clinicians":clinicians,"trend":trend})

@app.route("/activity-log")
@require_admin
def activity_log():
    conn = sqlite3.connect("database.db"); c = conn.cursor()
    c.execute("""SELECT username,action,details,ip,created_at
                 FROM activity_log ORDER BY id DESC LIMIT 200""")
    rows = c.fetchall(); conn.close()
    keys = ["username","action","details","ip","created_at"]
    return jsonify({"logs":[dict(zip(keys,r)) for r in rows]})

@app.route("/system-stats")
@require_admin
def system_stats():
    conn = sqlite3.connect("database.db"); c = conn.cursor()
    c.execute("SELECT COUNT(*) FROM users"); total_users = c.fetchone()[0]
    c.execute("SELECT COUNT(*) FROM predictions"); total_preds = c.fetchone()[0]
    c.execute("SELECT COUNT(*) FROM predictions WHERE DATE(created_at)=DATE('now')"); today = c.fetchone()[0]
    c.execute("SELECT COUNT(*) FROM predictions WHERE result='High Risk'"); high = c.fetchone()[0]
    c.execute("""SELECT u.username,COUNT(p.id) as cnt FROM users u
                 LEFT JOIN predictions p ON u.id=p.user_id
                 GROUP BY u.id ORDER BY cnt DESC LIMIT 5""")
    top_users = [{"username":r[0],"predictions":r[1]} for r in c.fetchall()]
    c.execute("""SELECT DATE(created_at),COUNT(*),SUM(CASE WHEN result='High Risk' THEN 1 ELSE 0 END)
                 FROM predictions WHERE created_at>=DATE('now','-30 days')
                 GROUP BY DATE(created_at) ORDER BY DATE(created_at)""")
    monthly = [{"date":r[0],"total":r[1],"high_risk":r[2]} for r in c.fetchall()]
    conn.close()
    return jsonify({"total_users":total_users,"total_predictions":total_preds,
                    "today_predictions":today,"high_risk":high,
                    "top_users":top_users,"monthly_trend":monthly})

@app.route("/model-metrics")
@require_auth
def model_metrics():
    return jsonify({
        "model_type":"XGBoost" if model else "Clinical Scoring Fallback",
        "accuracy":0.624,"precision":0.62,"recall":0.62,"f1_score":0.62,"auc_roc":0.67,
        "confusion_matrix":{"tn":7570,"fp":3403,"fn":4222,"tp":5159},
        "roc_curve":[
            {"fpr":0.0,"tpr":0.0},{"fpr":0.05,"tpr":0.18},{"fpr":0.10,"tpr":0.32},
            {"fpr":0.15,"tpr":0.43},{"fpr":0.20,"tpr":0.51},{"fpr":0.25,"tpr":0.57},
            {"fpr":0.30,"tpr":0.62},{"fpr":0.40,"tpr":0.71},{"fpr":0.50,"tpr":0.78},
            {"fpr":0.60,"tpr":0.84},{"fpr":0.70,"tpr":0.89},{"fpr":0.80,"tpr":0.93},
            {"fpr":0.90,"tpr":0.97},{"fpr":1.0,"tpr":1.0}
        ],
        "feature_importance":[
            {"feature":"Prior Inpatient Visits","importance":0.364},
            {"feature":"Number of Diagnoses","importance":0.096},
            {"feature":"Prior ER Visits","importance":0.091},
            {"feature":"Num Medications","importance":0.084},
            {"feature":"Lab Procedures","importance":0.079},
            {"feature":"Prior Outpatient Visits","importance":0.076},
            {"feature":"Age","importance":0.064},
            {"feature":"Days in Hospital","importance":0.052},
            {"feature":"Num Procedures","importance":0.041},
            {"feature":"Insulin Use","importance":0.023},
            {"feature":"On Diabetes Med","importance":0.022},
            {"feature":"Med Change","importance":0.008},
        ]
    })

if __name__ == "__main__":
    init_db()
    print("\n"+"="*60)
    print(" 🏥  Healthcare Readmission Prediction API v3.0")
    print(f" 🤖 Model   : {'XGBoost (trained)' if model else 'Clinical Scoring Fallback'}")
    print(" 🔐 Admin   : admin / admin123")
    print("="*60+"\n")
    #app.run(host="0.0.0.0",port=5000,debug=True)
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
  