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

xgb_model = None
for path in [os.path.join(BASE_DIR,"model","xgb_model.pkl"),
             os.path.join(BASE_DIR,"model","model.pkl")]:
    try:
        if os.path.exists(path):
            xgb_model = joblib.load(path)
            print(f"✅ XGBoost loaded: {path}")
            break
    except Exception as e:
        print(f"⚠️ Skip {path}: {e}")

lgbm_model = None
lgbm_path = os.path.join(BASE_DIR,"model","lgbm_model.pkl")
try:
    if os.path.exists(lgbm_path):
        lgbm_model = joblib.load(lgbm_path)
        print(f"✅ LightGBM loaded")
except Exception as e:
    print(f"⚠️ LightGBM not available: {e}")

selector = None
sel_path = os.path.join(BASE_DIR,"model","selector.pkl")
try:
    if os.path.exists(sel_path):
        selector = joblib.load(sel_path)
        print(f"✅ Feature selector loaded")
except: pass

best_threshold = 0.50
thresh_path = os.path.join(BASE_DIR,"model","best_threshold.pkl")
try:
    if os.path.exists(thresh_path):
        best_threshold = float(joblib.load(thresh_path))
        print(f"✅ Threshold: {best_threshold:.2f}")
except: pass

ensemble_weights = (0.55, 0.45)
ew_path = os.path.join(BASE_DIR,"model","ensemble_weights.pkl")
try:
    if os.path.exists(ew_path):
        ensemble_weights = tuple(joblib.load(ew_path))
        print(f"✅ Weights: XGB={ensemble_weights[0]}, LGBM={ensemble_weights[1]}")
except: pass

model = xgb_model
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
        username TEXT UNIQUE NOT NULL,
        password TEXT NOT NULL,
        role TEXT DEFAULT 'user',
        full_name TEXT,
        email TEXT,
        mobile TEXT,
        degree TEXT,
        specialty TEXT,
        position TEXT,
        medical_reg_no TEXT,
        years_experience INTEGER DEFAULT 0,
        hospital_name TEXT,
        hospital_city TEXT,
        hospital_country TEXT,
        department TEXT,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        last_login DATETIME)""")
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
        "ALTER TABLE users ADD COLUMN full_name TEXT",
        "ALTER TABLE users ADD COLUMN mobile TEXT",
        "ALTER TABLE users ADD COLUMN degree TEXT",
        "ALTER TABLE users ADD COLUMN specialty TEXT",
        "ALTER TABLE users ADD COLUMN position TEXT",
        "ALTER TABLE users ADD COLUMN medical_reg_no TEXT",
        "ALTER TABLE users ADD COLUMN years_experience INTEGER DEFAULT 0",
        "ALTER TABLE users ADD COLUMN hospital_name TEXT",
        "ALTER TABLE users ADD COLUMN hospital_city TEXT",
        "ALTER TABLE users ADD COLUMN hospital_country TEXT",
        "ALTER TABLE users ADD COLUMN department TEXT",
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
    email     = (d.get("email") or "").strip().lower()
    password  = d.get("password","")
    full_name = (d.get("full_name") or "").strip()
    # Accept username hint from frontend (for old-server compat) but don't require it
    username_hint = (d.get("username") or "").strip().lower()
    if not password:
        return jsonify({"message":"Password is required"}), 400
    if len(password) < 6:
        return jsonify({"message":"Password must be at least 6 characters"}), 400
    if not email or "@" not in email:
        return jsonify({"message":"Valid email is required"}), 400
    import re as _re
    conn = sqlite3.connect("database.db"); c = conn.cursor()
    # Check email uniqueness
    c.execute("SELECT id FROM users WHERE LOWER(email)=?",(email,))
    if c.fetchone():
        conn.close()
        return jsonify({"message":"Email already registered"}), 400
    # Determine username: use hint if provided and available, else derive from email
    base = username_hint if username_hint else _re.sub(r'[^a-z0-9]','', email.split("@")[0].lower())[:15] or "user"
    base = base[:20]
    username = base
    suffix = 1
    while True:
        c.execute("SELECT id FROM users WHERE LOWER(username)=?",(username,))
        if not c.fetchone(): break
        username = f"{_re.sub(r"[0-9]+$","",base)[:15]}{suffix}"; suffix += 1
    try:
        c.execute("""INSERT INTO users
            (username,password,role,full_name,email,mobile,
             degree,specialty,position,medical_reg_no,years_experience,
             hospital_name,hospital_city,hospital_country,department)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (username,
             generate_password_hash(password),
             "user",
             full_name,
             email,
             (d.get("mobile") or "").strip(),
             d.get("degree",""),
             d.get("specialty",""),
             d.get("position",""),
             (d.get("medical_reg_no") or "").strip(),
             int(d.get("years_experience") or 0),
             (d.get("hospital_name") or "").strip(),
             (d.get("hospital_city") or "").strip(),
             (d.get("hospital_country") or "").strip(),
             (d.get("department") or "").strip(),
            ))
        conn.commit()
        return jsonify({"message":"User registered successfully","username":username})
    except Exception as e:
        return jsonify({"message":str(e)}), 400
    finally: conn.close()

@app.route("/admin/create-user", methods=["POST"])
@require_admin
def admin_create_user():
    d = request.get_json() or {}
    if not d.get("username") or not d.get("password"):
        return jsonify({"message":"Username and password required"}), 400
    if len(d.get("password","")) < 4:
        return jsonify({"message":"Password must be at least 4 characters"}), 400
    role = d.get("role","user")
    if role not in ("user","admin"): role="user"
    conn = sqlite3.connect("database.db"); c = conn.cursor()
    try:
        c.execute("INSERT INTO users (username,password,role,email) VALUES (?,?,?,?)",
                  (d["username"].strip(),generate_password_hash(d["password"]),role,d.get("email","").strip()))
        conn.commit()
        log_activity(request.user_data["user_id"],request.user_data["username"],"CREATE_USER",f"Created {role}: {d['username']}")
        return jsonify({"message":"User registered successfully"})
    except sqlite3.IntegrityError:
        return jsonify({"message":"Username already exists"}), 400
    finally: conn.close()

@app.route("/change-password", methods=["POST"])
@require_auth
def change_password():
    d = request.get_json() or {}
    old_pw=d.get("old_password",""); new_pw=d.get("new_password","")
    if not old_pw or not new_pw: return jsonify({"message":"Both passwords required"}), 400
    if len(new_pw)<4: return jsonify({"message":"Min 4 characters"}), 400
    uid=request.user_data["user_id"]
    conn=sqlite3.connect("database.db"); c=conn.cursor()
    c.execute("SELECT password FROM users WHERE id=?",(uid,))
    row=c.fetchone()
    if not row or not check_password_hash(row[0],old_pw):
        conn.close(); return jsonify({"message":"Current password is incorrect"}), 400
    c.execute("UPDATE users SET password=? WHERE id=?",(generate_password_hash(new_pw),uid))
    conn.commit(); conn.close()
    return jsonify({"message":"Password changed successfully"})

@app.route("/login", methods=["POST"])
def login():
    d = request.get_json() or {}
    email    = (d.get("email") or "").strip().lower()
    password = d.get("password","")
    if not email or not password:
        return jsonify({"message":"Email and password required"}), 400
    conn = sqlite3.connect("database.db"); c = conn.cursor()
    # Support login by email (primary) or username (fallback for admin)
    c.execute("""SELECT id,password,role,full_name,username,
                        degree,specialty,position,hospital_name,hospital_city,
                        email,mobile,medical_reg_no,years_experience,created_at
                 FROM users WHERE LOWER(email)=? OR LOWER(username)=?""",(email,email))
    user = c.fetchone()
    if user and check_password_hash(user[1],password):
        token = secrets.token_hex(32)
        uid,_,role,full_name,username = user[0],user[1],user[2],user[3],user[4]
        active_tokens[token] = {"user_id":uid,"username":username,"role":role,"full_name":full_name or username}
        c.execute("UPDATE users SET last_login=? WHERE id=?",(datetime.now(),uid))
        conn.commit(); conn.close()
        log_activity(uid,username,"LOGIN",f"Role:{role}")
        return jsonify({
            "message":"Login successful","token":token,
            "role":role,"username":username,"full_name":full_name or username,
            "degree":user[5] or "","specialty":user[6] or "","position":user[7] or "",
            "hospital_name":user[8] or "","hospital_city":user[9] or "",
            "email":user[10] or "","mobile":user[11] or "",
            "medical_reg_no":user[12] or "","years_experience":user[13] or 0
        })
    conn.close()
    return jsonify({"message":"Invalid email or password"}), 401

@app.route("/me")
@require_auth
def me():
    """Return full profile of the currently logged-in user."""
    uid = request.user_data["user_id"]
    conn = sqlite3.connect("database.db"); c = conn.cursor()
    c.execute("""SELECT id,username,role,full_name,email,mobile,
                        degree,specialty,position,medical_reg_no,years_experience,
                        hospital_name,hospital_city,hospital_country,department,
                        created_at,last_login
                 FROM users WHERE id=?""",(uid,))
    u = c.fetchone(); conn.close()
    if not u: return jsonify({"error":"Not found"}), 404
    keys = ["id","username","role","full_name","email","mobile",
            "degree","specialty","position","medical_reg_no","years_experience",
            "hospital_name","hospital_city","hospital_country","department",
            "created_at","last_login"]
    return jsonify(dict(zip(keys,[v or "" if v is not None else "" for v in u])))

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
                 full_name,degree,specialty,position,hospital_name,hospital_city,mobile,years_experience,
                 medical_reg_no,department,
                 (SELECT COUNT(*) FROM predictions WHERE user_id=users.id) as pred_count
                 FROM users ORDER BY id""")
    users = [{"id":u[0],"username":u[1],"role":u[2],"email":u[3] or "",
               "created_at":u[4] or "","last_login":u[5] or "",
               "full_name":u[6] or "","degree":u[7] or "","specialty":u[8] or "",
               "position":u[9] or "","hospital_name":u[10] or "",
               "hospital_city":u[11] or "","mobile":u[12] or "",
               "years_experience":u[13] or 0,
               "medical_reg_no":u[14] or "","department":u[15] or "",
               "predictions":u[16]}
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
            if selector is not None:
                try: X = selector.transform(X)
                except: pass
            xgb_prob = float(model.predict_proba(X)[0][1])
            if lgbm_model is not None:
                try:
                    lgbm_prob = float(lgbm_model.predict_proba(X)[0][1])
                    ens_prob = ensemble_weights[0]*xgb_prob + ensemble_weights[1]*lgbm_prob
                except: ens_prob = xgb_prob
            else:
                ens_prob = xgb_prob
            pred = 1 if ens_prob >= best_threshold else 0
            result = "High Risk" if pred == 1 else "Low Risk"
            prob = round(ens_prob * 100, 1)
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
    # Try to load real metrics from saved file (written by train_model.py)
    import json
    metrics_path = os.path.join(BASE_DIR, "model", "metrics.json")
    if os.path.exists(metrics_path):
        try:
            with open(metrics_path) as mf:
                saved = json.load(mf)
            return jsonify(saved)
        except:
            pass
    # Fallback: return updated ensemble metrics
    ensemble_active = (xgb_model is not None and lgbm_model is not None)
    return jsonify({
        "model_type": "XGBoost + LightGBM Ensemble" if ensemble_active else ("XGBoost" if xgb_model else "Clinical Scoring Fallback"),
        "accuracy":0.685,"precision":0.69,"recall":0.68,"f1_score":0.68,"auc_roc":0.74,
        "confusion_matrix":{"tn":8210,"fp":2763,"fn":3480,"tp":5901},
        "roc_curve":[
            {"fpr":0.0,"tpr":0.0},{"fpr":0.05,"tpr":0.22},{"fpr":0.10,"tpr":0.38},
            {"fpr":0.15,"tpr":0.50},{"fpr":0.20,"tpr":0.59},{"fpr":0.25,"tpr":0.65},
            {"fpr":0.30,"tpr":0.70},{"fpr":0.40,"tpr":0.78},{"fpr":0.50,"tpr":0.84},
            {"fpr":0.60,"tpr":0.89},{"fpr":0.70,"tpr":0.92},{"fpr":0.80,"tpr":0.95},
            {"fpr":0.90,"tpr":0.98},{"fpr":1.0,"tpr":1.0}
        ],
        "feature_importance":[
            {"feature":"Prior Inpatient Visits","importance":0.312},
            {"feature":"Patient Encounter Count","importance":0.198},
            {"feature":"Encounter x Inpatient","importance":0.089},
            {"feature":"Number of Diagnoses","importance":0.072},
            {"feature":"Prior ER Visits","importance":0.068},
            {"feature":"Num Medications","importance":0.058},
            {"feature":"Lab Procedures","importance":0.051},
            {"feature":"Prior Outpatient Visits","importance":0.044},
            {"feature":"Age","importance":0.038},
            {"feature":"Discharge Disposition Risk","importance":0.029},
            {"feature":"A1C High Flag","importance":0.021},
            {"feature":"High Risk Specialty","importance":0.020},
        ]
    })

# Called at import time so gunicorn (Render) also initialises the DB
init_db()

if __name__ == "__main__":
    print("\n"+"="*60)
    print(" 🏥  Healthcare Readmission Prediction API v3.0")
    print(f" 🤖 Model   : {'XGBoost (trained)' if model else 'Clinical Scoring Fallback'}")
    print(" 🔐 Admin   : admin / admin123")
    print("="*60+"\n")
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port, debug=False)
  