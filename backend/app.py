from flask import Flask, request, jsonify
import joblib
import numpy as np
import sqlite3
import os
import secrets
from werkzeug.security import generate_password_hash, check_password_hash
from functools import wraps

app = Flask(__name__)
active_tokens = {}

# ── 12 FEATURES the model was trained on (order matters!) ─────────────────
FEATURES = [
    'age',               # numeric (patient age as midpoint: 5,15,25...95)
    'time_in_hospital',  # 1-14 days
    'num_lab_procedures',# 1-132
    'num_procedures',    # 0-6
    'num_medications',   # 1-81
    'number_outpatient', # 0-42
    'number_emergency',  # 0-76
    'number_inpatient',  # 0-21
    'number_diagnoses',  # 1-16
    'insulin',           # encoded: Down=0, No=1, Steady=2, Up=3
    'change',            # encoded: Ch=0, No=1
    'diabetesMed',       # encoded: No=0, Yes=1
]

# ── LOAD MODEL ────────────────────────────────────────────────────────────
model_paths = [
    "readmission_model_12f.pkl",
    "readmission_model.pkl",
    os.path.join(os.path.dirname(__file__), "readmission_model_12f.pkl"),
]
model = None
for path in model_paths:
    try:
        if os.path.exists(path):
            model = joblib.load(path)
            n = getattr(model, 'n_features_in_', '?')
            print(f"✅ Model loaded: {path}  (expects {n} features)")
            break
    except Exception as e:
        print(f"   skip {path}: {e}")

if model is None:
    print("⚠️  Using fallback dummy model")
    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(np.zeros((2, 12)), [0, 1])


# ── DB INIT ───────────────────────────────────────────────────────────────
def init_db():
    conn = sqlite3.connect("database.db")
    c = conn.cursor()
    c.execute("""CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE, password TEXT, role TEXT,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP)""")
    c.execute("""CREATE TABLE IF NOT EXISTS predictions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER,
        age INTEGER, time_in_hospital INTEGER, num_lab_procedures INTEGER,
        num_procedures INTEGER, num_medications INTEGER,
        number_outpatient INTEGER, number_emergency INTEGER,
        number_inpatient INTEGER, number_diagnoses INTEGER,
        insulin TEXT, change_med TEXT, diabetes_med TEXT,
        result TEXT, probability REAL,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY(user_id) REFERENCES users(id))""")
    # migrations for existing DBs
    for sql in [
        "ALTER TABLE predictions ADD COLUMN probability REAL",
        "ALTER TABLE predictions ADD COLUMN created_at DATETIME DEFAULT CURRENT_TIMESTAMP",
        "ALTER TABLE predictions ADD COLUMN age INTEGER",
        "ALTER TABLE predictions ADD COLUMN num_procedures INTEGER",
        "ALTER TABLE predictions ADD COLUMN number_outpatient INTEGER",
        "ALTER TABLE predictions ADD COLUMN number_emergency INTEGER",
        "ALTER TABLE predictions ADD COLUMN number_diagnoses INTEGER",
        "ALTER TABLE predictions ADD COLUMN insulin TEXT",
        "ALTER TABLE predictions ADD COLUMN change_med TEXT",
        "ALTER TABLE predictions ADD COLUMN diabetes_med TEXT",
    ]:
        try: c.execute(sql)
        except: pass
    c.execute("SELECT id FROM users WHERE username='admin'")
    if not c.fetchone():
        c.execute("INSERT INTO users (username,password,role) VALUES (?,?,?)",
                  ("admin", generate_password_hash("admin123"), "admin"))
        print("✅ Default admin: admin / admin123")
    conn.commit(); conn.close()


# ── CORS ──────────────────────────────────────────────────────────────────
@app.after_request
def add_cors(response):
    origin = request.headers.get("Origin", "")
    response.headers["Access-Control-Allow-Origin"] = origin if origin else "*"
    response.headers["Access-Control-Allow-Credentials"] = "true"
    response.headers["Access-Control-Allow-Methods"] = "GET,POST,PUT,DELETE,OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type,Authorization,X-Auth-Token"
    return response

@app.before_request
def handle_preflight():
    if request.method == "OPTIONS":
        origin = request.headers.get("Origin", "*")
        resp = app.make_response("")
        resp.status_code = 200
        resp.headers["Access-Control-Allow-Origin"] = origin if origin else "*"
        resp.headers["Access-Control-Allow-Credentials"] = "true"
        resp.headers["Access-Control-Allow-Methods"] = "GET,POST,PUT,DELETE,OPTIONS"
        resp.headers["Access-Control-Allow-Headers"] = "Content-Type,Authorization,X-Auth-Token"
        resp.headers["Access-Control-Max-Age"] = "86400"
        return resp


# ── AUTH ──────────────────────────────────────────────────────────────────
def require_auth(f):
    @wraps(f)
    def wrapped(*args, **kwargs):
        token = request.headers.get("X-Auth-Token")
        if not token or token not in active_tokens:
            return jsonify({"error": "Unauthorized"}), 401
        request.user_data = active_tokens[token]
        return f(*args, **kwargs)
    return wrapped

def require_admin(f):
    @wraps(f)
    def wrapped(*args, **kwargs):
        token = request.headers.get("X-Auth-Token")
        if not token or token not in active_tokens:
            return jsonify({"error": "Unauthorized"}), 401
        if active_tokens[token]["role"] != "admin":
            return jsonify({"error": "Admin only"}), 403
        request.user_data = active_tokens[token]
        return f(*args, **kwargs)
    return wrapped


# ── ROUTES ────────────────────────────────────────────────────────────────
@app.route("/")
def home():
    return jsonify({"status": "ok", "features": len(FEATURES)})

@app.route("/register", methods=["POST"])
def register():
    d = request.get_json() or {}
    if not d.get("username") or not d.get("password"):
        return jsonify({"message": "Username and password required"}), 400
    conn = sqlite3.connect("database.db"); c = conn.cursor()
    try:
        c.execute("INSERT INTO users (username,password,role) VALUES (?,?,?)",
                  (d["username"], generate_password_hash(d["password"]), d.get("role","user")))
        conn.commit()
        return jsonify({"message": "User registered successfully"})
    except sqlite3.IntegrityError:
        return jsonify({"message": "Username already exists"}), 400
    finally: conn.close()

@app.route("/login", methods=["POST"])
def login():
    d = request.get_json() or {}
    username = d.get("username","").strip()
    password = d.get("password","")
    conn = sqlite3.connect("database.db"); c = conn.cursor()
    c.execute("SELECT id,password,role FROM users WHERE username=?", (username,))
    user = c.fetchone(); conn.close()
    if user and check_password_hash(user[1], password):
        token = secrets.token_hex(32)
        active_tokens[token] = {"user_id": user[0], "username": username, "role": user[2]}
        return jsonify({"message":"Login successful","token":token,"role":user[2],"username":username})
    return jsonify({"message": "Invalid credentials"}), 401

@app.route("/logout", methods=["POST"])
@require_auth
def logout():
    active_tokens.pop(request.headers.get("X-Auth-Token"), None)
    return jsonify({"message": "Logged out"})

@app.route("/users")
@require_admin
def get_users():
    conn = sqlite3.connect("database.db"); c = conn.cursor()
    c.execute("SELECT id,username,role FROM users ORDER BY id")
    users = [{"id":u[0],"username":u[1],"role":u[2]} for u in c.fetchall()]
    conn.close()
    return jsonify({"users": users})

@app.route("/delete-user/<int:uid>", methods=["DELETE"])
@require_admin
def delete_user(uid):
    conn = sqlite3.connect("database.db"); c = conn.cursor()
    c.execute("SELECT username FROM users WHERE id=?", (uid,))
    u = c.fetchone()
    if not u: conn.close(); return jsonify({"message":"Not found"}), 404
    if u[0] == "admin": conn.close(); return jsonify({"message":"Cannot delete admin"}), 400
    c.execute("DELETE FROM predictions WHERE user_id=?", (uid,))
    c.execute("DELETE FROM users WHERE id=?", (uid,))
    conn.commit(); conn.close()
    return jsonify({"message": "User deleted successfully"})


@app.route("/predict", methods=["POST"])
@require_auth
def predict():
    d = request.get_json() or {}

    # Categorical encodings (must match training)
    insulin_map  = {"Down": 0, "No": 1, "Steady": 2, "Up": 3}
    change_map   = {"Ch": 0, "No": 1}
    diabetes_map = {"No": 0, "Yes": 1}

    try:
        age_val      = float(d.get("age", 55))
        insulin_str  = d.get("insulin", "No")
        change_str   = d.get("change", "No")
        diabetes_str = d.get("diabetesMed", "Yes")

        X = np.array([[
            age_val,
            float(d.get("time_in_hospital", 3)),
            float(d.get("num_lab_procedures", 40)),
            float(d.get("num_procedures", 1)),
            float(d.get("num_medications", 15)),
            float(d.get("number_outpatient", 0)),
            float(d.get("number_emergency", 0)),
            float(d.get("number_inpatient", 0)),
            float(d.get("number_diagnoses", 7)),
            float(insulin_map.get(insulin_str, 1)),
            float(change_map.get(change_str, 1)),
            float(diabetes_map.get(diabetes_str, 1)),
        ]])

        pred   = int(model.predict(X)[0])
        result = "High Risk" if pred == 1 else "Low Risk"
        prob   = None
        if hasattr(model, "predict_proba"):
            prob = round(float(max(model.predict_proba(X)[0])) * 100, 1)

        conn = sqlite3.connect("database.db"); c = conn.cursor()
        c.execute("""INSERT INTO predictions
            (user_id, age, time_in_hospital, num_lab_procedures, num_procedures,
             num_medications, number_outpatient, number_emergency, number_inpatient,
             number_diagnoses, insulin, change_med, diabetes_med, result, probability)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (request.user_data["user_id"], age_val,
             d.get("time_in_hospital"), d.get("num_lab_procedures"), d.get("num_procedures"),
             d.get("num_medications"), d.get("number_outpatient"), d.get("number_emergency"),
             d.get("number_inpatient"), d.get("number_diagnoses"),
             insulin_str, change_str, diabetes_str, result, prob))
        conn.commit(); conn.close()

        print(f"✅ {request.user_data['username']}: {result} ({prob}%)")
        return jsonify({"prediction": result, "probability": prob})

    except Exception as e:
        print(f"❌ Predict error: {e}")
        import traceback; traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/history")
@require_auth
def history():
    conn = sqlite3.connect("database.db"); c = conn.cursor()
    c.execute("""SELECT age, time_in_hospital, num_lab_procedures, num_procedures,
                        num_medications, number_outpatient, number_emergency,
                        number_inpatient, number_diagnoses, insulin, change_med,
                        diabetes_med, result, probability, created_at
                 FROM predictions WHERE user_id=? ORDER BY id DESC LIMIT 100""",
              (request.user_data["user_id"],))
    rows = c.fetchall(); conn.close()
    keys = ["age","time_in_hospital","num_lab_procedures","num_procedures",
            "num_medications","number_outpatient","number_emergency","number_inpatient",
            "number_diagnoses","insulin","change","diabetesMed","result","probability","created_at"]
    return jsonify({"history": [dict(zip(keys, r)) for r in rows]})


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
        c.execute("SELECT DATE(created_at),COUNT(*) FROM predictions WHERE created_at>=DATE('now','-7 days') GROUP BY DATE(created_at) ORDER BY DATE(created_at)")
    else:
        c.execute("SELECT COUNT(*) FROM predictions WHERE user_id=?", (uid,)); total = c.fetchone()[0]
        c.execute("SELECT COUNT(*) FROM predictions WHERE user_id=? AND result='High Risk'", (uid,)); high = c.fetchone()[0]
        c.execute("SELECT COUNT(*) FROM predictions WHERE user_id=? AND result='Low Risk'", (uid,)); low = c.fetchone()[0]
        au = 1
        c.execute("SELECT DATE(created_at),COUNT(*) FROM predictions WHERE user_id=? AND created_at>=DATE('now','-7 days') GROUP BY DATE(created_at) ORDER BY DATE(created_at)", (uid,))
    trend = [{"date": r[0], "count": r[1]} for r in c.fetchall()]
    conn.close()
    return jsonify({"total":total,"high":high,"low":low,"active_users":au,"trend":trend})


@app.route("/model-metrics")
@require_auth
def model_metrics():
    return jsonify({
        "accuracy": 0.624, "precision": 0.62, "recall": 0.62,
        "f1_score": 0.62,  "auc_roc": 0.67,
        "confusion_matrix": {"tn": 7570, "fp": 3403, "fn": 4222, "tp": 5159},
        "roc_curve": [
            {"fpr":0.0,"tpr":0.0},{"fpr":0.05,"tpr":0.18},{"fpr":0.10,"tpr":0.32},
            {"fpr":0.15,"tpr":0.43},{"fpr":0.20,"tpr":0.51},{"fpr":0.25,"tpr":0.57},
            {"fpr":0.30,"tpr":0.62},{"fpr":0.40,"tpr":0.71},{"fpr":0.50,"tpr":0.78},
            {"fpr":0.60,"tpr":0.84},{"fpr":0.70,"tpr":0.89},{"fpr":0.80,"tpr":0.93},
            {"fpr":0.90,"tpr":0.97},{"fpr":1.0,"tpr":1.0}
        ],
        "feature_importance": [
            {"feature":"Prior Inpatient Visits", "importance":0.364},
            {"feature":"Number of Diagnoses",    "importance":0.096},
            {"feature":"Prior ER Visits",         "importance":0.091},
            {"feature":"Num Medications",         "importance":0.084},
            {"feature":"Lab Procedures",          "importance":0.079},
            {"feature":"Prior Outpatient Visits", "importance":0.076},
            {"feature":"Age",                     "importance":0.064},
            {"feature":"Days in Hospital",        "importance":0.052},
            {"feature":"Num Procedures",          "importance":0.041},
            {"feature":"Insulin Use",             "importance":0.023},
            {"feature":"On Diabetes Med",         "importance":0.022},
            {"feature":"Med Change",              "importance":0.008},
        ]
    })


if __name__ == "__main__":
    init_db()

    print("\n" + "="*60)
    print(" 🏥  Healthcare Readmission Prediction API")
    print(" --------------------------------------------------")
    print(" ✅ Status  : RUNNING")
    print(" 🌐 URL     : http://localhost:5000")
    print(" 🤖 Model   : XGBoost (Full Feature Model)")
    print(" 🔐 Admin   : admin / admin123")
    print("="*60 + "\n")

    app.run(host="0.0.0.0", port=5000, debug=True)
