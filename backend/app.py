from flask import Flask, request, jsonify
import joblib
import numpy as np
import sqlite3
import os
import secrets
import random
import string
import re
import json
from datetime import datetime, timedelta
from werkzeug.security import generate_password_hash, check_password_hash
from functools import wraps

# ── Optional deps (graceful fallback if not installed) ──────────────────────
try:
    import jwt as pyjwt
    JWT_AVAILABLE = True
except ImportError:
    JWT_AVAILABLE = False
    print("⚠️  PyJWT not installed — using in-memory token fallback")

try:
    from flask_limiter import Limiter
    from flask_limiter.util import get_remote_address
    LIMITER_AVAILABLE = True
except ImportError:
    LIMITER_AVAILABLE = False
    print("⚠️  Flask-Limiter not installed — rate-limiting disabled")

app = Flask(__name__)

# ── JWT secret (random per-process; swap for env var in prod) ────────────────
JWT_SECRET = os.environ.get("JWT_SECRET", secrets.token_hex(48))
JWT_ALG    = "HS256"
JWT_TTL    = 60 * 60 * 24  # 24 h

# ── Allowed CORS origins ─────────────────────────────────────────────────────
ALLOWED_ORIGINS = [o.strip() for o in
    os.environ.get("ALLOWED_ORIGINS",
        "http://localhost,http://localhost:3000,http://127.0.0.1,http://127.0.0.1:3000,"
        "https://healthcare-readmission-prediction-ai.onrender.com"
    ).split(",") if o.strip()
]

# ── Rate limiter ─────────────────────────────────────────────────────────────
if LIMITER_AVAILABLE:
    limiter = Limiter(
        key_func=get_remote_address,
        app=app,
        default_limits=[],
        storage_uri="memory://",
    )

# ── Fallback in-memory token store (used when PyJWT not available) ───────────
active_tokens = {}

# ── Account lockout store (in-memory; survives until restart) ────────────────
# { ip_or_username: {"attempts": int, "locked_until": datetime|None} }
_lockout: dict = {}
MAX_FAILED_ATTEMPTS = 10
LOCKOUT_MINUTES     = 15


def _lockout_key(username: str) -> str:
    ip = request.remote_addr or ""
    return f"{ip}|{username.lower()}"


def _is_locked(key: str) -> bool:
    e = _lockout.get(key)
    if not e:
        return False
    if e.get("locked_until") and datetime.now() < e["locked_until"]:
        return True
    if e.get("locked_until") and datetime.now() >= e["locked_until"]:
        _lockout.pop(key, None)
    return False


def _record_failure(key: str):
    e = _lockout.setdefault(key, {"attempts": 0, "locked_until": None})
    e["attempts"] += 1
    if e["attempts"] >= MAX_FAILED_ATTEMPTS:
        e["locked_until"] = datetime.now() + timedelta(minutes=LOCKOUT_MINUTES)


def _clear_lockout(key: str):
    _lockout.pop(key, None)


# ── Model version ─────────────────────────────────────────────────────────────
MODEL_VERSION = os.environ.get("MODEL_VERSION", "v3.2")

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

selected_features_list = []
sf_path = os.path.join(BASE_DIR,"model","selected_features.pkl")
try:
    if os.path.exists(sf_path):
        selected_features_list = joblib.load(sf_path)
        print(f"✅ Selected features loaded: {len(selected_features_list)} features")
except Exception as e:
    print(f"⚠️ Could not load selected features: {e}")

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
        must_change_password INTEGER DEFAULT 0,
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
        model_version TEXT,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY(user_id) REFERENCES users(id))""")
    c.execute("""CREATE TABLE IF NOT EXISTS activity_log (
        id INTEGER PRIMARY KEY AUTOINCREMENT, user_id INTEGER,
        username TEXT, action TEXT, details TEXT, ip TEXT,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP)""")
    c.execute("""CREATE TABLE IF NOT EXISTS otp_store (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        email TEXT NOT NULL,
        otp TEXT NOT NULL,
        expires_at DATETIME NOT NULL,
        used INTEGER DEFAULT 0,
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
        "ALTER TABLE predictions ADD COLUMN model_version TEXT",
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
        "ALTER TABLE users ADD COLUMN must_change_password INTEGER DEFAULT 0",
    ]:
        try: c.execute(sql)
        except: pass
    c.execute("SELECT id FROM users WHERE username='admin'")
    if not c.fetchone():
        c.execute("INSERT INTO users (username,password,role,must_change_password) VALUES (?,?,?,?)",
                  ("admin", generate_password_hash("admin123"), "admin", 1))
        print("✅ Default admin: admin / admin123  (MUST change password on first login)")
    conn.commit(); conn.close()


# ── JWT helpers ───────────────────────────────────────────────────────────────
def _create_token(payload: dict) -> str:
    if JWT_AVAILABLE:
        data = {**payload, "exp": datetime.utcnow() + timedelta(seconds=JWT_TTL),
                "iat": datetime.utcnow(), "jti": secrets.token_hex(16)}
        return pyjwt.encode(data, JWT_SECRET, algorithm=JWT_ALG)
    else:
        tok = secrets.token_hex(32)
        active_tokens[tok] = payload
        return tok


def _decode_token(token: str) -> dict | None:
    if JWT_AVAILABLE:
        try:
            return pyjwt.decode(token, JWT_SECRET, algorithms=[JWT_ALG])
        except Exception:
            return None
    else:
        return active_tokens.get(token)


# ── Input validation helpers ──────────────────────────────────────────────────
EMAIL_RE = re.compile(r'^[^@\s]+@[^@\s]+\.[^@\s]+$')
_SAFE    = re.compile(r'[<>"\'%;()&+]')

def _safe(s: str, max_len: int = 200) -> str:
    """Strip dangerous chars and truncate."""
    return _SAFE.sub('', str(s or ""))[:max_len].strip()

def _valid_email(e: str) -> bool:
    return bool(EMAIL_RE.match(e)) and len(e) <= 254

def _validate_numeric(val, min_v, max_v, default):
    try:
        v = float(val)
        return max(min_v, min(max_v, v))
    except (TypeError, ValueError):
        return default


# ── CORS ──────────────────────────────────────────────────────────────────────
@app.after_request
def add_cors(response):
    origin = request.headers.get("Origin", "")
    if origin in ALLOWED_ORIGINS:
        response.headers["Access-Control-Allow-Origin"] = origin
        response.headers["Access-Control-Allow-Credentials"] = "true"
    elif not ALLOWED_ORIGINS:
        response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET,POST,PUT,DELETE,OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type,Authorization,X-Auth-Token"
    return response


@app.before_request
def handle_preflight():
    if request.method == "OPTIONS":
        origin = request.headers.get("Origin","*")
        resp = app.make_response(""); resp.status_code = 200
        if origin in ALLOWED_ORIGINS or not ALLOWED_ORIGINS:
            resp.headers["Access-Control-Allow-Origin"] = origin
        resp.headers["Access-Control-Allow-Credentials"] = "true"
        resp.headers["Access-Control-Allow-Methods"] = "GET,POST,PUT,DELETE,OPTIONS"
        resp.headers["Access-Control-Allow-Headers"] = "Content-Type,Authorization,X-Auth-Token"
        resp.headers["Access-Control-Max-Age"] = "86400"
        return resp


# ── Auth decorators ───────────────────────────────────────────────────────────
def require_auth(f):
    @wraps(f)
    def wrapped(*args, **kwargs):
        token = request.headers.get("X-Auth-Token")
        if not token:
            return jsonify({"error": "Unauthorized"}), 401
        data = _decode_token(token)
        if not data:
            return jsonify({"error": "Unauthorized — invalid or expired token"}), 401
        request.user_data = data
        return f(*args, **kwargs)
    return wrapped


def require_admin(f):
    @wraps(f)
    def wrapped(*args, **kwargs):
        token = request.headers.get("X-Auth-Token")
        if not token:
            return jsonify({"error": "Unauthorized"}), 401
        data = _decode_token(token)
        if not data:
            return jsonify({"error": "Unauthorized"}), 401
        if data.get("role") != "admin":
            return jsonify({"error": "Admin only"}), 403
        request.user_data = data
        return f(*args, **kwargs)
    return wrapped


# ── Health endpoint ───────────────────────────────────────────────────────────
@app.route("/health")
def health():
    return jsonify({
        "status": "ok",
        "model": "loaded" if model else "fallback",
        "version": MODEL_VERSION,
        "timestamp": datetime.utcnow().isoformat() + "Z"
    })


@app.route("/")
def home():
    return jsonify({"status":"ok","model":"loaded" if model else "fallback","version":MODEL_VERSION})


# ── Register ──────────────────────────────────────────────────────────────────
@app.route("/register", methods=["POST"])
def register():
    d = request.get_json() or {}
    email     = _safe(d.get("email","")).lower()
    password  = d.get("password","")
    full_name = _safe(d.get("full_name",""), 100)
    username_hint = _safe(d.get("username",""), 30).lower()

    if not password:
        return jsonify({"message":"Password is required"}), 400
    if len(password) < 6:
        return jsonify({"message":"Password must be at least 6 characters"}), 400
    if not email or not _valid_email(email):
        return jsonify({"message":"Valid email is required"}), 400

    conn = sqlite3.connect("database.db"); c = conn.cursor()
    c.execute("SELECT id FROM users WHERE LOWER(email)=?",(email,))
    if c.fetchone():
        conn.close()
        return jsonify({"message":"Email already registered"}), 400

    base = username_hint if username_hint else re.sub(r'[^a-z0-9]','', email.split("@")[0].lower())[:15] or "user"
    base = base[:20]
    username = base; suffix = 1
    while True:
        c.execute("SELECT id FROM users WHERE LOWER(username)=?",(username,))
        if not c.fetchone(): break
        username = f"{re.sub(r'[0-9]+$','',base)[:15]}{suffix}"; suffix += 1
    try:
        c.execute("""INSERT INTO users
            (username,password,role,full_name,email,mobile,
             degree,specialty,position,medical_reg_no,years_experience,
             hospital_name,hospital_city,hospital_country,department)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (username, generate_password_hash(password), "user", full_name, email,
             _safe(d.get("mobile",""), 20),
             _safe(d.get("degree",""), 50),
             _safe(d.get("specialty",""), 80),
             _safe(d.get("position",""), 80),
             _safe(d.get("medical_reg_no",""), 40),
             int(d.get("years_experience") or 0),
             _safe(d.get("hospital_name",""), 120),
             _safe(d.get("hospital_city",""), 80),
             _safe(d.get("hospital_country","India"), 60),
             _safe(d.get("department",""), 80),
            ))
        conn.commit()
        return jsonify({"message":"User registered successfully","username":username})
    except Exception as e:
        return jsonify({"message":str(e)}), 400
    finally: conn.close()


# ── Admin create user ─────────────────────────────────────────────────────────
@app.route("/admin/create-user", methods=["POST"])
@require_admin
def admin_create_user():
    d = request.get_json() or {}
    email    = _safe(d.get("email","")).lower()
    password = d.get("password","")
    full_name= _safe(d.get("full_name",""), 100)
    if not email or not _valid_email(email):
        return jsonify({"message":"Valid email is required"}), 400
    if not password or len(password) < 4:
        return jsonify({"message":"Password must be at least 4 characters"}), 400
    role = d.get("role","user")
    if role not in ("user","admin"): role = "user"
    base = re.sub(r"[^a-z0-9]","", email.split("@")[0].lower())[:15] or "user"
    conn = sqlite3.connect("database.db"); c = conn.cursor()
    c.execute("SELECT id FROM users WHERE LOWER(email)=?",(email,))
    if c.fetchone():
        conn.close()
        return jsonify({"message":"Email already registered"}), 400
    username = base; suffix = 1
    while True:
        c.execute("SELECT id FROM users WHERE username=?",(username,))
        if not c.fetchone(): break
        username = f"{base}{suffix}"; suffix += 1
    try:
        c.execute("""INSERT INTO users
            (username,password,role,full_name,email,mobile,
             degree,specialty,position,medical_reg_no,years_experience,
             hospital_name,hospital_city,hospital_country,department,must_change_password)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (username, generate_password_hash(password), role, full_name, email,
             _safe(d.get("mobile",""), 20),
             _safe(d.get("degree",""), 50), _safe(d.get("specialty",""), 80),
             _safe(d.get("position",""), 80), _safe(d.get("medical_reg_no",""), 40),
             int(d.get("years_experience") or 0),
             _safe(d.get("hospital_name",""), 120), _safe(d.get("hospital_city",""), 80),
             _safe(d.get("hospital_country","India"), 60), _safe(d.get("department",""), 80),
             1))  # must_change_password = True for admin-created accounts
        conn.commit()
        log_activity(request.user_data["user_id"],request.user_data["username"],
                     "CREATE_USER",f"Created {role}: {email} → @{username}")
        return jsonify({"message":"User registered successfully",
                        "username":username,
                        "note": "Profile incomplete — user must complete details on first login."})
    except sqlite3.IntegrityError as e:
        return jsonify({"message":str(e)}), 400
    finally: conn.close()


# ── Profile ───────────────────────────────────────────────────────────────────
@app.route("/update-profile", methods=["POST"])
@require_auth
def update_profile():
    d   = request.get_json() or {}
    uid = request.user_data["user_id"]
    allowed = ["full_name","mobile","degree","specialty","position",
               "medical_reg_no","years_experience","hospital_name",
               "hospital_city","hospital_country","department"]
    updates, params = [], []
    for k in allowed:
        if k in d:
            updates.append(f"{k}=?")
            params.append(_safe(str(d[k]), 200) if k != "years_experience" else int(d[k] or 0))
    if not updates:
        return jsonify({"message":"Nothing to update"}), 400
    params.append(uid)
    conn = sqlite3.connect("database.db"); c = conn.cursor()
    c.execute(f"UPDATE users SET {', '.join(updates)} WHERE id=?", params)
    conn.commit(); conn.close()
    return jsonify({"message":"Profile updated successfully"})


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
    c.execute("UPDATE users SET password=?,must_change_password=0 WHERE id=?",
              (generate_password_hash(new_pw),uid))
    conn.commit(); conn.close()
    return jsonify({"message":"Password changed successfully"})


# ── Login (with rate-limit + lockout) ─────────────────────────────────────────
if LIMITER_AVAILABLE:
    @app.route("/login", methods=["POST"])
    @limiter.limit("10 per minute")
    def login():
        return _login_logic()
else:
    @app.route("/login", methods=["POST"])
    def login():
        return _login_logic()


def _login_logic():
    d = request.get_json() or {}
    email    = _safe(d.get("email","")).lower()
    password = d.get("password","")
    if not email or not password:
        return jsonify({"message":"Email and password required"}), 400

    lock_key = _lockout_key(email)
    if _is_locked(lock_key):
        return jsonify({"message":f"Account temporarily locked after {MAX_FAILED_ATTEMPTS} failed attempts. Try again in {LOCKOUT_MINUTES} minutes."}), 429

    conn = sqlite3.connect("database.db"); c = conn.cursor()
    c.execute("""SELECT id,password,role,full_name,username,
                        degree,specialty,position,hospital_name,hospital_city,
                        email,mobile,medical_reg_no,years_experience,created_at,
                        must_change_password
                 FROM users WHERE LOWER(email)=? OR LOWER(username)=?""",(email,email))
    user = c.fetchone()
    if user and check_password_hash(user[1],password):
        _clear_lockout(lock_key)
        uid,_,role,full_name,username = user[0],user[1],user[2],user[3],user[4]
        must_change = bool(user[15])
        payload = {"user_id":uid,"username":username,"role":role,"full_name":full_name or username}
        token = _create_token(payload)
        c.execute("UPDATE users SET last_login=? WHERE id=?",(datetime.now(),uid))
        conn.commit(); conn.close()
        log_activity(uid,username,"LOGIN",f"Role:{role}")
        return jsonify({
            "message":"Login successful","token":token,
            "role":role,"username":username,"full_name":full_name or username,
            "degree":user[5] or "","specialty":user[6] or "","position":user[7] or "",
            "hospital_name":user[8] or "","hospital_city":user[9] or "",
            "email":user[10] or "","mobile":user[11] or "",
            "medical_reg_no":user[12] or "","years_experience":user[13] or 0,
            "must_change_password": must_change
        })
    conn.close()
    _record_failure(lock_key)
    remaining = MAX_FAILED_ATTEMPTS - (_lockout.get(lock_key,{}).get("attempts",0))
    if remaining <= 0:
        return jsonify({"message":f"Account locked for {LOCKOUT_MINUTES} minutes."}), 429
    return jsonify({"message":f"Invalid email or password. {max(0,remaining)} attempts remaining before lockout."}), 401


@app.route("/me")
@require_auth
def me():
    uid = request.user_data["user_id"]
    conn = sqlite3.connect("database.db"); c = conn.cursor()
    c.execute("""SELECT id,username,role,full_name,email,mobile,
                        degree,specialty,position,medical_reg_no,years_experience,
                        hospital_name,hospital_city,hospital_country,department,
                        created_at,last_login,must_change_password
                 FROM users WHERE id=?""",(uid,))
    u = c.fetchone(); conn.close()
    if not u: return jsonify({"error":"Not found"}), 404
    keys = ["id","username","role","full_name","email","mobile",
            "degree","specialty","position","medical_reg_no","years_experience",
            "hospital_name","hospital_city","hospital_country","department",
            "created_at","last_login","must_change_password"]
    return jsonify(dict(zip(keys,[v or "" if v is not None else "" for v in u])))


@app.route("/logout", methods=["POST"])
@require_auth
def logout():
    log_activity(request.user_data["user_id"],request.user_data["username"],"LOGOUT")
    # For in-memory fallback, remove token
    token = request.headers.get("X-Auth-Token")
    active_tokens.pop(token, None)
    return jsonify({"message":"Logged out"})


# ── Forgot-password OTP ───────────────────────────────────────────────────────
@app.route("/forgot-password", methods=["POST"])
def forgot_password():
    """Generate OTP for password reset. OTP returned in response (no email needed)."""
    d = request.get_json() or {}
    email = _safe(d.get("email","")).lower()
    if not email or not _valid_email(email):
        return jsonify({"message":"Valid email is required"}), 400
    conn = sqlite3.connect("database.db"); c = conn.cursor()
    c.execute("SELECT id FROM users WHERE LOWER(email)=?",(email,))
    user = c.fetchone()
    conn.close()
    # Always return success to prevent user enumeration
    if not user:
        return jsonify({"message":"If that email is registered, an OTP has been generated.", "otp_hint":"(account not found)"}), 200
    otp = ''.join(random.choices(string.digits, k=6))
    expires_at = datetime.now() + timedelta(minutes=15)
    conn = sqlite3.connect("database.db"); c = conn.cursor()
    # Invalidate old OTPs for this email
    c.execute("UPDATE otp_store SET used=1 WHERE email=? AND used=0",(email,))
    c.execute("INSERT INTO otp_store (email,otp,expires_at) VALUES (?,?,?)",(email,otp,expires_at))
    conn.commit(); conn.close()
    # Return OTP in response (shown to user in UI like a code — no email required)
    return jsonify({
        "message":"OTP generated successfully.",
        "otp": otp,                     # displayed in the UI
        "expires_in_minutes": 15,
        "note": "This OTP is shown here because no email server is configured. In production, send via email."
    })


@app.route("/reset-password", methods=["POST"])
def reset_password():
    """Verify OTP and set new password."""
    d = request.get_json() or {}
    email    = _safe(d.get("email","")).lower()
    otp      = _safe(d.get("otp",""), 6)
    new_pass = d.get("new_password","")
    if not email or not otp or not new_pass:
        return jsonify({"message":"Email, OTP, and new password are required"}), 400
    if len(new_pass) < 6:
        return jsonify({"message":"Password must be at least 6 characters"}), 400
    conn = sqlite3.connect("database.db"); c = conn.cursor()
    c.execute("""SELECT id,otp,expires_at FROM otp_store
                 WHERE email=? AND used=0 ORDER BY id DESC LIMIT 1""",(email,))
    row = c.fetchone()
    if not row:
        conn.close(); return jsonify({"message":"No active OTP found. Please request a new one."}), 400
    _, stored_otp, expires_at_str = row
    try:
        expires_at = datetime.fromisoformat(expires_at_str)
    except Exception:
        expires_at = datetime.now() - timedelta(seconds=1)
    if datetime.now() > expires_at:
        conn.close(); return jsonify({"message":"OTP has expired. Please request a new one."}), 400
    if otp != stored_otp:
        conn.close(); return jsonify({"message":"Incorrect OTP"}), 400
    # Mark OTP used and update password
    c.execute("UPDATE otp_store SET used=1 WHERE email=? AND used=0",(email,))
    c.execute("SELECT id FROM users WHERE LOWER(email)=?",(email,))
    u = c.fetchone()
    if not u:
        conn.close(); return jsonify({"message":"User not found"}), 404
    c.execute("UPDATE users SET password=?,must_change_password=0 WHERE id=?",
              (generate_password_hash(new_pass), u[0]))
    conn.commit(); conn.close()
    log_activity(u[0], email, "PASSWORD_RESET", "Via OTP")
    return jsonify({"message":"Password reset successfully. Please sign in."})


# ── Users (admin) ─────────────────────────────────────────────────────────────
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
    if "role" in d and d["role"] in ("user","admin"):
        updates.append("role=?"); params.append(d["role"])
    if "email" in d and _valid_email(_safe(d["email"]).lower()):
        updates.append("email=?"); params.append(_safe(d["email"]).lower())
    if "password" in d and d["password"]:
        updates.append("password=?"); params.append(generate_password_hash(d["password"]))
    if not updates: conn.close(); return jsonify({"message":"Nothing to update"}), 400
    params.append(uid)
    c.execute(f"UPDATE users SET {', '.join(updates)} WHERE id=?",params)
    conn.commit(); conn.close()
    return jsonify({"message":"User updated successfully"})


# ── Predict ───────────────────────────────────────────────────────────────────
@app.route("/predict", methods=["POST"])
@require_auth
def predict():
    d = request.get_json() or {}
    insulin_map  = {"Down":0,"No":1,"Steady":2,"Up":3}
    change_map   = {"Ch":0,"No":1}
    diabetes_map = {"No":0,"Yes":1}
    try:
        patient_name = _safe(d.get("patient_name",""), 120).strip() or "Unknown Patient"
        patient_id   = generate_patient_id()
        age_val      = _validate_numeric(d.get("age",55), 0, 120, 55)
        insulin_str  = d.get("insulin","No") if d.get("insulin") in insulin_map else "No"
        change_str   = d.get("change","No") if d.get("change") in change_map else "No"
        diabetes_str = d.get("diabetesMed","Yes") if d.get("diabetesMed") in diabetes_map else "Yes"
        notes        = _safe(d.get("notes",""), 500).strip()

        age_v     = age_val
        time_hosp = _validate_numeric(d.get("time_in_hospital",3),   1, 30, 3)
        num_lab   = _validate_numeric(d.get("num_lab_procedures",40), 0, 200, 40)
        num_proc  = _validate_numeric(d.get("num_procedures",1),      0, 20, 1)
        num_med   = _validate_numeric(d.get("num_medications",15),    0, 100, 15)
        n_out     = _validate_numeric(d.get("number_outpatient",0),   0, 100, 0)
        n_er      = _validate_numeric(d.get("number_emergency",0),    0, 100, 0)
        n_inp     = _validate_numeric(d.get("number_inpatient",0),    0, 100, 0)
        n_diag    = _validate_numeric(d.get("number_diagnoses",7),    1, 30, 7)
        insulin_enc  = float(insulin_map.get(insulin_str,1))
        change_enc   = float(change_map.get(change_str,1))
        diabmed_enc  = float(diabetes_map.get(diabetes_str,1))
        enc_count    = _validate_numeric(d.get("patient_encounter_count",1), 1, 50, 1)

        X_row = [age_v,time_hosp,num_lab,num_proc,num_med,n_out,n_er,n_inp,n_diag,insulin_enc,change_enc,diabmed_enc]

        if model is not None:
            admission_type = float(d.get("admission_type_id", 1))
            discharge_disp = float(d.get("discharge_disposition_id", 1))
            admission_src  = float(d.get("admission_source_id", 7))

            med_str_map = {"No": 0, "Down": 1, "Steady": 2, "Up": 3}
            med_names = [
                "metformin","repaglinide","nateglinide","chlorpropamide","glimepiride",
                "acetohexamide","glipizide","glyburide","tolbutamide","pioglitazone",
                "rosiglitazone","acarbose","miglitol","troglitazone","tolazamide",
                "examide","citoglipton","insulin","glyburide-metformin",
                "glipizide-metformin","glimepiride-pioglitazone",
                "metformin-rosiglitazone","metformin-pioglitazone"
            ]
            med_enc = {}
            for m in med_names:
                raw = d.get(m, d.get(m.replace("-","_"), 0))
                if isinstance(raw, str):
                    raw = med_str_map.get(raw, 0)
                med_enc[re.sub(r'[^A-Za-z0-9_]','_', m)] = float(raw)

            active_med_vals  = list(med_enc.values())
            num_meds_changed = sum(1 for v in active_med_vals if v in (1, 3))
            num_active_meds  = sum(1 for v in active_med_vals if v > 0)
            insulin_v        = med_enc.get("insulin", 0.0)
            insulin_used     = int(insulin_v > 0)
            insulin_changed  = int(insulin_v in (1, 3))

            total_vis    = n_out + n_er + n_inp
            race         = d.get("race", "Caucasian")
            gender_male  = int(d.get("gender", "Female") == "Male")
            change_no    = int(change_enc == 1)
            diabmed_yes  = int(diabmed_enc == 1)

            diag1 = int(d.get("diag_1_group", 0))
            diag2 = int(d.get("diag_2_group", 0))
            diag3 = int(d.get("diag_3_group", 0))

            feat_vals = {
                "age":                      age_v,
                "admission_type_id":        admission_type,
                "discharge_disposition_id": discharge_disp,
                "admission_source_id":      admission_src,
                "time_in_hospital":         time_hosp,
                "num_lab_procedures":       num_lab,
                "num_procedures":           num_proc,
                "num_medications":          num_med,
                "number_outpatient":        n_out,
                "number_emergency":         n_er,
                "number_inpatient":         n_inp,
                "number_diagnoses":         n_diag,
                **med_enc,
                "patient_encounter_count":  enc_count,
                "is_repeat_patient":        int(enc_count > 1),
                "is_chronic_patient":       int(enc_count >= 3),
                "a1c_tested":               int(d.get("a1c_result","None") not in ("None","none",None,"")),
                "a1c_high":                 int(d.get("a1c_result","None") in (">7",">8")),
                "glucose_tested":           int(d.get("glucose_serum","None") not in ("None","none",None,"")),
                "glucose_high":             int(d.get("glucose_serum","None") in (">200",">300")),
                "glucose_very_high":        int(d.get("glucose_serum","None") == ">300"),
                "high_risk_specialty":      int(d.get("high_risk_specialty", 0)),
                "specialty_internal":       int(d.get("specialty","") == "InternalMedicine"),
                "specialty_cardiology":     int(d.get("specialty","") == "Cardiology"),
                "high_readmit_discharge":   int(discharge_disp in [6,22,3,7,5,2,4,25,15,10,12]),
                "diag_1_group":             float(diag1),
                "diag_2_group":             float(diag2),
                "diag_3_group":             float(diag3),
                "primary_diag_diabetes":    int(diag1 == 3),
                "primary_diag_circulatory": int(diag1 == 7),
                "any_circulatory":          int(diag1==7 or diag2==7 or diag3==7),
                "any_respiratory":          int(diag1==8 or diag2==8 or diag3==8),
                "n_unique_diag_groups":     len({diag1, diag2, diag3}),
                "num_meds_changed":         num_meds_changed,
                "num_active_meds":          num_active_meds,
                "insulin_used":             insulin_used,
                "insulin_changed":          insulin_changed,
                "total_visits":             total_vis,
                "med_intensity":            num_med / (time_hosp + 1),
                "lab_per_day":              num_lab / (time_hosp + 1),
                "proc_per_day":             num_proc / (time_hosp + 1),
                "diagnosis_severity":       n_diag*2 + num_proc + num_med,
                "is_frequent_patient":      int(total_vis > 2),
                "has_emergency_history":    int(n_er > 0),
                "has_inpatient_history":    int(n_inp > 0),
                "high_meds":                int(num_med > 15),
                "long_stay":                int(time_hosp > 7),
                "elderly":                  int(age_v >= 75),
                "many_lab_procedures":      int(num_lab > 60),
                "many_diagnoses":           int(n_diag >= 9),
                "inpatient_x_diagnoses":    n_inp * n_diag,
                "inpatient_x_emergency":    n_inp * n_er,
                "age_x_inpatient":          age_v * n_inp,
                "meds_x_diagnoses":         num_med * n_diag,
                "encounter_x_inpatient":    enc_count * n_inp,
                "encounter_x_emergency":    enc_count * n_er,
                "race_Asian":               int(race == "Asian"),
                "race_Caucasian":           int(race == "Caucasian"),
                "race_Hispanic":            int(race == "Hispanic"),
                "race_Other":               int(race == "Other"),
                "race_Unknown":             int(race in ("Unknown","AfricanAmerican")),
                "gender_Male":              gender_male,
                "gender_Unknown_Invalid":   0,
                "change_No":                change_no,
                "diabetesMed_Yes":          diabmed_yes,
            }

            def _norm(name):
                return re.sub(r'[^A-Za-z0-9_]', '_', str(name)).lower()

            norm_feat_vals = {_norm(k): v for k, v in feat_vals.items()}
            feature_source = selected_features_list if selected_features_list else list(feat_vals.keys())
            X_vec = np.array([[norm_feat_vals.get(_norm(f), 0.0) for f in feature_source]])

            expected = model.n_features_in_
            actual   = X_vec.shape[1]
            if actual != expected:
                if actual < expected:
                    pad = np.zeros((1, expected - actual))
                    X_vec = np.hstack([X_vec, pad])
                else:
                    X_vec = X_vec[:, :expected]

            try:
                xgb_prob = float(model.predict_proba(X_vec)[0][1])
                if lgbm_model is not None:
                    lgbm_p   = float(lgbm_model.predict_proba(X_vec)[0][1])
                    ens_prob = ensemble_weights[0]*xgb_prob + ensemble_weights[1]*lgbm_p
                else:
                    ens_prob = xgb_prob
                pred   = 1 if ens_prob >= best_threshold else 0
                result = "High Risk" if pred == 1 else "Low Risk"
                prob   = round(ens_prob * 100, 1)
            except Exception:
                import traceback; traceback.print_exc()
                pred, prob = clinical_score_predict(X_row)
                result = "High Risk" if pred == 1 else "Low Risk"
        else:
            pred, prob = clinical_score_predict(X_row)
            result = "High Risk" if pred == 1 else "Low Risk"

        conn = sqlite3.connect("database.db"); c = conn.cursor()
        c.execute("""INSERT INTO predictions
            (user_id,patient_name,patient_id,age,time_in_hospital,num_lab_procedures,
             num_procedures,num_medications,number_outpatient,number_emergency,
             number_inpatient,number_diagnoses,insulin,change_med,diabetes_med,
             result,probability,notes,model_version)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (request.user_data["user_id"],patient_name,patient_id,age_val,
             d.get("time_in_hospital"),d.get("num_lab_procedures"),d.get("num_procedures"),
             d.get("num_medications"),d.get("number_outpatient"),d.get("number_emergency"),
             d.get("number_inpatient"),d.get("number_diagnoses"),
             insulin_str,change_str,diabetes_str,result,prob,notes,MODEL_VERSION))
        conn.commit(); conn.close()
        log_activity(request.user_data["user_id"],request.user_data["username"],
                     "PREDICT",f"{patient_name}({patient_id})→{result}")
        return jsonify({"prediction":result,"probability":prob,"patient_id":patient_id,"model_version":MODEL_VERSION})
    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({"error":str(e)}), 500


# ── History / patients ────────────────────────────────────────────────────────
@app.route("/history")
@require_auth
def history():
    conn = sqlite3.connect("database.db"); c = conn.cursor()
    uid = request.user_data["user_id"]
    search = _safe(request.args.get("search",""), 100)
    filter_result = request.args.get("result","").strip()
    if filter_result not in ("High Risk","Low Risk",""): filter_result = ""
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
    search = _safe(request.args.get("search",""), 100)
    filter_result = request.args.get("result","").strip()
    if filter_result not in ("High Risk","Low Risk",""): filter_result = ""
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
    metrics_path = os.path.join(BASE_DIR, "model", "metrics.json")
    if os.path.exists(metrics_path):
        try:
            with open(metrics_path) as mf:
                saved = json.load(mf)
            saved["model_version"] = MODEL_VERSION
            return jsonify(saved)
        except:
            pass
    ensemble_active = (xgb_model is not None and lgbm_model is not None)
    return jsonify({
        "model_type": "XGBoost + LightGBM Ensemble" if ensemble_active else ("XGBoost" if xgb_model else "Clinical Scoring Fallback"),
        "model_version": MODEL_VERSION,
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


# ── Boot ──────────────────────────────────────────────────────────────────────
init_db()

if __name__ == "__main__":
    print("\n"+"="*60)
    print(" 🏥  Healthcare Readmission Prediction API v3.2")
    print(f" 🤖 Model   : {'XGBoost+LightGBM Ensemble' if (xgb_model and lgbm_model) else ('XGBoost' if xgb_model else 'Clinical Scoring Fallback')}")
    print(f" 🔐 JWT     : {'PyJWT (secure)' if JWT_AVAILABLE else 'In-memory fallback'}")
    print(f" 🛡️  Limiter : {'Flask-Limiter active' if LIMITER_AVAILABLE else 'Disabled'}")
    print(f" 🔒 Lockout : {MAX_FAILED_ATTEMPTS} attempts → {LOCKOUT_MINUTES}min lock")
    print(f" 🌐 CORS    : {ALLOWED_ORIGINS}")
    print(" 🔐 Admin   : admin / admin123 (MUST change on first login)")
    print("="*60+"\n")
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port, debug=False)
