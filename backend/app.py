from flask import Flask, request, jsonify
import joblib
import numpy as np
import sqlite3
import os
import secrets
from werkzeug.security import generate_password_hash, check_password_hash
from functools import wraps

app = Flask(__name__)

# Simple token storage (in production, use Redis or database)
active_tokens = {}

# Load model
model_paths = [
    "readmission_model.pkl",
    "model/readmission_model.pkl",
    "../model/readmission_model.pkl",
    os.path.join(os.path.dirname(__file__), "readmission_model.pkl"),
    os.path.join(os.path.dirname(__file__), "..", "model", "readmission_model.pkl")
]

model = None
for path in model_paths:
    try:
        if os.path.exists(path):
            model = joblib.load(path)
            print(f"✅ Model loaded from: {path}")
            break
    except:
        continue

if model is None:
    print("⚠️ WARNING: Using dummy model!")
    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    X_dummy = np.array([[5, 30, 10, 1], [10, 60, 20, 3]])
    y_dummy = np.array([0, 1])
    model.fit(X_dummy, y_dummy)


# CORS helper
@app.after_request
def after_request(response):
    origin = request.headers.get('Origin', '*')
    response.headers['Access-Control-Allow-Origin'] = origin
    response.headers['Access-Control-Allow-Credentials'] = 'true'
    response.headers['Access-Control-Allow-Methods'] = 'GET, POST, PUT, DELETE, OPTIONS'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization, X-Auth-Token'
    return response


# Auth decorator
def require_auth(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        token = request.headers.get('X-Auth-Token')
        
        if not token or token not in active_tokens:
            return jsonify({"error": "Unauthorized"}), 401
        
        # Attach user info to request
        request.user_data = active_tokens[token]
        return f(*args, **kwargs)
    return decorated_function


def require_admin(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        token = request.headers.get('X-Auth-Token')
        
        if not token or token not in active_tokens:
            return jsonify({"error": "Unauthorized"}), 401
        
        if active_tokens[token]['role'] != 'admin':
            return jsonify({"error": "Admin access required"}), 403
        
        request.user_data = active_tokens[token]
        return f(*args, **kwargs)
    return decorated_function


@app.route("/", methods=["GET", "OPTIONS"])
def home():
    if request.method == "OPTIONS":
        return "", 200
    return "Healthcare Readmission Prediction API is running"


@app.route("/register", methods=["POST", "OPTIONS"])
def register():
    if request.method == "OPTIONS":
        return "", 200
        
    data = request.get_json()
    username = data["username"]
    password = generate_password_hash(data["password"])
    role = data.get("role", "user")

    conn = sqlite3.connect("database.db")
    c = conn.cursor()

    try:
        c.execute("INSERT INTO users (username, password, role) VALUES (?, ?, ?)",
                  (username, password, role))
        conn.commit()
        print(f"✅ User registered: {username} ({role})")
        return jsonify({"message": "User registered successfully"})
    except:
        return jsonify({"message": "Username already exists"}), 400
    finally:
        conn.close()


@app.route("/login", methods=["POST", "OPTIONS"])
def login():
    if request.method == "OPTIONS":
        return "", 200
    
    print("\n" + "="*50)
    print("🔐 LOGIN ATTEMPT")
    
    data = request.get_json()
    username = data.get("username")
    password = data.get("password")
    
    print(f"Username: {username}")

    conn = sqlite3.connect("database.db")
    c = conn.cursor()

    c.execute("SELECT id, password, role FROM users WHERE username=?", (username,))
    user = c.fetchone()
    conn.close()

    if user and check_password_hash(user[1], password):
        # Generate token
        token = secrets.token_hex(32)
        
        # Store token
        active_tokens[token] = {
            "user_id": user[0],
            "username": username,
            "role": user[2]
        }
        
        print(f"✅ LOGIN SUCCESS")
        print(f"User ID: {user[0]}, Role: {user[2]}")
        print(f"Token: {token[:16]}...")
        print("="*50 + "\n")
        
        return jsonify({
            "message": "Login successful",
            "token": token,
            "role": user[2],
            "username": username
        })
    else:
        print(f"❌ LOGIN FAILED")
        print("="*50 + "\n")
        return jsonify({"message": "Invalid credentials"}), 401


@app.route("/logout", methods=["POST", "OPTIONS"])
@require_auth
def logout():
    if request.method == "OPTIONS":
        return "", 200
        
    token = request.headers.get('X-Auth-Token')
    if token in active_tokens:
        del active_tokens[token]
    print("🚪 User logged out")
    return jsonify({"message": "Logged out"})


@app.route("/users", methods=["GET", "OPTIONS"])
@require_admin
def get_users():
    if request.method == "OPTIONS":
        return "", 200

    conn = sqlite3.connect("database.db")
    c = conn.cursor()
    
    c.execute("SELECT id, username, role FROM users ORDER BY id")
    users = c.fetchall()
    conn.close()

    users_list = [{"id": u[0], "username": u[1], "role": u[2]} for u in users]
    return jsonify({"users": users_list})


@app.route("/delete-user/<int:user_id>", methods=["DELETE", "OPTIONS"])
@require_admin
def delete_user(user_id):
    if request.method == "OPTIONS":
        return "", 200

    conn = sqlite3.connect("database.db")
    c = conn.cursor()

    c.execute("SELECT username FROM users WHERE id=?", (user_id,))
    user = c.fetchone()
    
    if user and user[0] == "admin":
        conn.close()
        return jsonify({"message": "Cannot delete admin user"}), 400

    try:
        c.execute("DELETE FROM predictions WHERE user_id=?", (user_id,))
        c.execute("DELETE FROM users WHERE id=?", (user_id,))
        conn.commit()
        print(f"✅ User deleted: ID {user_id}")
        return jsonify({"message": "User deleted successfully"})
    except Exception as e:
        return jsonify({"message": str(e)}), 500
    finally:
        conn.close()


@app.route("/predict", methods=["POST", "OPTIONS"])
@require_auth
def predict():
    if request.method == "OPTIONS":
        return "", 200
    
    print("\n" + "="*50)
    print("🔮 PREDICTION REQUEST")
    print(f"User: {request.user_data['username']}")
    
    data = request.get_json()

    try:
        features = np.array([[ 
            data["time_in_hospital"],
            data["num_lab_procedures"],
            data["num_medications"],
            data["number_inpatient"]
        ]])

        prediction = model.predict(features)[0]
        result = "High Risk" if prediction == 1 else "Low Risk"

        conn = sqlite3.connect("database.db")
        c = conn.cursor()

        c.execute("""
            INSERT INTO predictions 
            (user_id, time_in_hospital, num_lab_procedures, 
             num_medications, number_inpatient, result)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            request.user_data['user_id'],
            data["time_in_hospital"],
            data["num_lab_procedures"],
            data["num_medications"],
            data["number_inpatient"],
            result
        ))

        conn.commit()
        conn.close()

        print(f"✅ PREDICTION: {result}")
        print("="*50 + "\n")
        return jsonify({"prediction": result})

    except Exception as e:
        print(f"❌ ERROR: {str(e)}")
        print("="*50 + "\n")
        return jsonify({"error": str(e)}), 500


@app.route("/history", methods=["GET", "OPTIONS"])
@require_auth
def history():
    if request.method == "OPTIONS":
        return "", 200

    conn = sqlite3.connect("database.db")
    c = conn.cursor()

    c.execute("""
        SELECT time_in_hospital, num_lab_procedures, 
               num_medications, number_inpatient, result
        FROM predictions
        WHERE user_id=?
        ORDER BY id DESC
    """, (request.user_data['user_id'],))

    rows = c.fetchall()
    conn.close()

    history_list = []
    for row in rows:
        history_list.append({
            "time_in_hospital": row[0],
            "num_lab_procedures": row[1],
            "num_medications": row[2],
            "number_inpatient": row[3],
            "result": row[4]
        })

    return jsonify({"history": history_list})


@app.route("/dashboard-data", methods=["GET", "OPTIONS"])
@require_auth
def dashboard_data():
    if request.method == "OPTIONS":
        return "", 200

    conn = sqlite3.connect("database.db")
    c = conn.cursor()

    if request.user_data['role'] == "admin":
        c.execute("SELECT COUNT(*) FROM predictions")
        total = c.fetchone()[0]
        c.execute("SELECT COUNT(*) FROM predictions WHERE result='High Risk'")
        high = c.fetchone()[0]
        c.execute("SELECT COUNT(*) FROM predictions WHERE result='Low Risk'")
        low = c.fetchone()[0]
    else:
        user_id = request.user_data['user_id']
        c.execute("SELECT COUNT(*) FROM predictions WHERE user_id=?", (user_id,))
        total = c.fetchone()[0]
        c.execute("SELECT COUNT(*) FROM predictions WHERE user_id=? AND result='High Risk'", (user_id,))
        high = c.fetchone()[0]
        c.execute("SELECT COUNT(*) FROM predictions WHERE user_id=? AND result='Low Risk'", (user_id,))
        low = c.fetchone()[0]

    conn.close()
    return jsonify({"total": total, "high": high, "low": low})


if __name__ == "__main__":
    print("\n" + "="*60)
    print("🏥 Healthcare Readmission Prediction API")
    print("="*60)
    print(f"✅ Server starting on http://localhost:5000")
    print(f"✅ Using TOKEN-BASED authentication")
    print("="*60 + "\n")
    
    app.run(host="0.0.0.0", port=5000, debug=True)