import sqlite3
from werkzeug.security import generate_password_hash

conn = sqlite3.connect("database.db")
c = conn.cursor()

# =============================
# USERS TABLE
# =============================
c.execute("""
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT UNIQUE,
    password TEXT,
    role TEXT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
)
""")

# =============================
# PREDICTIONS TABLE (UPDATED with probability & timestamp)
# =============================
c.execute("""
CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER,
    time_in_hospital INTEGER,
    num_lab_procedures INTEGER,
    num_medications INTEGER,
    number_inpatient INTEGER,
    result TEXT,
    probability REAL,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(user_id) REFERENCES users(id)
)
""")

# =============================
# DEFAULT ADMIN
# =============================
c.execute("SELECT * FROM users WHERE username=?", ("admin",))
if not c.fetchone():
    hashed_password = generate_password_hash("admin123")
    c.execute(
        "INSERT INTO users (username, password, role) VALUES (?, ?, ?)",
        ("admin", hashed_password, "admin")
    )
    print("✅ Admin user created: admin / admin123")

conn.commit()
conn.close()
print("✅ Database ready!")
