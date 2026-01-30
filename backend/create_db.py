import sqlite3

conn = sqlite3.connect("database.db")
c = conn.cursor()

# Users table (Admin login)
c.execute("""
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT,
    password TEXT
)
""")

# Predictions history table
c.execute("""
CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    time_in_hospital INTEGER,
    num_lab_procedures INTEGER,
    num_medications INTEGER,
    number_inpatient INTEGER,
    result TEXT
)
""")

# Insert default admin
c.execute("INSERT INTO users (username, password) VALUES (?, ?)",
          ("admin", "admin123"))

conn.commit()
conn.close()

print("database.db created successfully")