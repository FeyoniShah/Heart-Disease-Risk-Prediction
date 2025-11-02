# database.py
import mysql.connector
import re
import json
from datetime import datetime

class Database:
    def __init__(self, host="localhost", user="root", password="Feyoni@1819", database="heart_app"):
        self.host = host
        self.user = user
        self.password = password
        self.database = database
        self._create_database()
        self._create_tables()

    def _get_conn(self):
        return mysql.connector.connect(
            host=self.host,
            user=self.user,
            password=self.password,
            database=self.database
        )

    def _create_database(self):
        conn = mysql.connector.connect(
            host=self.host,
            user=self.user,
            password=self.password
        )
        cursor = conn.cursor()
        cursor.execute(f"CREATE DATABASE IF NOT EXISTS {self.database}")
        cursor.close()
        conn.close()

    def _create_tables(self):
        conn = self._get_conn()
        cursor = conn.cursor()

        cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INT AUTO_INCREMENT PRIMARY KEY,
            email VARCHAR(255) UNIQUE NOT NULL,
            password VARCHAR(255) NOT NULL,
            name VARCHAR(100),
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
        """)

        cursor.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INT AUTO_INCREMENT PRIMARY KEY,
            user_id INT,
            input_json TEXT,
            result_json TEXT,
            model_name VARCHAR(100),
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
        )
        """)

        conn.commit()
        cursor.close()
        conn.close()

    # ---------------------------
    # User operations
    # ---------------------------
    def is_valid_email(self, email):
        return re.match(r"[^@]+@[^@]+\.[^@]+", email)

    def create_user(self, email, password, name=None):
        if not self.is_valid_email(email):
            raise ValueError("Invalid email format")
        if len(password) < 8:
            raise ValueError("Password must be at least 8 characters long")

        conn = self._get_conn()
        cursor = conn.cursor()
        try:
            cursor.execute(
                "INSERT INTO users (email, password, name) VALUES (%s, %s, %s)",
                (email, password, name)
            )
            conn.commit()
            return cursor.lastrowid
        except mysql.connector.IntegrityError:
            raise ValueError("Email already registered")
        finally:
            cursor.close()
            conn.close()

    def verify_user(self, email, password):
        conn = self._get_conn()
        cursor = conn.cursor(dictionary=True)
        cursor.execute("SELECT * FROM users WHERE email=%s AND password=%s", (email, password))
        user = cursor.fetchone()
        cursor.close()
        conn.close()
        return user

    # ---------------------------
    # Prediction operations
    # ---------------------------
    def insert_prediction(self, user_id, input_data, result_data, model_name="Unknown"):
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO predictions (user_id, input_json, result_json, model_name) VALUES (%s, %s, %s, %s)",
            (user_id, json.dumps(input_data), json.dumps(result_data), model_name)
        )
        conn.commit()
        pid = cursor.lastrowid
        cursor.close()
        conn.close()
        return pid

    def get_predictions(self, user_id):
        conn = self._get_conn()
        cursor = conn.cursor(dictionary=True)
        cursor.execute(
            "SELECT * FROM predictions WHERE user_id=%s ORDER BY created_at DESC",
            (user_id,)
        )
        rows = cursor.fetchall()
        for row in rows:
            try:
                row["input_json"] = json.loads(row["input_json"])
                row["result_json"] = json.loads(row["result_json"])
            except Exception:
                pass
        cursor.close()
        conn.close()
        return rows
    

    def get_recent_predictions(self, limit=50):
        conn = self._get_conn()
        cursor = conn.cursor(dictionary=True)
        cursor.execute(
            """
            SELECT p.*, u.email, u.name
            FROM predictions p
            LEFT JOIN users u ON p.user_id = u.id
            ORDER BY p.created_at DESC
            LIMIT %s
            """,
            (limit,)
        )
        rows = cursor.fetchall()
        for row in rows:
            try:
                row["input_json"] = json.loads(row["input_json"])
                row["result_json"] = json.loads(row["result_json"])
            except Exception:
                pass
        cursor.close()
        conn.close()
        return rows





if __name__ == "__main__":
    db = Database()

    # Just to confirm tables are set up
    print("✅ Database and tables created successfully.")

    # Test user creation
    try:
        user_id = db.create_user("test@example.com", "password123", "Test User")
        print(f"User created with ID: {user_id}")
    except ValueError as e:
        print(f"User creation failed: {e}")

    # Test login
    user = db.verify_user("test@example.com", "password123")
    if user:
        print("✅ User verified:", user)
    else:
        print("❌ Invalid login")
