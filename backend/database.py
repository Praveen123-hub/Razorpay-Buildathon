import os
import sqlite3
from typing import Generator
from contextlib import contextmanager

# Database file location: backend/data/app.db
BACKEND_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
DB_PATH = os.path.join(BACKEND_DATA_DIR, "app.db")


@contextmanager
def get_db_connection() -> Generator[sqlite3.Connection, None, None]:
    """
    Context manager for SQLite database connection.
    Ensures data directory exists and returns row-factory enabled connection.
    """
    os.makedirs(BACKEND_DATA_DIR, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db() -> None:
    """
    Initializes the SQLite database tables if they do not exist.
    Never deletes or resets existing tables or data.
    """
    os.makedirs(BACKEND_DATA_DIR, exist_ok=True)
    with get_db_connection() as conn:
        cursor = conn.cursor()

        # 1. users table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                email TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                salt TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)

        # 2. user_addresses table (one current/default address per user)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS user_addresses (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER UNIQUE NOT NULL,
                recipient_name TEXT NOT NULL,
                phone TEXT NOT NULL,
                address_line1 TEXT NOT NULL,
                address_line2 TEXT,
                city TEXT NOT NULL,
                state TEXT NOT NULL,
                pincode TEXT NOT NULL,
                country TEXT NOT NULL,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
            );
        """)

        # 3. user_sessions table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS user_sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                token TEXT UNIQUE NOT NULL,
                user_id INTEGER NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                expires_at TIMESTAMP NOT NULL,
                FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
            );
        """)

        # 4. user_orders table (per-user order history with immutable shipping address snapshot)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS user_orders (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                order_id TEXT NOT NULL,
                merchant TEXT NOT NULL,
                total_amount REAL NOT NULL,
                payment_status TEXT NOT NULL,
                order_status TEXT NOT NULL,
                items_json TEXT NOT NULL,
                shipping_address TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
            );
        """)

        # Index on token for fast session lookup
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_sessions_token ON user_sessions(token);")
        # Index on user_orders user_id for fast user orders lookup
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_user_orders_user ON user_orders(user_id);")
        # Index on user_orders order_id to prevent duplicates
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_user_orders_order ON user_orders(order_id);")
