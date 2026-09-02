import os
import json
import secrets
import hashlib
import hmac
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any

from backend.database import get_db_connection


class UserAlreadyExistsError(Exception):
    """Raised when an account already exists with the given email."""
    pass


class InvalidCredentialsError(Exception):
    """Raised when authentication credentials (email/password) are incorrect."""
    pass


class AuthenticationError(Exception):
    """Raised when a session token is missing, invalid, or expired."""
    pass


class AuthService:
    """
    Handles secure password hashing (PBKDF2-HMAC-SHA256), cryptographic session tokens,
    user registration, authentication, address management, and isolated per-user order records.
    """

    @staticmethod
    def hash_password(password: str) -> tuple[str, str]:
        """
        Generates a random 16-byte hex salt and hashes the password using PBKDF2-HMAC-SHA256.
        Returns (password_hash, salt).
        """
        salt = secrets.token_hex(16)
        pwd_bytes = password.encode("utf-8")
        salt_bytes = salt.encode("utf-8")
        pwd_hash = hashlib.pbkdf2_hmac("sha256", pwd_bytes, salt_bytes, 100_000).hex()
        return pwd_hash, salt

    @staticmethod
    def verify_password(password: str, password_hash: str, salt: str) -> bool:
        """
        Verifies a plaintext password against the stored password_hash and salt using constant-time comparison.
        """
        pwd_bytes = password.encode("utf-8")
        salt_bytes = salt.encode("utf-8")
        calculated_hash = hashlib.pbkdf2_hmac("sha256", pwd_bytes, salt_bytes, 100_000).hex()
        return hmac.compare_digest(calculated_hash, password_hash)

    def register_user(self, name: str, email: str, password: str) -> Dict[str, Any]:
        """
        Registers a new user.
        Email is normalized to lowercase.
        Raises UserAlreadyExistsError if email is already taken.
        """
        normalized_email = email.strip().lower()
        if not normalized_email:
            raise ValueError("Email cannot be empty.")
        if not name.strip():
            raise ValueError("Name cannot be empty.")
        if not password:
            raise ValueError("Password cannot be empty.")

        pwd_hash, salt = self.hash_password(password)

        with get_db_connection() as conn:
            cursor = conn.cursor()
            # Check duplicate email (case-insensitive)
            cursor.execute("SELECT id FROM users WHERE LOWER(email) = ?;", (normalized_email,))
            existing = cursor.fetchone()
            if existing:
                raise UserAlreadyExistsError("User already exists. Please log in.")

            cursor.execute("""
                INSERT INTO users (name, email, password_hash, salt, created_at)
                VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP);
            """, (name.strip(), normalized_email, pwd_hash, salt))
            user_id = cursor.lastrowid

            # Create session
            token = self._create_session(conn, user_id)

            return {
                "token": token,
                "user": {
                    "id": user_id,
                    "name": name.strip(),
                    "email": normalized_email,
                    "created_at": datetime.utcnow().isoformat()
                }
            }

    def login_user(self, email: str, password: str) -> Dict[str, Any]:
        """
        Authenticates an existing user.
        Raises InvalidCredentialsError on invalid email or password.
        """
        normalized_email = email.strip().lower()
        if not normalized_email or not password:
            raise InvalidCredentialsError("Invalid email or password.")

        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT id, name, email, password_hash, salt, created_at
                FROM users
                WHERE LOWER(email) = ?;
            """, (normalized_email,))
            user_row = cursor.fetchone()
            if not user_row:
                raise InvalidCredentialsError("Invalid email or password.")

            if not self.verify_password(password, user_row["password_hash"], user_row["salt"]):
                raise InvalidCredentialsError("Invalid email or password.")

            user_id = user_row["id"]
            token = self._create_session(conn, user_id)

            return {
                "token": token,
                "user": {
                    "id": user_id,
                    "name": user_row["name"],
                    "email": user_row["email"],
                    "created_at": str(user_row["created_at"])
                }
            }

    def _create_session(self, conn, user_id: int, days_valid: int = 30) -> str:
        """
        Internal helper to create a cryptographically secure session token in database.
        """
        token = secrets.token_urlsafe(32)
        expires_at = datetime.utcnow() + timedelta(days=days_valid)
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO user_sessions (token, user_id, created_at, expires_at)
            VALUES (?, ?, CURRENT_TIMESTAMP, ?);
        """, (token, user_id, expires_at.strftime("%Y-%m-%d %H:%M:%S")))
        return token

    def get_user_by_token(self, token: str) -> Optional[Dict[str, Any]]:
        """
        Validates token and returns the current user dict, or None if token is invalid/expired.
        """
        if not token:
            return None

        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT u.id, u.name, u.email, u.created_at, s.expires_at
                FROM user_sessions s
                JOIN users u ON s.user_id = u.id
                WHERE s.token = ?;
            """, (token,))
            row = cursor.fetchone()
            if not row:
                return None

            # Verify expiration
            try:
                expires_at = datetime.strptime(str(row["expires_at"]), "%Y-%m-%d %H:%M:%S")
                if expires_at < datetime.utcnow():
                    # Expired session -> clean up
                    cursor.execute("DELETE FROM user_sessions WHERE token = ?;", (token,))
                    return None
            except Exception:
                pass

            return {
                "id": row["id"],
                "name": row["name"],
                "email": row["email"],
                "created_at": str(row["created_at"])
            }

    def logout_user(self, token: str) -> bool:
        """
        Invalidates a session token upon logout.
        """
        if not token:
            return False
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM user_sessions WHERE token = ?;", (token,))
            return cursor.rowcount > 0

    def get_user_address(self, user_id: int) -> Optional[Dict[str, Any]]:
        """
        Retrieves the current default delivery address for the authenticated user.
        """
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT id, user_id, recipient_name, phone, address_line1, address_line2,
                       city, state, pincode, country, updated_at
                FROM user_addresses
                WHERE user_id = ?;
            """, (user_id,))
            row = cursor.fetchone()
            if not row:
                return None

            return {
                "id": row["id"],
                "user_id": row["user_id"],
                "recipient_name": row["recipient_name"],
                "phone": row["phone"],
                "address_line1": row["address_line1"],
                "address_line2": row["address_line2"] or "",
                "city": row["city"],
                "state": row["state"],
                "pincode": row["pincode"],
                "country": row["country"],
                "updated_at": str(row["updated_at"])
            }

    def save_user_address(self, user_id: int, address_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Saves or updates the delivery address for the authenticated user.
        """
        recipient_name = address_data.get("recipient_name", "").strip()
        phone = address_data.get("phone", "").strip()
        address_line1 = address_data.get("address_line1", "").strip()
        address_line2 = address_data.get("address_line2", "").strip()
        city = address_data.get("city", "").strip()
        state = address_data.get("state", "").strip()
        pincode = address_data.get("pincode", "").strip()
        country = address_data.get("country", "India").strip()

        if not recipient_name or not phone or not address_line1 or not city or not state or not pincode:
            raise ValueError("All required address fields must be provided.")

        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO user_addresses (
                    user_id, recipient_name, phone, address_line1, address_line2,
                    city, state, pincode, country, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(user_id) DO UPDATE SET
                    recipient_name = excluded.recipient_name,
                    phone = excluded.phone,
                    address_line1 = excluded.address_line1,
                    address_line2 = excluded.address_line2,
                    city = excluded.city,
                    state = excluded.state,
                    pincode = excluded.pincode,
                    country = excluded.country,
                    updated_at = CURRENT_TIMESTAMP;
            """, (user_id, recipient_name, phone, address_line1, address_line2, city, state, pincode, country))

        return self.get_user_address(user_id)

    def record_user_order(
        self,
        user_id: int,
        order_id: str,
        merchant: str,
        total_amount: float,
        payment_status: str,
        order_status: str,
        items: List[Dict[str, Any]],
        shipping_address: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Records an immutable order entry in user_orders for the authenticated user.
        Includes an immutable snapshot of shipping_address.
        Idempotent: Prevents duplicate records for the same order_id.
        """
        with get_db_connection() as conn:
            cursor = conn.cursor()
            # Idempotency check
            cursor.execute("SELECT id FROM user_orders WHERE order_id = ?;", (order_id,))
            existing = cursor.fetchone()
            if existing:
                return self.get_user_order_by_id(user_id, order_id)

            items_json_str = json.dumps(items, ensure_ascii=False)
            shipping_json_str = json.dumps(shipping_address, ensure_ascii=False)

            cursor.execute("""
                INSERT INTO user_orders (
                    user_id, order_id, merchant, total_amount, payment_status,
                    order_status, items_json, shipping_address, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP);
            """, (
                user_id,
                order_id,
                merchant,
                float(total_amount),
                payment_status,
                order_status,
                items_json_str,
                shipping_json_str
            ))

            new_id = cursor.lastrowid
            return {
                "id": new_id,
                "user_id": user_id,
                "order_id": order_id,
                "merchant": merchant,
                "total_amount": float(total_amount),
                "payment_status": payment_status,
                "order_status": order_status,
                "items": items,
                "shipping_address": shipping_address,
                "created_at": datetime.utcnow().isoformat()
            }

    def get_user_orders(self, user_id: int) -> List[Dict[str, Any]]:
        """
        Retrieves all orders belonging ONLY to the authenticated user, sorted latest first.
        """
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT id, user_id, order_id, merchant, total_amount, payment_status,
                       order_status, items_json, shipping_address, created_at
                FROM user_orders
                WHERE user_id = ?
                ORDER BY id DESC;
            """, (user_id,))
            rows = cursor.fetchall()
            orders = []
            for row in rows:
                try:
                    items = json.loads(row["items_json"])
                except Exception:
                    items = []
                try:
                    shipping_addr = json.loads(row["shipping_address"])
                except Exception:
                    shipping_addr = {}

                orders.append({
                    "id": row["id"],
                    "user_id": row["user_id"],
                    "order_id": row["order_id"],
                    "merchant": row["merchant"],
                    "total_amount": row["total_amount"],
                    "payment_status": row["payment_status"],
                    "order_status": row["order_status"],
                    "items": items,
                    "shipping_address": shipping_addr,
                    "created_at": str(row["created_at"])
                })
            return orders

    def get_user_order_by_id(self, user_id: int, order_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieves a specific order ONLY if it belongs to the authenticated user.
        Enforces user ownership security.
        """
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT id, user_id, order_id, merchant, total_amount, payment_status,
                       order_status, items_json, shipping_address, created_at
                FROM user_orders
                WHERE user_id = ? AND order_id = ?;
            """, (user_id, order_id))
            row = cursor.fetchone()
            if not row:
                return None

            try:
                items = json.loads(row["items_json"])
            except Exception:
                items = []
            try:
                shipping_addr = json.loads(row["shipping_address"])
            except Exception:
                shipping_addr = {}

            return {
                "id": row["id"],
                "user_id": row["user_id"],
                "order_id": row["order_id"],
                "merchant": row["merchant"],
                "total_amount": row["total_amount"],
                "payment_status": row["payment_status"],
                "order_status": row["order_status"],
                "items": items,
                "shipping_address": shipping_addr,
                "created_at": str(row["created_at"])
            }


auth_service = AuthService()
