"""Account storage, password hashing and password recovery.

Standard library only (sqlite3, hashlib.scrypt, secrets, smtplib) so the app
keeps its current dependency footprint.

Configuration (environment variables, all optional except SMTP for email):
    LCA_AUTH_DB            SQLite path (default: data/auth.db)
    LCA_APP_BASE_URL       Public URL of the app, used in reset links
                           (default: http://localhost:8501)
    LCA_RESET_TTL_MINUTES  Reset link lifetime (default: 30)
    LCA_ADMIN_EMAIL / LCA_ADMIN_PASSWORD / LCA_ADMIN_NAME
                           Seed the first account when no account exists yet.
    SMTP_HOST, SMTP_PORT (587), SMTP_USERNAME, SMTP_PASSWORD,
    SMTP_FROM, SMTP_USE_SSL (0/1; default STARTTLS)
    LCA_DEV_PRINT_RESET_LINK=1
                           Local development only: print reset links to the
                           server console when SMTP is not configured.
"""
from __future__ import annotations

import hashlib
import hmac
import logging
import os
import re
import secrets
import smtplib
import sqlite3
import time
from contextlib import contextmanager
from email.message import EmailMessage
from typing import Optional

log = logging.getLogger("lca.auth")

PASSWORD_MIN_LENGTH = 10
PASSWORD_MAX_LENGTH = 128
LOGIN_MAX_FAILURES = 5          # per email within the window below
LOGIN_WINDOW_SECONDS = 15 * 60
RESET_MAX_REQUESTS = 3          # per email within the window below
RESET_WINDOW_SECONDS = 60 * 60

_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")

# scrypt parameters (n=2**14, r=8, p=1 => ~16 MB, interactive-login strength)
_SCRYPT_N, _SCRYPT_R, _SCRYPT_P = 2 ** 14, 8, 1


def _db_path() -> str:
    return os.environ.get("LCA_AUTH_DB", "data/auth.db")


def _reset_ttl_seconds() -> int:
    try:
        return max(5, int(os.environ.get("LCA_RESET_TTL_MINUTES", "30"))) * 60
    except ValueError:
        return 30 * 60


@contextmanager
def _connect():
    path = _db_path()
    if os.path.dirname(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
    conn = sqlite3.connect(path, timeout=10)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db() -> None:
    with _connect() as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS users (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                email         TEXT NOT NULL UNIQUE COLLATE NOCASE,
                name          TEXT NOT NULL DEFAULT '',
                password_hash TEXT NOT NULL,
                is_active     INTEGER NOT NULL DEFAULT 1,
                created_at    REAL NOT NULL,
                password_changed_at REAL NOT NULL
            );
            CREATE TABLE IF NOT EXISTS password_reset_tokens (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id     INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                token_hash  TEXT NOT NULL UNIQUE,
                created_at  REAL NOT NULL,
                expires_at  REAL NOT NULL,
                used_at     REAL
            );
            CREATE INDEX IF NOT EXISTS idx_reset_user ON password_reset_tokens(user_id);
            CREATE TABLE IF NOT EXISTS auth_events (
                id    INTEGER PRIMARY KEY AUTOINCREMENT,
                kind  TEXT NOT NULL,          -- 'login_failed' | 'reset_requested'
                key   TEXT NOT NULL,          -- normalised email
                at    REAL NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_events ON auth_events(kind, key, at);
            """
        )
    _seed_admin_from_env()


def _seed_admin_from_env() -> None:
    email = os.environ.get("LCA_ADMIN_EMAIL", "").strip()
    password = os.environ.get("LCA_ADMIN_PASSWORD", "")
    if not email or not password or count_users() > 0:
        return
    try:
        create_user(email, password, os.environ.get("LCA_ADMIN_NAME", "Administrator"))
        log.info("Seeded first account from LCA_ADMIN_EMAIL")
    except ValueError as exc:
        log.error("Could not seed first account: %s", exc)


# ---------------------------------------------------------------- passwords
def hash_password(password: str) -> str:
    salt = secrets.token_bytes(16)
    digest = hashlib.scrypt(password.encode(), salt=salt, n=_SCRYPT_N, r=_SCRYPT_R,
                            p=_SCRYPT_P, dklen=32)
    return f"scrypt${_SCRYPT_N}${_SCRYPT_R}${_SCRYPT_P}${salt.hex()}${digest.hex()}"


def verify_password(password: str, stored: str) -> bool:
    try:
        algo, n, r, p, salt_hex, digest_hex = stored.split("$")
        if algo != "scrypt":
            return False
        digest = hashlib.scrypt(password.encode(), salt=bytes.fromhex(salt_hex), n=int(n),
                                r=int(r), p=int(p), dklen=len(digest_hex) // 2)
        return hmac.compare_digest(digest.hex(), digest_hex)
    except (ValueError, TypeError):
        return False


# Used to spend the same time on unknown emails as on known ones.
_DUMMY_HASH = hash_password(secrets.token_urlsafe(16))


def normalise_email(email: str) -> str:
    return (email or "").strip().lower()


def is_valid_email(email: str) -> bool:
    return bool(_EMAIL_RE.match(normalise_email(email))) and len(email) <= 254


def password_problems(password: str, email: str = "") -> list[str]:
    """Return human-readable reasons the password is not acceptable."""
    problems = []
    if len(password) < PASSWORD_MIN_LENGTH:
        problems.append(f"Use at least {PASSWORD_MIN_LENGTH} characters.")
    if len(password) > PASSWORD_MAX_LENGTH:
        problems.append(f"Use at most {PASSWORD_MAX_LENGTH} characters.")
    if password and password.strip() != password:
        problems.append("Do not start or end with a space.")
    if email and password.lower() == normalise_email(email):
        problems.append("Do not use your email address as the password.")
    if password and len(set(password)) < 4:
        problems.append("Use a less repetitive password.")
    return problems


# -------------------------------------------------------------------- users
def count_users() -> int:
    with _connect() as conn:
        return conn.execute("SELECT COUNT(*) FROM users").fetchone()[0]


def create_user(email: str, password: str, name: str = "") -> int:
    email = normalise_email(email)
    if not is_valid_email(email):
        raise ValueError("Invalid email address.")
    problems = password_problems(password, email)
    if problems:
        raise ValueError(" ".join(problems))
    now = time.time()
    try:
        with _connect() as conn:
            cur = conn.execute(
                "INSERT INTO users (email, name, password_hash, created_at, password_changed_at)"
                " VALUES (?, ?, ?, ?, ?)",
                (email, name.strip(), hash_password(password), now, now),
            )
            return cur.lastrowid
    except sqlite3.IntegrityError:
        raise ValueError("An account with this email already exists.")


def set_active(email: str, active: bool) -> bool:
    with _connect() as conn:
        cur = conn.execute("UPDATE users SET is_active = ? WHERE email = ?",
                           (1 if active else 0, normalise_email(email)))
        return cur.rowcount > 0


def list_users() -> list[dict]:
    with _connect() as conn:
        rows = conn.execute("SELECT id, email, name, is_active, created_at FROM users ORDER BY id")
        return [dict(r) for r in rows]


def _get_user_by_email(conn, email: str) -> Optional[sqlite3.Row]:
    return conn.execute("SELECT * FROM users WHERE email = ?", (normalise_email(email),)).fetchone()


# ------------------------------------------------------------- rate limits
def _recent_events(conn, kind: str, key: str, window: int) -> int:
    return conn.execute("SELECT COUNT(*) FROM auth_events WHERE kind = ? AND key = ? AND at > ?",
                        (kind, key, time.time() - window)).fetchone()[0]


def _record_event(conn, kind: str, key: str) -> None:
    now = time.time()
    conn.execute("INSERT INTO auth_events (kind, key, at) VALUES (?, ?, ?)", (kind, key, now))
    # keep the table small
    conn.execute("DELETE FROM auth_events WHERE at < ?", (now - 24 * 3600,))


# ------------------------------------------------------------------- login
class LoginResult:
    OK = "ok"
    INVALID = "invalid"
    LOCKED = "locked"

    def __init__(self, status: str, user: Optional[dict] = None):
        self.status = status
        self.user = user


def authenticate(email: str, password: str) -> LoginResult:
    key = normalise_email(email)
    with _connect() as conn:
        if _recent_events(conn, "login_failed", key, LOGIN_WINDOW_SECONDS) >= LOGIN_MAX_FAILURES:
            return LoginResult(LoginResult.LOCKED)
        row = _get_user_by_email(conn, key)
        ok = verify_password(password, row["password_hash"] if row else _DUMMY_HASH)
        if row is None or not ok or not row["is_active"]:
            _record_event(conn, "login_failed", key)
            return LoginResult(LoginResult.INVALID)
        conn.execute("DELETE FROM auth_events WHERE kind = 'login_failed' AND key = ?", (key,))
        return LoginResult(LoginResult.OK, {
            "id": row["id"], "email": row["email"], "name": row["name"],
            "password_changed_at": row["password_changed_at"],
        })


def session_still_valid(user: dict) -> bool:
    """A session ends when the account is disabled or its password changed."""
    with _connect() as conn:
        row = conn.execute("SELECT is_active, password_changed_at FROM users WHERE id = ?",
                           (user["id"],)).fetchone()
    return bool(row and row["is_active"]
                and row["password_changed_at"] <= user["password_changed_at"] + 1e-6)


# -------------------------------------------------------- password recovery
def _token_hash(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


def smtp_configured() -> bool:
    return bool(os.environ.get("SMTP_HOST") and os.environ.get("SMTP_FROM"))


def reset_url(token: str) -> str:
    base = os.environ.get("LCA_APP_BASE_URL", "http://localhost:8501").rstrip("/")
    return f"{base}/?reset_token={token}"


def _send_reset_email(to_email: str, name: str, link: str, ttl_minutes: int) -> None:
    msg = EmailMessage()
    msg["Subject"] = "Reset your Live Chat Analysis password"
    msg["From"] = os.environ["SMTP_FROM"]
    msg["To"] = to_email
    greeting = f"Hi {name}," if name else "Hi,"
    msg.set_content(
        f"{greeting}\n\n"
        "We received a request to reset the password for your Live Chat Analysis account.\n\n"
        f"Open this link to choose a new password (valid for {ttl_minutes} minutes, one use only):\n"
        f"{link}\n\n"
        "If you did not request this, you can ignore this email. Your password stays the same.\n"
    )
    host = os.environ["SMTP_HOST"]
    port = int(os.environ.get("SMTP_PORT", "587"))
    user = os.environ.get("SMTP_USERNAME")
    password = os.environ.get("SMTP_PASSWORD")
    if os.environ.get("SMTP_USE_SSL", "0") == "1":
        server = smtplib.SMTP_SSL(host, port, timeout=15)
    else:
        server = smtplib.SMTP(host, port, timeout=15)
        server.starttls()
    with server:
        if user:
            server.login(user, password or "")
        server.send_message(msg)


def request_password_reset(email: str) -> None:
    """Create a reset token and email it, if the account exists.

    Deliberately returns nothing: the caller shows the same message whether or
    not the email is registered, so the form cannot be used to discover accounts.
    """
    key = normalise_email(email)
    if not is_valid_email(key):
        return
    ttl = _reset_ttl_seconds()
    with _connect() as conn:
        if _recent_events(conn, "reset_requested", key, RESET_WINDOW_SECONDS) >= RESET_MAX_REQUESTS:
            log.warning("Password reset rate limit reached for an account")
            return
        _record_event(conn, "reset_requested", key)
        row = _get_user_by_email(conn, key)
        if row is None or not row["is_active"]:
            return
        token = secrets.token_urlsafe(32)
        now = time.time()
        # only the most recent link stays valid
        conn.execute("UPDATE password_reset_tokens SET used_at = ? WHERE user_id = ? AND used_at IS NULL",
                     (now, row["id"]))
        conn.execute(
            "INSERT INTO password_reset_tokens (user_id, token_hash, created_at, expires_at)"
            " VALUES (?, ?, ?, ?)",
            (row["id"], _token_hash(token), now, now + ttl),
        )
        user_email, user_name = row["email"], row["name"]

    link = reset_url(token)
    if smtp_configured():
        try:
            _send_reset_email(user_email, user_name, link, ttl // 60)
        except Exception as exc:  # never surface SMTP details to the browser
            log.error("Password reset email could not be sent: %s", exc.__class__.__name__)
    elif os.environ.get("LCA_DEV_PRINT_RESET_LINK") == "1":
        print(f"[dev] Password reset link for {user_email}: {link}", flush=True)
    else:
        log.error("Password reset requested but SMTP is not configured (set SMTP_HOST and SMTP_FROM)")


class TokenStatus:
    VALID = "valid"
    INVALID = "invalid"
    EXPIRED = "expired"
    USED = "used"


def check_reset_token(token: str) -> str:
    if not token or len(token) > 200:
        return TokenStatus.INVALID
    with _connect() as conn:
        row = conn.execute("SELECT * FROM password_reset_tokens WHERE token_hash = ?",
                           (_token_hash(token),)).fetchone()
    if row is None:
        return TokenStatus.INVALID
    if row["used_at"] is not None:
        return TokenStatus.USED
    if row["expires_at"] < time.time():
        return TokenStatus.EXPIRED
    return TokenStatus.VALID


def reset_password(token: str, new_password: str) -> tuple[str, list[str]]:
    """Returns (token status, password problems). Success is (VALID, [])."""
    status = check_reset_token(token)
    if status != TokenStatus.VALID:
        return status, []
    now = time.time()
    with _connect() as conn:
        row = conn.execute(
            "SELECT t.id AS token_id, u.id AS user_id, u.email FROM password_reset_tokens t"
            " JOIN users u ON u.id = t.user_id WHERE t.token_hash = ? AND t.used_at IS NULL"
            " AND t.expires_at >= ?", (_token_hash(token), now)).fetchone()
        if row is None:  # used or expired between the check and now
            return TokenStatus.INVALID, []
        problems = password_problems(new_password, row["email"])
        if problems:
            return TokenStatus.VALID, problems
        # claim the token atomically so two concurrent submissions cannot both use it
        claimed = conn.execute(
            "UPDATE password_reset_tokens SET used_at = ? WHERE id = ? AND used_at IS NULL",
            (now, row["token_id"])).rowcount
        if claimed != 1:
            return TokenStatus.USED, []
        conn.execute("UPDATE users SET password_hash = ?, password_changed_at = ? WHERE id = ?",
                     (hash_password(new_password), now, row["user_id"]))
        # invalidate this and any other outstanding links for the account
        conn.execute("UPDATE password_reset_tokens SET used_at = ? WHERE user_id = ? AND used_at IS NULL",
                     (now, row["user_id"]))
        conn.execute("DELETE FROM auth_events WHERE kind = 'login_failed' AND key = ?",
                     (row["email"],))
    return TokenStatus.VALID, []


def reset_ttl_minutes() -> int:
    return _reset_ttl_seconds() // 60
