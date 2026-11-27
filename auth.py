"""Accounts, password hashing, password recovery and account provisioning.

Storage
    * PostgreSQL when LCA_DATABASE_URL (or DATABASE_URL) is set. Use this for any
      hosted deployment: Streamlit Community Cloud has no persistent disk, so a
      SQLite file there is wiped on every restart or redeploy.
    * Otherwise SQLite at LCA_AUTH_DB, default <app folder>/data/auth.db. The default
      is anchored to this file's folder so the app and manage_users.py always use
      the same database, whatever directory they are started from.

Settings are read from environment variables, then from Streamlit secrets
(.streamlit/secrets.toml or the Streamlit Cloud "Secrets" box):
    LCA_DATABASE_URL       postgresql://user:password@host:5432/dbname
    LCA_AUTH_DB            SQLite path (ignored when a database URL is set)
    LCA_APP_BASE_URL       Public URL of the app, used in emailed links
    LCA_RESET_TTL_MINUTES  Password reset link lifetime (default 30)
    LCA_INVITE_TTL_HOURS   Link lifetime for new accounts (default 72)
    LCA_ADMIN_EMAIL        First administrator, created only while no account exists,
    LCA_ADMIN_PASSWORD_HASH  with a hash from `python manage_users.py hash`
    LCA_ADMIN_PASSWORD     (or a plain password, discouraged)
    LCA_ADMIN_NAME
    SMTP_HOST, SMTP_PORT (587), SMTP_USERNAME, SMTP_PASSWORD, SMTP_FROM,
    SMTP_USE_SSL (0/1, default STARTTLS)
    LCA_DEV_PRINT_RESET_LINK=1  local development only: print emailed links to the console
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
from urllib.parse import urlparse

log = logging.getLogger("lca.auth")

APP_DIR = os.path.dirname(os.path.abspath(__file__))

PASSWORD_MIN_LENGTH = 10
PASSWORD_MAX_LENGTH = 128
LOGIN_MAX_FAILURES = 5          # per email within the window below
LOGIN_WINDOW_SECONDS = 15 * 60
RESET_MAX_REQUESTS = 3          # per email within the window below
RESET_WINDOW_SECONDS = 60 * 60

PURPOSE_RESET = "reset"
PURPOSE_INVITE = "invite"

_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
_SCRYPT_N, _SCRYPT_R, _SCRYPT_P = 2 ** 14, 8, 1
UNUSABLE_PASSWORD = "!"          # accounts waiting for their owner to set a password


class AuthUnavailable(RuntimeError):
    """The account database cannot be reached or initialised."""


# ------------------------------------------------------------------ settings
def setting(name: str, default: str = "") -> str:
    value = os.environ.get(name)
    if value not in (None, ""):
        return value
    try:  # Streamlit secrets, when running inside Streamlit with a secrets file
        import streamlit as st
        if name in st.secrets:
            return str(st.secrets[name])
    except Exception:
        pass
    return default


def _int_setting(name: str, default: int, minimum: int) -> int:
    try:
        return max(minimum, int(setting(name, str(default))))
    except ValueError:
        return default


def _reset_ttl_seconds() -> int:
    return _int_setting("LCA_RESET_TTL_MINUTES", 30, 5) * 60


def reset_ttl_minutes() -> int:
    return _reset_ttl_seconds() // 60


def invite_ttl_hours() -> int:
    return _int_setting("LCA_INVITE_TTL_HOURS", 72, 1)


def database_url() -> str:
    return setting("LCA_DATABASE_URL") or setting("DATABASE_URL")


def sqlite_path() -> str:
    path = setting("LCA_AUTH_DB", os.path.join("data", "auth.db"))
    return path if os.path.isabs(path) else os.path.join(APP_DIR, path)


def running_on_streamlit_cloud() -> bool:
    return os.path.isdir("/mount/src")


def storage_description() -> dict:
    """Where accounts are stored, without credentials."""
    url = database_url()
    if url:
        parsed = urlparse(url)
        return {"backend": "postgres", "persistent": True,
                "location": f"{parsed.hostname or 'localhost'}/{(parsed.path or '/').lstrip('/')}"}
    return {"backend": "sqlite", "persistent": not running_on_streamlit_cloud(), "location": sqlite_path()}


# ------------------------------------------------------------------ database
class _Conn:
    """Tiny adapter so the same SQL ('?' placeholders) runs on SQLite and PostgreSQL."""

    def __init__(self, raw, backend: str):
        self.raw, self.backend = raw, backend

    def execute(self, sql: str, params=()):
        if self.backend == "postgres":
            sql = sql.replace("?", "%s")
        cur = self.raw.cursor()
        cur.execute(sql, params)
        return cur

    def one(self, sql: str, params=()) -> Optional[dict]:
        cur = self.execute(sql, params)
        row = cur.fetchone()
        if row is None:
            return None
        return dict(zip([d[0] for d in cur.description], row))

    def all(self, sql: str, params=()) -> list[dict]:
        cur = self.execute(sql, params)
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, r)) for r in cur.fetchall()]

    def scalar(self, sql: str, params=()):
        row = self.execute(sql, params).fetchone()
        return row[0] if row else None


@contextmanager
def _connect():
    url = database_url()
    try:
        if url:
            import psycopg  # imported lazily: only needed for PostgreSQL
            raw = psycopg.connect(url, connect_timeout=10)
            conn = _Conn(raw, "postgres")
        else:
            path = sqlite_path()
            os.makedirs(os.path.dirname(path), exist_ok=True)
            raw = sqlite3.connect(path, timeout=10)
            raw.execute("PRAGMA foreign_keys = ON")
            conn = _Conn(raw, "sqlite")
    except Exception as exc:
        log.error("Account database unavailable (%s): %s", storage_description()["backend"], exc.__class__.__name__)
        raise AuthUnavailable("account database unavailable") from exc
    try:
        yield conn
        raw.commit()
    except Exception:
        raw.rollback()
        raise
    finally:
        raw.close()


_SCHEMA = {
    "sqlite": """
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT NOT NULL UNIQUE COLLATE NOCASE,
            name TEXT NOT NULL DEFAULT '',
            password_hash TEXT NOT NULL,
            is_active INTEGER NOT NULL DEFAULT 1,
            created_at REAL NOT NULL,
            password_changed_at REAL NOT NULL);
        CREATE TABLE IF NOT EXISTS password_reset_tokens (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
            token_hash TEXT NOT NULL UNIQUE,
            created_at REAL NOT NULL,
            expires_at REAL NOT NULL,
            used_at REAL);
        CREATE INDEX IF NOT EXISTS idx_reset_user ON password_reset_tokens(user_id);
        CREATE TABLE IF NOT EXISTS auth_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            kind TEXT NOT NULL, key TEXT NOT NULL, at REAL NOT NULL);
        CREATE INDEX IF NOT EXISTS idx_events ON auth_events(kind, key, at);
    """,
    "postgres": """
        CREATE TABLE IF NOT EXISTS users (
            id BIGSERIAL PRIMARY KEY,
            email TEXT NOT NULL UNIQUE,
            name TEXT NOT NULL DEFAULT '',
            password_hash TEXT NOT NULL,
            is_active INTEGER NOT NULL DEFAULT 1,
            created_at DOUBLE PRECISION NOT NULL,
            password_changed_at DOUBLE PRECISION NOT NULL);
        CREATE TABLE IF NOT EXISTS password_reset_tokens (
            id BIGSERIAL PRIMARY KEY,
            user_id BIGINT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
            token_hash TEXT NOT NULL UNIQUE,
            created_at DOUBLE PRECISION NOT NULL,
            expires_at DOUBLE PRECISION NOT NULL,
            used_at DOUBLE PRECISION);
        CREATE INDEX IF NOT EXISTS idx_reset_user ON password_reset_tokens(user_id);
        CREATE TABLE IF NOT EXISTS auth_events (
            id BIGSERIAL PRIMARY KEY,
            kind TEXT NOT NULL, key TEXT NOT NULL, at DOUBLE PRECISION NOT NULL);
        CREATE INDEX IF NOT EXISTS idx_events ON auth_events(kind, key, at);
    """,
}

# Additive migrations, applied in order when the column is missing.
_MIGRATIONS = [
    ("users", "is_admin", "INTEGER NOT NULL DEFAULT 0"),
    ("password_reset_tokens", "purpose", "TEXT NOT NULL DEFAULT 'reset'"),
]


def _columns(conn: _Conn, table: str) -> set[str]:
    if conn.backend == "sqlite":
        return {r["name"] for r in conn.all(f"PRAGMA table_info({table})")}
    return {r["column_name"] for r in conn.all(
        "SELECT column_name FROM information_schema.columns WHERE table_name = ? AND table_schema = current_schema()",
        (table,))}


def init_db() -> None:
    """Create or migrate the schema (idempotent) and seed the first admin if configured."""
    with _connect() as conn:
        for statement in _SCHEMA[conn.backend].split(";"):
            if statement.strip():
                conn.execute(statement)
        for table, column, ddl in _MIGRATIONS:
            if column not in _columns(conn, table):
                conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {ddl}")
    _seed_admin_from_settings()


def _seed_admin_from_settings() -> None:
    email = setting("LCA_ADMIN_EMAIL").strip()
    pw_hash = setting("LCA_ADMIN_PASSWORD_HASH").strip()
    password = setting("LCA_ADMIN_PASSWORD")
    if not email or not (pw_hash or password) or count_users() > 0:
        return
    try:
        if pw_hash:
            if not pw_hash.startswith("scrypt$"):
                raise ValueError("LCA_ADMIN_PASSWORD_HASH is not a hash from manage_users.py hash")
            _insert_user(email, setting("LCA_ADMIN_NAME", "Administrator"), pw_hash, is_admin=True)
        else:
            create_user(email, password, setting("LCA_ADMIN_NAME", "Administrator"), is_admin=True)
        log.info("Created the first administrator account from LCA_ADMIN_EMAIL")
    except ValueError as exc:
        # A concurrent session may have seeded it first; anything else is a config problem.
        if count_users() == 0:
            log.error("Could not create the first administrator: %s", exc)


# ---------------------------------------------------------------- passwords
def hash_password(password: str) -> str:
    salt = secrets.token_bytes(16)
    digest = hashlib.scrypt(password.encode(), salt=salt, n=_SCRYPT_N, r=_SCRYPT_R, p=_SCRYPT_P, dklen=32)
    return f"scrypt${_SCRYPT_N}${_SCRYPT_R}${_SCRYPT_P}${salt.hex()}${digest.hex()}"


def verify_password(password: str, stored: str) -> bool:
    try:
        algo, n, r, p, salt_hex, digest_hex = stored.split("$")
        if algo != "scrypt":
            return False
        digest = hashlib.scrypt(password.encode(), salt=bytes.fromhex(salt_hex), n=int(n), r=int(r),
                                p=int(p), dklen=len(digest_hex) // 2)
        return hmac.compare_digest(digest.hex(), digest_hex)
    except (ValueError, TypeError):
        return False


_DUMMY_HASH = hash_password(secrets.token_urlsafe(16))  # equal timing for unknown emails


def normalise_email(email: str) -> str:
    return (email or "").strip().lower()


def is_valid_email(email: str) -> bool:
    return bool(_EMAIL_RE.match(normalise_email(email))) and len(email) <= 254


def password_problems(password: str, email: str = "") -> list[str]:
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
_USER_COLUMNS = "id, email, name, is_active, is_admin, created_at, password_changed_at"


def count_users() -> int:
    with _connect() as conn:
        return int(conn.scalar("SELECT COUNT(*) FROM users"))


def _insert_user(email: str, name: str, password_hash: str, is_admin: bool = False) -> int:
    email = normalise_email(email)
    if not is_valid_email(email):
        raise ValueError("Enter a valid email address.")
    now = time.time()
    with _connect() as conn:
        if conn.one("SELECT id FROM users WHERE email = ?", (email,)):
            raise ValueError("An account with this email already exists.")
        try:
            return int(conn.scalar(
                "INSERT INTO users (email, name, password_hash, is_admin, created_at, password_changed_at)"
                " VALUES (?, ?, ?, ?, ?, ?) RETURNING id",
                (email, (name or "").strip(), password_hash, 1 if is_admin else 0, now, now)))
        except Exception as exc:  # unique violation from a concurrent insert
            if "unique" in str(exc).lower():
                raise ValueError("An account with this email already exists.")
            raise


def create_user(email: str, password: str, name: str = "", is_admin: bool = False) -> int:
    problems = password_problems(password, email)
    if problems:
        raise ValueError(" ".join(problems))
    return _insert_user(email, name, hash_password(password), is_admin)


def get_user(user_id: int) -> Optional[dict]:
    with _connect() as conn:
        return conn.one(f"SELECT {_USER_COLUMNS} FROM users WHERE id = ?", (user_id,))


def get_user_by_email(email: str) -> Optional[dict]:
    with _connect() as conn:
        return conn.one(f"SELECT {_USER_COLUMNS} FROM users WHERE email = ?", (normalise_email(email),))


def list_users() -> list[dict]:
    with _connect() as conn:
        return conn.all(f"SELECT {_USER_COLUMNS}, password_hash = ? AS pending FROM users ORDER BY id",
                        (UNUSABLE_PASSWORD,))


def _active_admin_count(conn: _Conn) -> int:
    return int(conn.scalar("SELECT COUNT(*) FROM users WHERE is_admin = 1 AND is_active = 1"))


def set_active(email: str, active: bool) -> bool:
    """Enable or disable an account. Refuses to disable the last active administrator."""
    with _connect() as conn:
        row = conn.one("SELECT id, is_admin, is_active FROM users WHERE email = ?", (normalise_email(email),))
        if row is None:
            return False
        if not active and row["is_admin"] and row["is_active"] and _active_admin_count(conn) <= 1:
            raise ValueError("This is the only active administrator. Make someone else an administrator first.")
        conn.execute("UPDATE users SET is_active = ? WHERE id = ?", (1 if active else 0, row["id"]))
        return True


def set_admin(email: str, admin: bool) -> bool:
    with _connect() as conn:
        row = conn.one("SELECT id, is_admin, is_active FROM users WHERE email = ?", (normalise_email(email),))
        if row is None:
            return False
        if not admin and row["is_admin"] and row["is_active"] and _active_admin_count(conn) <= 1:
            raise ValueError("This is the only active administrator. Make someone else an administrator first.")
        conn.execute("UPDATE users SET is_admin = ? WHERE id = ?", (1 if admin else 0, row["id"]))
        return True


# ------------------------------------------------------------- rate limits
def _recent_events(conn: _Conn, kind: str, key: str, window: int) -> int:
    return int(conn.scalar("SELECT COUNT(*) FROM auth_events WHERE kind = ? AND key = ? AND at > ?",
                           (kind, key, time.time() - window)))


def _record_event(conn: _Conn, kind: str, key: str) -> None:
    now = time.time()
    conn.execute("INSERT INTO auth_events (kind, key, at) VALUES (?, ?, ?)", (kind, key, now))
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
        row = conn.one(f"SELECT {_USER_COLUMNS}, password_hash FROM users WHERE email = ?", (key,))
        ok = verify_password(password, row["password_hash"] if row else _DUMMY_HASH)
        if row is None or not ok or not row["is_active"]:
            _record_event(conn, "login_failed", key)
            return LoginResult(LoginResult.INVALID)
        conn.execute("DELETE FROM auth_events WHERE kind = 'login_failed' AND key = ?", (key,))
        row.pop("password_hash")
        return LoginResult(LoginResult.OK, row)


def refresh_session(user: dict) -> Optional[dict]:
    """Fresh account data for a signed-in session, or None if the session must end
    (account disabled or deleted, or password changed since sign-in)."""
    current = get_user(user["id"])
    if not current or not current["is_active"]:
        return None
    if current["password_changed_at"] > user["password_changed_at"] + 1e-6:
        return None
    return current


def session_still_valid(user: dict) -> bool:
    return refresh_session(user) is not None


# ------------------------------------------- password links (reset / invite)
def _token_hash(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


def smtp_configured() -> bool:
    return bool(setting("SMTP_HOST") and setting("SMTP_FROM"))


def reset_url(token: str) -> str:
    base = setting("LCA_APP_BASE_URL", "http://localhost:8501").rstrip("/")
    return f"{base}/?reset_token={token}"


def _issue_token(conn: _Conn, user_id: int, purpose: str, ttl_seconds: int) -> str:
    token = secrets.token_urlsafe(32)
    now = time.time()
    # only the most recent link for an account stays valid
    conn.execute("UPDATE password_reset_tokens SET used_at = ? WHERE user_id = ? AND used_at IS NULL", (now, user_id))
    conn.execute("INSERT INTO password_reset_tokens (user_id, token_hash, purpose, created_at, expires_at)"
                 " VALUES (?, ?, ?, ?, ?)", (user_id, _token_hash(token), purpose, now, now + ttl_seconds))
    return token


def _send_email(to_email: str, subject: str, body: str) -> None:
    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = setting("SMTP_FROM")
    msg["To"] = to_email
    msg.set_content(body)
    host, port = setting("SMTP_HOST"), int(setting("SMTP_PORT", "587"))
    user, password = setting("SMTP_USERNAME"), setting("SMTP_PASSWORD")
    if setting("SMTP_USE_SSL", "0") == "1":
        server = smtplib.SMTP_SSL(host, port, timeout=15)
    else:
        server = smtplib.SMTP(host, port, timeout=15)
        server.starttls()
    with server:
        if user:
            server.login(user, password or "")
        server.send_message(msg)


def _deliver_link(to_email: str, name: str, link: str, purpose: str, ttl_text: str) -> bool:
    """Email the link. Returns True when an email was handed to the SMTP server."""
    greeting = f"Hi {name}," if name else "Hi,"
    if purpose == PURPOSE_INVITE:
        subject = "Your Live Chat Analysis account"
        body = (f"{greeting}\n\nAn account has been created for you in Live Chat Analysis.\n\n"
                f"Open this link to choose your password (valid for {ttl_text}, one use only):\n{link}\n")
    else:
        subject = "Reset your Live Chat Analysis password"
        body = (f"{greeting}\n\nWe received a request to reset the password for your Live Chat Analysis account.\n\n"
                f"Open this link to choose a new password (valid for {ttl_text}, one use only):\n{link}\n\n"
                "If you did not request this, you can ignore this email. Your password stays the same.\n")
    if smtp_configured():
        try:
            _send_email(to_email, subject, body)
            return True
        except Exception as exc:  # never surface SMTP details to the browser
            log.error("Email could not be sent: %s", exc.__class__.__name__)
            return False
    if setting("LCA_DEV_PRINT_RESET_LINK") == "1":
        print(f"[dev] {purpose} link for {to_email}: {link}", flush=True)
    else:
        log.warning("Email not sent: SMTP is not configured (set SMTP_HOST and SMTP_FROM)")
    return False


def request_password_reset(email: str) -> None:
    """Self-service reset. Returns nothing so the UI cannot reveal whether the account exists."""
    key = normalise_email(email)
    if not is_valid_email(key):
        return
    ttl = _reset_ttl_seconds()
    with _connect() as conn:
        if _recent_events(conn, "reset_requested", key, RESET_WINDOW_SECONDS) >= RESET_MAX_REQUESTS:
            log.warning("Password reset rate limit reached for an account")
            return
        _record_event(conn, "reset_requested", key)
        row = conn.one("SELECT id, email, name, is_active FROM users WHERE email = ?", (key,))
        if row is None or not row["is_active"]:
            return
        token = _issue_token(conn, row["id"], PURPOSE_RESET, ttl)
    _deliver_link(row["email"], row["name"], reset_url(token), PURPOSE_RESET, f"{ttl // 60} minutes")


def invite_user(email: str, name: str = "", is_admin: bool = False) -> tuple[str, bool]:
    """Admin action: create an account without a password and issue a set-password link.
    Returns (link, emailed). The admin can share the link themselves when email is off."""
    user_id = _insert_user(email, name, UNUSABLE_PASSWORD, is_admin)
    return issue_password_link(user_id, PURPOSE_INVITE)


def issue_password_link(user_id: int, purpose: str = PURPOSE_RESET) -> tuple[str, bool]:
    """Admin action: new set-password link for an account. Returns (link, emailed)."""
    ttl = invite_ttl_hours() * 3600 if purpose == PURPOSE_INVITE else _reset_ttl_seconds()
    ttl_text = f"{ttl // 3600} hours" if purpose == PURPOSE_INVITE else f"{ttl // 60} minutes"
    with _connect() as conn:
        row = conn.one("SELECT id, email, name, is_active FROM users WHERE id = ?", (user_id,))
        if row is None or not row["is_active"]:
            raise ValueError("This account does not exist or is disabled.")
        token = _issue_token(conn, row["id"], purpose, ttl)
    link = reset_url(token)
    return link, _deliver_link(row["email"], row["name"], link, purpose, ttl_text)


class TokenStatus:
    VALID = "valid"
    INVALID = "invalid"
    EXPIRED = "expired"
    USED = "used"


def check_reset_token(token: str) -> str:
    return token_info(token)[0]


def token_info(token: str) -> tuple[str, Optional[str]]:
    """(status, purpose) for a link token."""
    if not token or len(token) > 200:
        return TokenStatus.INVALID, None
    with _connect() as conn:
        row = conn.one("SELECT used_at, expires_at, purpose FROM password_reset_tokens WHERE token_hash = ?",
                       (_token_hash(token),))
    if row is None:
        return TokenStatus.INVALID, None
    if row["used_at"] is not None:
        return TokenStatus.USED, row["purpose"]
    if row["expires_at"] < time.time():
        return TokenStatus.EXPIRED, row["purpose"]
    return TokenStatus.VALID, row["purpose"]


def reset_password(token: str, new_password: str) -> tuple[str, list[str]]:
    """Returns (token status, password problems). Success is (VALID, [])."""
    status = check_reset_token(token)
    if status != TokenStatus.VALID:
        return status, []
    now = time.time()
    with _connect() as conn:
        row = conn.one(
            "SELECT t.id AS token_id, u.id AS user_id, u.email FROM password_reset_tokens t"
            " JOIN users u ON u.id = t.user_id WHERE t.token_hash = ? AND t.used_at IS NULL"
            " AND t.expires_at >= ? AND u.is_active = 1", (_token_hash(token), now))
        if row is None:
            return TokenStatus.INVALID, []
        problems = password_problems(new_password, row["email"])
        if problems:
            return TokenStatus.VALID, problems
        claimed = conn.execute("UPDATE password_reset_tokens SET used_at = ? WHERE id = ? AND used_at IS NULL",
                               (now, row["token_id"])).rowcount
        if claimed != 1:
            return TokenStatus.USED, []
        conn.execute("UPDATE users SET password_hash = ?, password_changed_at = ? WHERE id = ?",
                     (hash_password(new_password), now, row["user_id"]))
        conn.execute("UPDATE password_reset_tokens SET used_at = ? WHERE user_id = ? AND used_at IS NULL",
                     (now, row["user_id"]))
        conn.execute("DELETE FROM auth_events WHERE kind = 'login_failed' AND key = ?", (row["email"],))
    return TokenStatus.VALID, []
