"""Tests for auth.py.

    python -m unittest discover -s tests

Runs against SQLite. Set LCA_TEST_DATABASE_URL to a throwaway PostgreSQL database
to run the same tests against PostgreSQL as well (its tables are dropped).
"""
import os
import sqlite3
import subprocess
import sys
import tempfile
import time
import unittest
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import auth  # noqa: E402

PW = "correct horse battery"
CLEAR = ("SMTP_HOST", "SMTP_FROM", "LCA_ADMIN_EMAIL", "LCA_ADMIN_PASSWORD", "LCA_ADMIN_PASSWORD_HASH",
         "LCA_DEV_PRINT_RESET_LINK", "LCA_DATABASE_URL", "DATABASE_URL", "LCA_AUTH_DB")


class _Base:
    """Shared tests; subclasses choose the backend."""

    def env(self):
        raise NotImplementedError

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        saved = {k: os.environ.pop(k) for k in CLEAR if k in os.environ}
        self.addCleanup(os.environ.update, saved)
        self.patch = mock.patch.dict(os.environ, self.env())
        self.patch.start()
        self.addCleanup(self.patch.stop)
        self.reset_backend()
        auth.init_db()

    def tearDown(self):
        self.tmp.cleanup()

    def reset_backend(self):
        pass

    def make_user(self, email="Analyst@Example.com", admin=False):
        return auth.create_user(email, PW, "Analyst", is_admin=admin)

    def _token_for(self, email):
        captured = {}
        real = auth.reset_url

        def spy(token):
            captured["token"] = token
            return real(token)
        with mock.patch.object(auth, "reset_url", side_effect=spy):
            auth.request_password_reset(email)
        return captured.get("token")

    # ------------------------------------------------ zero users / provisioning
    def test_fresh_database_has_zero_users(self):
        self.assertEqual(auth.count_users(), 0)

    def test_init_is_idempotent(self):
        self.make_user()
        auth.init_db()
        auth.init_db()
        self.assertEqual(auth.count_users(), 1)

    def test_seed_admin_from_hash_once(self):
        pw_hash = auth.hash_password(PW)
        with mock.patch.dict(os.environ, {"LCA_ADMIN_EMAIL": "Admin@Example.com", "LCA_ADMIN_PASSWORD_HASH": pw_hash}):
            auth.init_db()
            auth.init_db()
        users = auth.list_users()
        self.assertEqual([(u["email"], bool(u["is_admin"])) for u in users], [("admin@example.com", True)])
        self.assertEqual(auth.authenticate("admin@example.com", PW).status, "ok")

    def test_seed_ignored_when_accounts_exist(self):
        self.make_user()
        with mock.patch.dict(os.environ, {"LCA_ADMIN_EMAIL": "admin@example.com", "LCA_ADMIN_PASSWORD": PW}):
            auth.init_db()
        self.assertEqual([u["email"] for u in auth.list_users()], ["analyst@example.com"])

    def test_seed_rejects_non_hash_value(self):
        with mock.patch.dict(os.environ, {"LCA_ADMIN_EMAIL": "admin@example.com", "LCA_ADMIN_PASSWORD_HASH": PW}):
            auth.init_db()
        self.assertEqual(auth.count_users(), 0)

    def test_seed_weak_plain_password_rejected(self):
        with mock.patch.dict(os.environ, {"LCA_ADMIN_EMAIL": "admin@example.com", "LCA_ADMIN_PASSWORD": "short"}):
            auth.init_db()
        self.assertEqual(auth.count_users(), 0)

    def test_duplicate_and_invalid_accounts_rejected(self):
        self.make_user()
        with self.assertRaises(ValueError):
            self.make_user("analyst@EXAMPLE.com")
        with self.assertRaises(ValueError):
            auth.create_user("not-an-email", PW)
        with self.assertRaises(ValueError):
            auth.create_user("x@example.com", "short")
        self.assertEqual(auth.count_users(), 1)

    def test_invite_flow(self):
        admin_id = self.make_user("admin@example.com", admin=True)
        self.assertTrue(admin_id)
        link, emailed = auth.invite_user("new@example.com", "New Person")
        self.assertFalse(emailed)
        token = link.split("reset_token=")[1]
        self.assertEqual(auth.token_info(token), (auth.TokenStatus.VALID, auth.PURPOSE_INVITE))
        # an invited account cannot sign in before setting a password
        self.assertEqual(auth.authenticate("new@example.com", "").status, "invalid")
        pending = {u["email"]: bool(u["pending"]) for u in auth.list_users()}
        self.assertTrue(pending["new@example.com"])
        self.assertEqual(auth.reset_password(token, "my own passphrase"), (auth.TokenStatus.VALID, []))
        self.assertEqual(auth.authenticate("new@example.com", "my own passphrase").status, "ok")
        with self.assertRaises(ValueError):
            auth.invite_user("new@example.com")

    def test_last_admin_cannot_be_removed_or_disabled(self):
        self.make_user("admin@example.com", admin=True)
        with self.assertRaises(ValueError):
            auth.set_admin("admin@example.com", False)
        with self.assertRaises(ValueError):
            auth.set_active("admin@example.com", False)
        self.make_user("second@example.com", admin=True)
        self.assertTrue(auth.set_admin("admin@example.com", False))

    def test_role_change_visible_to_session_and_disable_ends_it(self):
        self.make_user("admin@example.com", admin=True)
        self.make_user()
        user = auth.authenticate("analyst@example.com", PW).user
        self.assertFalse(user["is_admin"])
        auth.set_admin("analyst@example.com", True)
        self.assertTrue(auth.refresh_session(user)["is_admin"])
        auth.set_active("analyst@example.com", False)
        self.assertIsNone(auth.refresh_session(user))

    def test_disabled_user_link_cannot_be_used(self):
        self.make_user("admin@example.com", admin=True)
        uid = self.make_user()
        link, _ = auth.issue_password_link(uid)
        auth.set_active("analyst@example.com", False)
        self.assertNotEqual(auth.reset_password(link.split("=")[1], "another passphrase")[0], auth.TokenStatus.VALID)
        with self.assertRaises(ValueError):
            auth.issue_password_link(uid)

    # ------------------------------------------------------------- login
    def test_hash_is_salted_and_verifies(self):
        h1, h2 = auth.hash_password(PW), auth.hash_password(PW)
        self.assertNotEqual(h1, h2)
        self.assertTrue(auth.verify_password(PW, h1))
        self.assertFalse(auth.verify_password(PW + "x", h1))
        self.assertFalse(auth.verify_password(PW, auth.UNUSABLE_PASSWORD))

    def test_login(self):
        self.make_user()
        self.assertEqual(auth.authenticate(" ANALYST@example.com ", PW).status, "ok")
        self.assertNotIn("password_hash", auth.authenticate("analyst@example.com", PW).user)
        self.assertEqual(auth.authenticate("analyst@example.com", "nope nope nope").status, "invalid")
        self.assertEqual(auth.authenticate("ghost@example.com", "nope nope nope").status, "invalid")

    def test_lockout(self):
        self.make_user()
        for _ in range(auth.LOGIN_MAX_FAILURES):
            auth.authenticate("analyst@example.com", "wrong password 1")
        self.assertEqual(auth.authenticate("analyst@example.com", PW).status, "locked")

    def test_sql_injection_in_email_is_inert(self):
        self.make_user()
        self.assertEqual(auth.authenticate("' OR '1'='1", PW).status, "invalid")
        self.assertEqual(auth.authenticate("analyst@example.com' --", PW).status, "invalid")
        self.assertEqual(auth.count_users(), 1)

    # ----------------------------------------------------------- recovery
    def test_full_reset_flow(self):
        self.make_user()
        user = auth.authenticate("analyst@example.com", PW).user
        token = self._token_for("analyst@example.com")
        self.assertEqual(auth.check_reset_token(token), auth.TokenStatus.VALID)
        self.assertEqual(auth.reset_password(token, "short")[0], auth.TokenStatus.VALID)
        self.assertEqual(auth.check_reset_token(token), auth.TokenStatus.VALID, "failed validation keeps the link")
        self.assertEqual(auth.reset_password(token, "a brand new passphrase"), (auth.TokenStatus.VALID, []))
        self.assertEqual(auth.check_reset_token(token), auth.TokenStatus.USED)
        self.assertEqual(auth.authenticate("analyst@example.com", PW).status, "invalid")
        self.assertEqual(auth.authenticate("analyst@example.com", "a brand new passphrase").status, "ok")
        self.assertFalse(auth.session_still_valid(user))

    def test_new_request_invalidates_older_link(self):
        self.make_user()
        first = self._token_for("analyst@example.com")
        second = self._token_for("analyst@example.com")
        self.assertEqual(auth.check_reset_token(first), auth.TokenStatus.USED)
        self.assertEqual(auth.check_reset_token(second), auth.TokenStatus.VALID)

    def test_expired_and_invalid_tokens(self):
        self.make_user()
        token = self._token_for("analyst@example.com")
        with mock.patch.object(auth.time, "time", return_value=time.time() + 31 * 60):
            self.assertEqual(auth.reset_password(token, "a brand new passphrase")[0], auth.TokenStatus.EXPIRED)
        self.assertEqual(auth.check_reset_token("not-a-token"), auth.TokenStatus.INVALID)

    def test_unknown_email_and_rate_limit(self):
        self.make_user()
        self.assertIsNone(self._token_for("ghost@example.com"))
        tokens = [self._token_for("analyst@example.com") for _ in range(auth.RESET_MAX_REQUESTS + 1)]
        self.assertIsNone(tokens[-1])

    def test_email_sent_via_smtp_when_configured(self):
        self.make_user()
        with mock.patch.dict(os.environ, {"SMTP_HOST": "smtp.test", "SMTP_FROM": "noreply@test"}), \
                mock.patch.object(auth.smtplib, "SMTP") as smtp:
            auth.request_password_reset("analyst@example.com")
        smtp.return_value.send_message.assert_called_once()
        self.assertIn("reset_token=", smtp.return_value.send_message.call_args[0][0].get_content())

    def test_smtp_failure_is_not_raised(self):
        self.make_user()
        with mock.patch.dict(os.environ, {"SMTP_HOST": "smtp.test", "SMTP_FROM": "noreply@test"}), \
                mock.patch.object(auth.smtplib, "SMTP", side_effect=OSError("down")):
            auth.request_password_reset("analyst@example.com")


class SQLiteAuthTests(_Base, unittest.TestCase):
    def env(self):
        return {"LCA_AUTH_DB": os.path.join(self.tmp.name, "a.db")}

    def test_default_path_is_independent_of_working_directory(self):
        os.environ.pop("LCA_AUTH_DB")
        here = auth.sqlite_path()
        cwd = os.getcwd()
        try:
            os.chdir(self.tmp.name)
            self.assertEqual(auth.sqlite_path(), here)
        finally:
            os.chdir(cwd)
        self.assertEqual(here, os.path.join(ROOT, "data", "auth.db"))

    def test_migrates_database_from_previous_version(self):
        path = os.path.join(self.tmp.name, "old.db")
        c = sqlite3.connect(path)
        c.executescript("""
            CREATE TABLE users (id INTEGER PRIMARY KEY AUTOINCREMENT, email TEXT NOT NULL UNIQUE COLLATE NOCASE,
              name TEXT NOT NULL DEFAULT '', password_hash TEXT NOT NULL, is_active INTEGER NOT NULL DEFAULT 1,
              created_at REAL NOT NULL, password_changed_at REAL NOT NULL);
            CREATE TABLE password_reset_tokens (id INTEGER PRIMARY KEY AUTOINCREMENT, user_id INTEGER NOT NULL,
              token_hash TEXT NOT NULL UNIQUE, created_at REAL NOT NULL, expires_at REAL NOT NULL, used_at REAL);
            CREATE TABLE auth_events (id INTEGER PRIMARY KEY AUTOINCREMENT, kind TEXT NOT NULL, key TEXT NOT NULL,
              at REAL NOT NULL);""")
        c.execute("INSERT INTO users (email, name, password_hash, created_at, password_changed_at) VALUES (?,?,?,?,?)",
                  ("old@example.com", "Old", auth.hash_password(PW), 1.0, 1.0))
        c.commit()
        c.close()
        with mock.patch.dict(os.environ, {"LCA_AUTH_DB": path}):
            auth.init_db()
            self.assertEqual(auth.authenticate("old@example.com", PW).status, "ok")
            self.assertFalse(auth.list_users()[0]["is_admin"])

    def test_cli_and_app_share_database(self):
        env = dict(os.environ, LCA_AUTH_DB=os.path.join(self.tmp.name, "a.db"))
        out = subprocess.run([sys.executable, os.path.join(ROOT, "manage_users.py"), "create", "--email",
                              "cli@example.com", "--admin"], input=f"{PW}\n{PW}\n", text=True,
                             capture_output=True, cwd=self.tmp.name, env=env)
        self.assertEqual(out.returncode, 0, out.stderr)
        self.assertEqual(auth.authenticate("cli@example.com", PW).status, "ok")
        self.assertTrue(auth.get_user_by_email("cli@example.com")["is_admin"])

    def test_unavailable_database_raises_auth_unavailable(self):
        with mock.patch.dict(os.environ, {"LCA_AUTH_DB": "/proc/forbidden/auth.db"}):
            with self.assertRaises(auth.AuthUnavailable):
                auth.count_users()


PG_URL = os.environ.get("LCA_TEST_DATABASE_URL")


@unittest.skipUnless(PG_URL, "set LCA_TEST_DATABASE_URL to run PostgreSQL tests")
class PostgresAuthTests(_Base, unittest.TestCase):
    def env(self):
        return {"LCA_DATABASE_URL": PG_URL}

    def reset_backend(self):
        import psycopg
        with psycopg.connect(PG_URL) as c:
            c.execute("DROP TABLE IF EXISTS auth_events, password_reset_tokens, users CASCADE")

    def test_unreachable_server_raises_auth_unavailable(self):
        with mock.patch.dict(os.environ, {"LCA_DATABASE_URL": "postgresql://nobody@127.0.0.1:1/none"}):
            with self.assertRaises(auth.AuthUnavailable):
                auth.count_users()


if __name__ == "__main__":
    unittest.main()
