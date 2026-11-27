"""Tests for auth.py (run: python -m unittest discover -s tests)."""
import os
import sqlite3
import sys
import tempfile
import time
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import auth  # noqa: E402

PW = "correct horse battery"


class AuthTestCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.env = mock.patch.dict(os.environ, {"LCA_AUTH_DB": os.path.join(self.tmp.name, "a.db")}, clear=False)
        self.env.start()
        for k in ("SMTP_HOST", "SMTP_FROM", "LCA_ADMIN_EMAIL", "LCA_ADMIN_PASSWORD", "LCA_DEV_PRINT_RESET_LINK"):
            os.environ.pop(k, None)
        auth.init_db()
        auth.create_user("Analyst@Example.com", PW, "Analyst")

    def tearDown(self):
        self.env.stop()
        self.tmp.cleanup()

    def _token_for(self, email):
        """Capture the token by intercepting the reset URL builder."""
        captured = {}
        real = auth.reset_url

        def spy(token):
            captured["token"] = token
            return real(token)
        with mock.patch.object(auth, "reset_url", side_effect=spy):
            auth.request_password_reset(email)
        return captured.get("token")

    # ---------------------------------------------------------- passwords
    def test_hash_is_salted_and_verifies(self):
        h1, h2 = auth.hash_password(PW), auth.hash_password(PW)
        self.assertNotEqual(h1, h2)
        self.assertTrue(auth.verify_password(PW, h1))
        self.assertFalse(auth.verify_password(PW + "x", h1))
        self.assertFalse(auth.verify_password(PW, "garbage"))

    def test_password_rules(self):
        self.assertTrue(auth.password_problems("short"))
        self.assertTrue(auth.password_problems("aaaaaaaaaaaa"))
        self.assertTrue(auth.password_problems("a@b.co12345", "a@b.co12345"))
        self.assertEqual(auth.password_problems(PW), [])

    # -------------------------------------------------------------- login
    def test_login_success_is_case_insensitive(self):
        self.assertEqual(auth.authenticate("analyst@example.com", PW).status, "ok")
        self.assertEqual(auth.authenticate(" ANALYST@example.com ", PW).status, "ok")

    def test_login_wrong_password_and_unknown_user_look_the_same(self):
        self.assertEqual(auth.authenticate("analyst@example.com", "nope nope nope").status, "invalid")
        self.assertEqual(auth.authenticate("ghost@example.com", "nope nope nope").status, "invalid")

    def test_lockout_after_repeated_failures(self):
        for _ in range(auth.LOGIN_MAX_FAILURES):
            auth.authenticate("analyst@example.com", "wrong password 1")
        self.assertEqual(auth.authenticate("analyst@example.com", PW).status, "locked")

    def test_disabled_account_cannot_sign_in(self):
        auth.set_active("analyst@example.com", False)
        self.assertEqual(auth.authenticate("analyst@example.com", PW).status, "invalid")

    def test_duplicate_account_rejected(self):
        with self.assertRaises(ValueError):
            auth.create_user("analyst@example.com", PW)

    # ----------------------------------------------------------- recovery
    def test_full_reset_flow(self):
        user = auth.authenticate("analyst@example.com", PW).user
        token = self._token_for("analyst@example.com")
        self.assertIsNotNone(token)
        self.assertEqual(auth.check_reset_token(token), auth.TokenStatus.VALID)
        # only the hash is stored
        c = sqlite3.connect(os.environ["LCA_AUTH_DB"])
        stored = [r[0] for r in c.execute("SELECT token_hash FROM password_reset_tokens")]
        c.close()
        self.assertNotIn(token, stored)

        status, problems = auth.reset_password(token, "short")
        self.assertEqual((status, bool(problems)), (auth.TokenStatus.VALID, True))
        self.assertEqual(auth.check_reset_token(token), auth.TokenStatus.VALID, "failed validation must not burn the link")

        new_pw = "a brand new passphrase"
        self.assertEqual(auth.reset_password(token, new_pw), (auth.TokenStatus.VALID, []))
        self.assertEqual(auth.check_reset_token(token), auth.TokenStatus.USED)
        self.assertEqual(auth.reset_password(token, "another passphrase")[0], auth.TokenStatus.USED)
        self.assertEqual(auth.authenticate("analyst@example.com", PW).status, "invalid")
        self.assertEqual(auth.authenticate("analyst@example.com", new_pw).status, "ok")
        # sessions created before the change are invalidated
        self.assertFalse(auth.session_still_valid(user))

    def test_new_request_invalidates_older_link(self):
        first = self._token_for("analyst@example.com")
        second = self._token_for("analyst@example.com")
        self.assertEqual(auth.check_reset_token(first), auth.TokenStatus.USED)
        self.assertEqual(auth.check_reset_token(second), auth.TokenStatus.VALID)

    def test_expired_token(self):
        token = self._token_for("analyst@example.com")
        with mock.patch.object(auth.time, "time", return_value=time.time() + 31 * 60):
            self.assertEqual(auth.check_reset_token(token), auth.TokenStatus.EXPIRED)
            self.assertEqual(auth.reset_password(token, "a brand new passphrase")[0], auth.TokenStatus.EXPIRED)

    def test_invalid_token(self):
        self.assertEqual(auth.check_reset_token("not-a-token"), auth.TokenStatus.INVALID)
        self.assertEqual(auth.check_reset_token(""), auth.TokenStatus.INVALID)

    def test_unknown_email_creates_no_token(self):
        self.assertIsNone(self._token_for("ghost@example.com"))

    def test_reset_requests_are_rate_limited(self):
        tokens = [self._token_for("analyst@example.com") for _ in range(auth.RESET_MAX_REQUESTS + 1)]
        self.assertIsNone(tokens[-1])

    def test_email_sent_via_smtp_when_configured(self):
        with mock.patch.dict(os.environ, {"SMTP_HOST": "smtp.test", "SMTP_FROM": "noreply@test", "SMTP_PORT": "587"}), \
                mock.patch.object(auth.smtplib, "SMTP") as smtp:
            auth.request_password_reset("analyst@example.com")
        server = smtp.return_value
        server.starttls.assert_called_once()
        server.send_message.assert_called_once()
        msg = server.send_message.call_args[0][0]
        self.assertEqual(msg["To"], "analyst@example.com")
        self.assertIn("reset_token=", msg.get_content())

    def test_smtp_failure_is_not_raised(self):
        with mock.patch.dict(os.environ, {"SMTP_HOST": "smtp.test", "SMTP_FROM": "noreply@test"}), \
                mock.patch.object(auth.smtplib, "SMTP", side_effect=OSError("down")):
            auth.request_password_reset("analyst@example.com")  # must not raise

    def test_seed_admin_only_when_empty(self):
        with mock.patch.dict(os.environ, {"LCA_ADMIN_EMAIL": "admin@example.com", "LCA_ADMIN_PASSWORD": PW}):
            auth.init_db()
        self.assertEqual([u["email"] for u in auth.list_users()], ["analyst@example.com"])


if __name__ == "__main__":
    unittest.main()
