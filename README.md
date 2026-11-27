# Live Chat Analysis

Internal tool for the live chat QA team. Upload a live chat conversation export and the complaint
list for the same period; the app classifies each ticket (normal, serious, complaint), finds the
first and final operator replies, measures lead times, detects customers who left, and produces a
performance rating and an Excel report.

## Run locally

```bash
pip install -r requirements.txt
python manage_users.py create --email you@company.com --name "Your Name" --admin   # prompts for a password
streamlit run app.py
```

Sign in at http://localhost:8501. To try it without real data, turn on **Use sample data**
(40 synthetic tickets in `sample_data/`).

`python main.py` runs the same analysis in batch mode on `data/raw_conversation.xlsx` and writes
results to `output/`.

## Where accounts are stored

| Setting | Storage | Use it for |
|---|---|---|
| `LCA_DATABASE_URL` (or `DATABASE_URL`) | PostgreSQL, e.g. Neon, Supabase, RDS | any hosted deployment |
| `LCA_AUTH_DB` | SQLite file at that path | a server with a persistent disk |
| neither | SQLite at `data/auth.db` next to `app.py` | local development |

Settings are read from environment variables first, then `.streamlit/secrets.toml` (see
`.streamlit/secrets.toml.example`). The app and `manage_users.py` read the same settings, so they
always use the same database; `python manage_users.py where` prints which one.

> **Streamlit Community Cloud has no persistent disk and no shell.** A SQLite file there is wiped
> on every restart or redeploy, which is why the app can come back with no accounts. Set
> `LCA_DATABASE_URL` in the app's secrets for any real use. The app logs a warning and the Team
> page shows one while it runs on storage that isn't persistent.

## First administrator

When the database has no accounts, the sign-in page tells people to contact their administrator
and the server log explains what to configure. To create the first administrator:

- **Hosted (no shell):** run `python manage_users.py hash` on your own machine, then set
  `LCA_ADMIN_EMAIL`, `LCA_ADMIN_NAME` and `LCA_ADMIN_PASSWORD_HASH` in the app's secrets. The app
  creates that administrator on start, only while no account exists, so the setting can't add or
  overwrite accounts later. `LCA_ADMIN_PASSWORD` (plain text) also works but the hash is preferred.
- **With a shell:** `python manage_users.py create --email admin@company.com --name "Admin" --admin`.

## Managing accounts

Administrators manage people on the **Team** page in the app: add a person (they get a link to
set their own password, valid 72 hours, `LCA_INVITE_TTL_HOURS`), send a new password link, change
the role, or disable an account. Disabling ends the person's open sessions. The last active
administrator can't be removed or disabled. When email isn't configured, the link is shown to the
administrator to share privately.

The same actions are available from the command line:

```bash
python manage_users.py where
python manage_users.py list
python manage_users.py create --email analyst@company.com --name "Analyst"
python manage_users.py link --email analyst@company.com      # new set-password link
python manage_users.py promote|demote --email analyst@company.com
python manage_users.py disable|enable --email analyst@company.com
```

Passwords are hashed with scrypt. Failed sign-ins are limited to 5 per email per 15 minutes.

## Password recovery

**Forgot password?** on the sign-in page emails a single-use link valid for 30 minutes
(`LCA_RESET_TTL_MINUTES`). Only a SHA-256 hash of the token is stored, requesting a new link
invalidates older ones, and the page shows the same message whether or not the email has an
account. Requests are limited to 3 per email per hour. When email isn't configured, the page says
so and asks the person to contact their administrator, who can issue a link from the Team page.

Email is sent over SMTP. Configure:

| Variable | Example | Notes |
|---|---|---|
| `SMTP_HOST` | `smtp.office365.com` | required |
| `SMTP_PORT` | `587` | default 587 |
| `SMTP_USERNAME` / `SMTP_PASSWORD` | | if the server needs login |
| `SMTP_FROM` | `no-reply@company.com` | required |
| `SMTP_USE_SSL` | `1` | use implicit TLS (port 465); default is STARTTLS |
| `LCA_APP_BASE_URL` | `https://chat-analysis.company.com` | used to build the link; default `http://localhost:8501` |

For local development only, `LCA_DEV_PRINT_RESET_LINK=1` prints links to the server console.

## Data handling

Uploaded files are processed for your session and are not kept on the server. The complaint list
and the Excel report are written to temporary files only while the analysis runs and are deleted
right after; the report is served to the browser from memory.

## Tests

```bash
python -m unittest discover -s tests
LCA_TEST_DATABASE_URL=postgresql://... python -m unittest discover -s tests   # also run the PostgreSQL tests
```

`LCA_TEST_DATABASE_URL` must point at a throwaway database: the PostgreSQL tests drop the auth
tables.
