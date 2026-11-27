# Live Chat Analysis

Internal tool for the live chat QA team. Upload a live chat conversation export and the complaint
list for the same period; the app classifies each ticket (normal, serious, complaint), finds the
first and final operator replies, measures lead times, detects customers who left, and produces a
performance rating and an Excel report.

## Run locally

```bash
pip install -r requirements.txt
python manage_users.py create --email you@company.com --name "Your Name"   # prompts for a password
streamlit run app.py
```

Sign in at http://localhost:8501. To try it without real data, turn on **Use sample data**
(40 synthetic tickets in `sample_data/`).

`python main.py` runs the same analysis in batch mode on `data/raw_conversation.xlsx` and writes
results to `output/`.

## Accounts

Every page requires sign-in. Accounts live in a SQLite file (`data/auth.db` by default; set
`LCA_AUTH_DB` to move it). Passwords are hashed with scrypt.

```bash
python manage_users.py create --email analyst@company.com --name "Analyst"
python manage_users.py list
python manage_users.py disable --email analyst@company.com   # ends their active sessions
python manage_users.py enable  --email analyst@company.com
```

On hosts without a shell (for example Streamlit Community Cloud), set `LCA_ADMIN_EMAIL` and
`LCA_ADMIN_PASSWORD` as secrets/environment variables: the first account is created from them when
no account exists yet. Change that password afterwards with **Forgot password?**.

> Streamlit Community Cloud has no persistent disk. Accounts and password changes are lost when the
> app restarts there, so for real use point `LCA_AUTH_DB` at persistent storage or deploy on a
> server with a persistent volume.

Failed sign-ins are limited to 5 per email per 15 minutes.

## Password recovery

**Forgot password?** on the sign-in page emails a single-use link valid for 30 minutes
(`LCA_RESET_TTL_MINUTES`). Only a SHA-256 hash of the token is stored, requesting a new link
invalidates older ones, and the page shows the same message whether or not the email has an
account. Requests are limited to 3 per email per hour.

Email is sent over SMTP. Configure:

| Variable | Example | Notes |
|---|---|---|
| `SMTP_HOST` | `smtp.office365.com` | required |
| `SMTP_PORT` | `587` | default 587 |
| `SMTP_USERNAME` / `SMTP_PASSWORD` | | if the server needs login |
| `SMTP_FROM` | `no-reply@company.com` | required |
| `SMTP_USE_SSL` | `1` | use implicit TLS (port 465); default is STARTTLS |
| `LCA_APP_BASE_URL` | `https://chat-analysis.company.com` | used to build the link; default `http://localhost:8501` |

Without SMTP the request is accepted but no email is sent (the server log says so). For local
development only, `LCA_DEV_PRINT_RESET_LINK=1` prints the link to the server console instead.

## Data handling

Uploaded files are processed for your session and are not kept on the server. The complaint list
and the Excel report are written to temporary files only while the analysis runs and are deleted
right after; the report is served to the browser from memory.

## Tests

```bash
python -m unittest discover -s tests
```
