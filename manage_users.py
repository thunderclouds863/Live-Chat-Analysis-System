"""Manage Live Chat Analysis accounts from the command line.

    python manage_users.py where                      # which database this command uses
    python manage_users.py create --email admin@company.com --name "Admin" --admin
    python manage_users.py list
    python manage_users.py link --email analyst@company.com     # set-password link to share
    python manage_users.py promote|demote --email analyst@company.com
    python manage_users.py disable|enable --email analyst@company.com
    python manage_users.py hash                       # value for LCA_ADMIN_PASSWORD_HASH

It reads the same settings as the app (environment variables, then
.streamlit/secrets.toml), so it always targets the app's database. Passwords are
read with a hidden prompt, never from the command line. Day-to-day account
management is also available in the app on the Team page (administrators only).
"""
import argparse
import getpass
import sys
from datetime import datetime

import auth


def _ask_password() -> str:
    password = getpass.getpass("Password: ")
    if password != getpass.getpass("Repeat password: "):
        raise ValueError("Passwords do not match.")
    problems = auth.password_problems(password)
    if problems:
        raise ValueError(" ".join(problems))
    return password


def _where() -> str:
    info = auth.storage_description()
    note = "" if info["persistent"] else "  (not persistent on this host)"
    return f"{info['backend']}: {info['location']}{note}"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("where", help="show which database is used")
    create = sub.add_parser("create", help="create an account with a password")
    create.add_argument("--email", required=True)
    create.add_argument("--name", default="")
    create.add_argument("--admin", action="store_true", help="can manage accounts in the app")
    sub.add_parser("list", help="list accounts")
    sub.add_parser("hash", help="print a password hash for LCA_ADMIN_PASSWORD_HASH")
    for name in ("disable", "enable", "promote", "demote", "link"):
        p = sub.add_parser(name)
        p.add_argument("--email", required=True)
    args = parser.parse_args(argv)

    if args.command == "hash":
        try:
            print(auth.hash_password(_ask_password()))
        except ValueError as exc:
            print(exc, file=sys.stderr)
            return 1
        return 0

    try:
        auth.init_db()
    except auth.AuthUnavailable:
        print(f"Cannot open the account database ({_where()}). Check LCA_DATABASE_URL / LCA_AUTH_DB.",
              file=sys.stderr)
        return 2
    print(f"Database: {_where()}", file=sys.stderr)

    try:
        if args.command == "where":
            print(f"{auth.count_users()} account(s)")
        elif args.command == "create":
            auth.create_user(args.email, _ask_password(), args.name, is_admin=args.admin)
            print(f"Created {'administrator' if args.admin else 'account'} {auth.normalise_email(args.email)}")
        elif args.command == "list":
            users = auth.list_users()
            if not users:
                print("No accounts yet.")
            for u in users:
                created = datetime.fromtimestamp(u["created_at"]).strftime("%Y-%m-%d")
                state = "disabled" if not u["is_active"] else ("invited" if u["pending"] else "active")
                role = "admin" if u["is_admin"] else "analyst"
                print(f"{u['email']:<40} {u['name']:<24} {role:<8} {state:<9} {created}")
        elif args.command == "link":
            user = auth.get_user_by_email(args.email)
            if not user:
                raise ValueError("No account with that email.")
            link, emailed = auth.issue_password_link(user["id"])
            print(link)
            print("Emailed to the user." if emailed else "Not emailed (SMTP not configured): share the link securely.",
                  file=sys.stderr)
        else:
            fn = {"disable": lambda e: auth.set_active(e, False), "enable": lambda e: auth.set_active(e, True),
                  "promote": lambda e: auth.set_admin(e, True), "demote": lambda e: auth.set_admin(e, False)}
            if not fn[args.command](args.email):
                raise ValueError("No account with that email.")
            print(f"{args.command}d {auth.normalise_email(args.email)}")
    except ValueError as exc:
        print(exc, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
