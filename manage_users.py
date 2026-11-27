"""Manage Live Chat Analysis accounts from the command line.

    python manage_users.py create --email analyst@example.com --name "Analyst"
    python manage_users.py list
    python manage_users.py disable --email analyst@example.com
    python manage_users.py enable --email analyst@example.com

Passwords are read with a hidden prompt, never from the command line.
"""
import argparse
import getpass
import sys
from datetime import datetime

import auth


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    create = sub.add_parser("create", help="create an account")
    create.add_argument("--email", required=True)
    create.add_argument("--name", default="")
    sub.add_parser("list", help="list accounts")
    for name in ("disable", "enable"):
        p = sub.add_parser(name, help=f"{name} an account")
        p.add_argument("--email", required=True)
    args = parser.parse_args(argv)

    auth.init_db()

    if args.command == "create":
        password = getpass.getpass("Password: ")
        if password != getpass.getpass("Repeat password: "):
            print("Passwords do not match.", file=sys.stderr)
            return 1
        try:
            auth.create_user(args.email, password, args.name)
        except ValueError as exc:
            print(exc, file=sys.stderr)
            return 1
        print(f"Created account {auth.normalise_email(args.email)}")
    elif args.command == "list":
        for u in auth.list_users():
            created = datetime.fromtimestamp(u["created_at"]).strftime("%Y-%m-%d")
            state = "active" if u["is_active"] else "disabled"
            print(f"{u['email']:<40} {u['name']:<25} {state:<9} {created}")
    else:
        if not auth.set_active(args.email, args.command == "enable"):
            print("No account with that email.", file=sys.stderr)
            return 1
        print(f"{args.command}d {auth.normalise_email(args.email)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
