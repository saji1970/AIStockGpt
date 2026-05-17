#!/usr/bin/env python3
"""
Create or update a bootstrap admin user (bypasses API password rules for local/dev setup).

Usage:
  python scripts/create_admin_user.py admin@admin.com 'Admin@1234'
  python scripts/create_admin_user.py admin@admin.com 'admin@1234' --allow-weak
"""
import argparse
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv

load_dotenv()

from backend.auth import auth_manager
from backend.database import db_manager


def main():
    parser = argparse.ArgumentParser(description="Create bootstrap admin user")
    parser.add_argument("email", help="Admin email")
    parser.add_argument("password", help="Plain-text password")
    parser.add_argument("--first-name", default="Admin", help="First name")
    parser.add_argument("--last-name", default="User", help="Last name")
    parser.add_argument(
        "--allow-weak",
        action="store_true",
        help="Skip password strength check (dev only)",
    )
    args = parser.parse_args()

    email = args.email.strip().lower()
    password = args.password

    if not args.allow_weak:
        from backend.security import security_config

        if not security_config.validate_password(password):
            print(
                "Password rejected: need 8+ chars with uppercase, lowercase, number, "
                "and special (@$!%*?&). Use --allow-weak for dev-only weak passwords."
            )
            sys.exit(1)

    db_manager.ensure_admin_schema()

    existing = db_manager.get_user_by_email(email)
    hashed = auth_manager.hash_password(password)

    if existing:
        uid = existing["user_id"]
        db_manager.update_user(
            uid,
            {
                "hashed_password": hashed,
                "is_admin": True,
                "is_active": True,
            },
        )
        print(f"Updated existing user {email} (id={uid}) — password reset, is_admin=True")
        return

    username = email.split("@")[0]
    if db_manager.get_user_by_username(username):
        username = f"{username}_admin"

    user_id = f"user_{int(datetime.utcnow().timestamp())}"
    ok = db_manager.create_user(
        user_id,
        {
            "email": email,
            "hashed_password": hashed,
            "first_name": args.first_name,
            "last_name": args.last_name,
            "username": username,
            "is_active": True,
            "is_admin": True,
        },
    )
    if not ok:
        print("Failed to create user (check DATABASE_URL and logs).")
        sys.exit(1)
    print(f"Created admin user {email} (id={user_id}, username={username})")


if __name__ == "__main__":
    main()
