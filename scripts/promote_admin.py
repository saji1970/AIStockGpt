#!/usr/bin/env python3
"""Promote a user to admin by email. Usage: python scripts/promote_admin.py user@example.com"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv

load_dotenv()

from backend.database import db_manager


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/promote_admin.py <email> [email2 ...]")
        sys.exit(1)
    db_manager.ensure_admin_schema()
    emails = sys.argv[1:]
    n = db_manager.promote_admin_by_emails(emails)
    print(f"Promoted {n} user(s): {', '.join(emails)}")


if __name__ == "__main__":
    main()
