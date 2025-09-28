import sys
sys.path.append('ml-service')
from app import app, db
from database_schema import User

with app.app_context():
    users = User.query.all()
    print("All users:")
    for user in users:
        print(f"Email: {user.email}")
        print(f"Name: {user.name}")
        print(f"Hash: {user.password_hash}")
        print(f"Two factor enabled: {user.two_factor_enabled}")
        print(f"Email verified: {user.email_verified}")
        print(f"Failed attempts: {user.failed_login_attempts}")
        print(f"Account locked: {user.account_locked}")
        if user.lockout_until:
            print(f"Lockout until: {user.lockout_until}")
        print("---")
