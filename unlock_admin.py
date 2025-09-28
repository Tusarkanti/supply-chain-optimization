import sys
sys.path.append('ml-service')
from app import app, db
from database_schema import User

with app.app_context():
    user = User.query.filter_by(email='admin@supplychain.com').first()
    if user:
        print(f"Before: locked={user.account_locked}, attempts={user.failed_login_attempts}, lockout={user.lockout_until}")
        user.failed_login_attempts = 0
        user.account_locked = False
        user.lockout_until = None
        db.session.commit()
        print("Admin user unlocked.")
        print(f"After: locked={user.account_locked}, attempts={user.failed_login_attempts}, lockout={user.lockout_until}")
    else:
        print("Admin user not found.")
