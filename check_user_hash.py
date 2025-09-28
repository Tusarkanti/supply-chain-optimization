import sys
sys.path.append('ml-service')
from app import app, db
from database_schema import User

with app.app_context():
    user = User.query.filter_by(email='admin@supplychain.com').first()
    if user:
        print(f"User found: {user.email}")
        print(f"Password hash: {user.password_hash}")
        from werkzeug.security import check_password_hash
        print(f"Password check with 'admin': {check_password_hash(user.password_hash, 'admin')}")
    else:
        print("Admin user not found.")
    
    demo_user = User.query.filter_by(email='demo@supplychain.com').first()
    if demo_user:
        print(f"\nDemo user found: {demo_user.email}")
        print(f"Demo password hash: {demo_user.password_hash}")
        from werkzeug.security import check_password_hash
        print(f"Password check with 'demo123': {check_password_hash(demo_user.password_hash, 'demo123')}")
    else:
        print("Demo user not found.")
