import sys
sys.path.append('ml-service')
from app import app, db
from database_schema import User

with app.app_context():
    user = User.query.filter_by(email='admin@supplychain.com').first()
    if user:
        print(f"User found: {user.email}, hash: {user.password_hash}")
    else:
        print("User not found")
