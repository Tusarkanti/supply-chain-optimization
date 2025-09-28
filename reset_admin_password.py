import sys
sys.path.append('ml-service')
from app import app, db
from database_schema import User
from passlib.hash import bcrypt

with app.app_context():
    user = User.query.filter_by(email='admin@supplychain.com').first()
    if user:
        user.password_hash = bcrypt.hash('admin')
        db.session.commit()
        print("Admin user password reset successfully.")
    else:
        print("Admin user not found.")
