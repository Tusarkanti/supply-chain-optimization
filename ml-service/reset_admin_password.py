from app import app, db
from database_schema import User
from werkzeug.security import generate_password_hash

with app.app_context():
    user = User.query.filter_by(email='admin@supplychain.com').first()
    if user:
        user.password_hash = generate_password_hash('admin')
        db.session.commit()
        print("Admin user password reset successfully.")
    else:
        print("Admin user not found.")
